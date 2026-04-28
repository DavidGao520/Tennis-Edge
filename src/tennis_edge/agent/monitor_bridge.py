"""Phase 2 v2 Monitor → Agent bridge.

Embeds the existing `EVScanner` (used by the standalone Monitor TUI)
inside the Agent daemon, runs it in a periodic loop, and emits
`MonitorSignal` objects via an async callback to whoever is listening
(in production, `AgentLoop.on_signal`).

This is the v2 replacement for v1's `MarketTickReader` DB-tail
approach. The two are mutually exclusive: v2 reads ticker prices via
Kalshi REST + the local model probability via the existing scanner,
which is the same path the standalone Monitor TUI uses.

                                                        ~5-15s/scan
                       ┌──────────────────────────────────────────┐
                       │                                          │
   KalshiClient ──REST─┤  EVScanner.analyze_market_pair  ────────►│ Opportunity
                       │  (per market: parse title,               │
                       │   resolve players, predict, edge)        │
                       │                                          │
                       └──────────────────┬───────────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │   Filter     │
                                   │ - whitelist  │
                                   │ - price band │
                                   │ - min EV     │
                                   └──────┬───────┘
                                          │
                                          ▼
                                  MonitorSignal(...)
                                          │
                                          ▼
                                  await on_signal(sig)

Why a poll loop instead of WebSocket:
  - Tick-logger already owns the WS for tennis tickers; v2 keeps that
    boundary clean (single writer, many readers).
  - Scanner needs the orderbook (bid/ask depth), which is REST-only
    in our current Kalshi client.
  - At ATP/WTA Main scale (~30-50 active markets), one scan pass
    fits in 10-15s under Kalshi's 7 req/s rate limit.

Why not just have AgentLoop poll the DB directly:
  - The DB only has ticker rows (last price); the scanner needs
    structured Market + Orderbook objects to extract player names
    and run the model. Reusing EVScanner is the DRY path.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from ..exchange.client import KalshiClient
from ..scanner import EVScanner, Opportunity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal shape — what the bridge emits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorSignal:
    """One detection event from the embedded scanner.

    Carries everything `AgentLoop.on_signal` needs to decide whether to
    enqueue a Gemini analysis. Note that we deliberately carry both
    sides of the YES/NO call (recommended_side) so the agent loop does
    not have to recompute. `prematch_ev` is the absolute edge — i.e.
    `|model_prob - market_prob|` — which is what gates the threshold.

    `category` matches `Opportunity.category` ('ATP Main', 'WTA Main',
    'ATP Challenger', etc.) and is what the whitelist filters on.
    """

    ticker: str
    player_yes: str
    player_no: str
    category: str
    market_yes_cents: int
    model_prob: float
    market_prob: float
    prematch_ev: float            # = abs(model_prob - market_prob)
    recommended_side: str         # "yes" or "no"
    detected_at: float            # time.monotonic()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Default ATP/WTA Main tour series. Kalshi ticker prefixes per
# realtime.py / scanner.py conventions.
WHITELIST_ATP_WTA_MAIN: tuple[str, ...] = (
    "KXATPMATCH",
    "KXWTAMATCH",
)

# Including challengers (Week 2+).
WHITELIST_ALL_TENNIS: tuple[str, ...] = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTAMATCH",
    "KXWTACHALLENGERMATCH",
)


@dataclass(frozen=True)
class MonitorBridgeConfig:
    """Tunables.

    series_whitelist     — which Kalshi series prefixes count.
                           Default = ATP/WTA Main only (Week 1).
    min_prematch_ev      — abs(edge) threshold below which we drop
                           the candidate before emitting. Higher than
                           the scanner default (3%) because grounded
                           Gemini is expensive — we only want to burn
                           dollars on candidates with real prematch
                           signal. 0.15 matches the v2 plan.
    price_band           — (min_cents, max_cents). Outside this band
                           the market has likely settled or is in late-
                           game state. Default 10-90c per v2 plan.
    poll_interval_s      — seconds between scan passes. ~15s leaves
                           comfortable headroom under Kalshi rate
                           limit when scanning ~30-50 Main markets.
    """

    series_whitelist: tuple[str, ...] = WHITELIST_ATP_WTA_MAIN
    min_prematch_ev: float = 0.15
    price_band: tuple[int, int] = (10, 90)
    poll_interval_s: float = 15.0


# ---------------------------------------------------------------------------
# Structural type for the scanner — keeps tests light
# ---------------------------------------------------------------------------


class _ScannerProto(Protocol):
    """Subset of EVScanner the bridge uses."""

    def analyze_market_pair(
        self,
        market_yes: Any, market_no: Any,
        orderbook_yes: Any, orderbook_no: Any,
    ) -> Opportunity | None: ...


class _ClientProto(Protocol):
    async def get_markets(
        self, series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str = "open",
    ) -> list: ...

    async def get_orderbook(self, ticker: str): ...


# Callback shape: async fn taking a signal, returning nothing.
SignalCallback = Callable[[MonitorSignal], Awaitable[None]]


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class MonitorBridge:
    """Periodic scanner runner that emits filtered signals.

    Usage:

        bridge = MonitorBridge(
            client=kalshi_client,
            scanner=ev_scanner,
            on_signal=agent_loop.on_signal,
            config=MonitorBridgeConfig(),
        )
        await bridge.run()   # blocks until request_stop()

    Single-loop design: one scan task at a time. If a scan takes longer
    than `poll_interval_s` (slow REST night), we wait for it before
    starting the next. No reentrancy.
    """

    def __init__(
        self,
        client: _ClientProto,
        scanner: _ScannerProto,
        on_signal: SignalCallback,
        config: MonitorBridgeConfig | None = None,
    ):
        self.client = client
        self.scanner = scanner
        self.on_signal = on_signal
        self.config = config or MonitorBridgeConfig()
        self._stop = asyncio.Event()
        self._scan_count = 0
        self._signal_count = 0

    def request_stop(self) -> None:
        """Stop the loop after the current scan completes."""
        self._stop.set()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "scans": self._scan_count,
            "signals_emitted": self._signal_count,
        }

    async def run(self) -> None:
        """Main entry. Polls the scanner, emits signals, sleeps, repeats."""
        logger.info(
            "monitor bridge starting whitelist=%s min_ev=%.2f band=%s interval=%.0fs",
            self.config.series_whitelist, self.config.min_prematch_ev,
            self.config.price_band, self.config.poll_interval_s,
        )
        try:
            while not self._stop.is_set():
                try:
                    await self.scan_once()
                except Exception:
                    logger.exception("monitor bridge: scan_once crashed; continuing")
                # Sleep with cancel-on-stop semantics so a `request_stop()`
                # mid-loop returns promptly without waiting the full
                # poll interval.
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.poll_interval_s,
                    )
                    break  # stop set during sleep
                except asyncio.TimeoutError:
                    continue
        finally:
            logger.info(
                "monitor bridge stopped scans=%d signals=%d",
                self._scan_count, self._signal_count,
            )

    async def scan_once(self) -> int:
        """Single pass. Returns the number of signals emitted.

        Public so the daemon and tests can drive it deterministically.
        """
        self._scan_count += 1
        markets = await self._fetch_markets()
        if not markets:
            return 0

        signals_emitted = 0
        for market in markets:
            sig = await self._analyze_one(market)
            if sig is None:
                continue
            try:
                await self.on_signal(sig)
                signals_emitted += 1
                self._signal_count += 1
            except Exception:
                # A crashing consumer must not take down the bridge.
                # Log and keep emitting; the consumer's bug is its
                # problem, not ours.
                logger.exception(
                    "monitor bridge: on_signal raised for %s",
                    sig.ticker,
                )
        return signals_emitted

    # ---- internals ----

    async def _fetch_markets(self) -> list:
        """Pull all markets matching the whitelist series prefixes."""
        out: list = []
        for series in self.config.series_whitelist:
            try:
                got = await self.client.get_markets(
                    series_ticker=series, status="open",
                )
            except Exception as e:
                # Per-series failure: log and continue with the others.
                # A flaky Kalshi response on one series should not
                # cascade into a full scan failure.
                logger.warning(
                    "monitor bridge: get_markets(%s) failed: %s", series, e,
                )
                continue
            out.extend(got)
        return out

    async def _analyze_one(self, market) -> MonitorSignal | None:
        """Run the scanner on one market, then apply v2 filters."""
        # Orderbook is best-effort: scanner falls back to last_price /
        # market.yes_bid if orderbook fetch fails.
        try:
            ob = await self.client.get_orderbook(market.ticker)
        except Exception as e:
            logger.debug(
                "monitor bridge: orderbook failed for %s: %s; continuing",
                market.ticker, e,
            )
            ob = None

        opp = self.scanner.analyze_market_pair(market, None, ob, None)
        if opp is None:
            return None

        return self._opportunity_to_signal(opp)

    def _opportunity_to_signal(self, opp: Opportunity) -> MonitorSignal | None:
        """Apply price-band + min-EV + whitelist-category filters.

        Returns None if any filter rejects.
        """
        cfg = self.config

        # Whitelist by category. EVScanner already labeled it; we
        # re-check here in case the scanner's whitelist diverges from
        # ours (it currently doesn't).
        if not _category_passes(opp.category, cfg.series_whitelist):
            return None

        # Need a price to evaluate; scanner gave us a mid_price in
        # cents (or None when no liquidity).
        if opp.mid_price is None:
            return None
        market_yes_cents = int(round(opp.mid_price))

        lo, hi = cfg.price_band
        if market_yes_cents < lo or market_yes_cents > hi:
            return None

        prematch_ev = abs(opp.edge)
        if prematch_ev < cfg.min_prematch_ev:
            return None

        return MonitorSignal(
            ticker=opp.ticker,
            player_yes=opp.player_name,
            player_no=opp.opponent_name,
            category=opp.category,
            market_yes_cents=market_yes_cents,
            model_prob=opp.model_prob,
            market_prob=opp.market_implied_prob,
            prematch_ev=prematch_ev,
            recommended_side=opp.recommended_side,
            detected_at=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _category_passes(category: str, whitelist: tuple[str, ...]) -> bool:
    """Map an Opportunity.category back to a series prefix and check.

    EVScanner's category labels:
      'ATP Main'           -> KXATPMATCH
      'ATP Challenger'     -> KXATPCHALLENGERMATCH
      'WTA Main'           -> KXWTAMATCH
      'WTA Challenger'     -> KXWTACHALLENGERMATCH
      'Other'              -> not tennis, never passes
    """
    cat = (category or "").upper().replace(" ", "")
    mapping = {
        "ATPMAIN": "KXATPMATCH",
        "ATPCHALLENGER": "KXATPCHALLENGERMATCH",
        "WTAMAIN": "KXWTAMATCH",
        "WTACHALLENGER": "KXWTACHALLENGERMATCH",
    }
    series = mapping.get(cat)
    if series is None:
        return False
    return series in whitelist
