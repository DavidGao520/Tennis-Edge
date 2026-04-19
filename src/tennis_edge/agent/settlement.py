"""Phase 2 Agent settlement poller.

For every AgentDecision logged in Phase 3A shadow mode, we want to know:
"if we had actually executed this recommendation, what would we have
made?" That is the single number that answers whether the agent is
worth promoting from 3A → 3B.

Approach: poll Kalshi REST for each un-settled decision's market
status. When a market has resolved (status in {"settled",
"finalized"}), compute counterfactual P&L at a fixed notional size
(default $50 to match the Phase 3C position cap) and append a
SettlementRecord joined on decision_id.

                                   every 15 min
                                        │
                                        ▼
   decisions.jsonl  ◄───────── DecisionLog.iter_decisions
        │   (all)                                │
        │                                        │
   settlements.jsonl ────────► DecisionLog.iter_settlements
        │   (already settled)                    │
        ▼                                        ▼
                    set diff: unresolved decisions
                                        │
                                        ▼
                     for each → kalshi.get_market(ticker)
                                        │
                                 ┌──────┴──────┐
                                 ▼             ▼
                             settled?        no → skip
                                 │
                                 ▼
                   ┌─ outcome = yes/no/void (from .result)
                   ├─ counterfactual_pnl(recommendation,
                   │                     market_yes_cents,
                   │                     outcome,
                   │                     notional)
                   └─ append_settlement(SettlementRecord)

Design notes:

1. SKIP decisions DO get a SettlementRecord so the join stays
   complete. realized_pnl=0, outcome matches what actually happened.
   This keeps `replay()` one-line-in, one-line-out per decision and
   lets analytics compute "what fraction of SKIPs would have been
   profitable".

2. Counterfactual P&L uses observed `market_yes_cents` at decision
   time as the fill price. That is what the plan's stale-edge
   recheck already assumes; it is the cleanest number to reason
   about even if real execution would have slipped.

3. Void markets (status settled but result empty) realize $0 at
   Kalshi (money returned). outcome="void", realized_pnl=0.

4. Idempotent: a market can only settle once. If a
   SettlementRecord already exists for a decision_id, skip. This
   lets the poller run as often as you want without duplicate
   rows or accidental double-counting.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .decisions import AgentDecision, DecisionLog, SettlementRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementConfig:
    """Tunables.

    counterfactual_notional_usd — dollars we assume would have been
                                  bet per decision. Default $50 matches
                                  the Phase 3C max position cap.
    poll_interval_s             — how often to scan. 900s = 15 min.
                                  Kalshi tennis markets settle within
                                  an hour of match end, so 15 min is
                                  plenty responsive without hammering.
    per_market_delay_s          — small sleep between REST calls to
                                  stay under the rate limit when
                                  backlog is large.
    """

    counterfactual_notional_usd: float = 50.0
    poll_interval_s: float = 900.0
    per_market_delay_s: float = 0.2


# Kalshi terminal statuses for a market. When we see one of these,
# the market will never move again and we can settle the decision.
_TERMINAL_STATUSES = frozenset({"settled", "finalized"})


# ---------------------------------------------------------------------------
# P&L math
# ---------------------------------------------------------------------------


def counterfactual_pnl(
    recommendation: str,
    market_yes_cents: int,
    market_result: str,
    notional_usd: float = 50.0,
) -> tuple[str, float]:
    """Compute counterfactual (outcome, realized_pnl) for a shadow decision.

    Kalshi contracts are $1 each. If we buy YES at P cents, each
    contract costs P/100 dollars; a YES win pays $1, a NO win pays
    $0. Symmetric for BUY_NO at (100-P) cents.

    SKIP + any outcome     → outcome mirrors market, pnl = 0
    BUY_YES + result=yes   → won,  pnl = notional * (100-P)/P
    BUY_YES + result=no    → lost, pnl = -notional
    BUY_NO  + result=no    → won,  pnl = notional * P/(100-P)
    BUY_NO  + result=yes   → lost, pnl = -notional
    any rec + result=void  → void, pnl = 0 (money returned)

    "notional" is the dollars spent, so a losing trade realizes
    exactly -notional (you lose your stake, nothing more).
    """
    if market_result not in ("yes", "no"):
        # void / empty / unknown → treat as void regardless of rec
        return "void", 0.0

    rec = recommendation.upper()

    if rec == "SKIP":
        # No counterfactual trade. Outcome still recorded so replay
        # analytics can tell "which SKIPs would have been profitable?"
        outcome = "won" if market_result == "yes" else "lost"
        return outcome, 0.0

    if rec == "BUY_YES":
        if market_yes_cents <= 0 or market_yes_cents >= 100:
            # Degenerate fill price, no meaningful counterfactual.
            return ("won" if market_result == "yes" else "lost"), 0.0
        if market_result == "yes":
            num_contracts = int(notional_usd * 100 / market_yes_cents)
            profit = num_contracts * (100 - market_yes_cents) / 100.0
            return "won", profit
        else:
            return "lost", -notional_usd

    if rec == "BUY_NO":
        no_cents = 100 - market_yes_cents
        if no_cents <= 0 or no_cents >= 100:
            return ("won" if market_result == "no" else "lost"), 0.0
        if market_result == "no":
            num_contracts = int(notional_usd * 100 / no_cents)
            profit = num_contracts * (100 - no_cents) / 100.0
            return "won", profit
        else:
            return "lost", -notional_usd

    # Unknown recommendation (validated at AgentDecision load time, so
    # this should never fire in practice). Treat as SKIP.
    logger.warning("counterfactual_pnl: unknown recommendation %r", recommendation)
    outcome = "won" if market_result == "yes" else "lost"
    return outcome, 0.0


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------


class _MarketFetcher:
    """Structural-typed market status getter.

    Accepts anything with an async `get_market(ticker) -> Market`, so
    tests can inject a fake without standing up KalshiClient.
    """

    async def get_market(self, ticker: str): ...  # pragma: no cover


class SettlementPoller:
    """Backfills outcomes for Phase 3A shadow decisions."""

    def __init__(
        self,
        log: DecisionLog,
        exchange: _MarketFetcher,
        config: SettlementConfig,
    ):
        self.log = log
        self.exchange = exchange
        self.config = config
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def poll_once(self) -> int:
        """Single scan. Returns number of new settlements written.

        Safe to call concurrently with the agent loop — both append
        to the same JSONL via DecisionLog, and append is atomic per
        line on POSIX.
        """
        already_settled = {s.decision_id for s in self.log.iter_settlements()}
        unresolved: list[AgentDecision] = [
            d for d in self.log.iter_decisions()
            if d.decision_id not in already_settled
        ]

        if not unresolved:
            return 0

        logger.info("settlement poller: %d unresolved decisions", len(unresolved))

        written = 0
        # Dedup by ticker — one decision per ticker is the common case
        # but a ticker can have multiple candidates if cooldown=0 in
        # tests. One get_market call resolves all of them.
        by_ticker: dict[str, list[AgentDecision]] = {}
        for d in unresolved:
            by_ticker.setdefault(d.ticker, []).append(d)

        for ticker, decisions in by_ticker.items():
            if self._stop.is_set():
                break
            try:
                market = await self.exchange.get_market(ticker)
            except Exception as e:
                logger.warning("settlement poller: get_market(%s) failed: %s", ticker, e)
                continue

            if market.status not in _TERMINAL_STATUSES:
                continue  # still open, try again next poll

            result = (market.result or "").lower()
            for d in decisions:
                outcome, pnl = counterfactual_pnl(
                    recommendation=d.analysis.recommendation,
                    market_yes_cents=d.market_yes_cents,
                    market_result=result,
                    notional_usd=self.config.counterfactual_notional_usd,
                )
                record = SettlementRecord(
                    ts=datetime.now(timezone.utc),
                    decision_id=d.decision_id,
                    ticker=d.ticker,
                    outcome=outcome,  # type: ignore[arg-type]
                    realized_pnl=pnl,
                    settled_at=datetime.now(timezone.utc),
                )
                self.log.append_settlement(record)
                written += 1
                logger.info(
                    "settled %s (%s) rec=%s outcome=%s pnl=$%.2f",
                    d.ticker, d.decision_id[:12], d.analysis.recommendation,
                    outcome, pnl,
                )

            # Rate-limit: modest sleep between markets to stay under
            # the REST quota even when backlog is large. Skippable if
            # the stop event is already set.
            if not self._stop.is_set() and self.config.per_market_delay_s > 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.per_market_delay_s,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        return written

    async def run(self) -> None:
        """Poll forever at `poll_interval_s`. Exits on request_stop()."""
        logger.info(
            "settlement poller started, interval=%.0fs",
            self.config.poll_interval_s,
        )
        try:
            while not self._stop.is_set():
                try:
                    await self.poll_once()
                except Exception:
                    logger.exception("settlement poller: poll_once crashed")
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.poll_interval_s,
                    )
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            logger.info("settlement poller stopped")
