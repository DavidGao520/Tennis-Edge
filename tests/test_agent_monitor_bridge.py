"""Tests for agent/monitor_bridge.py.

Approach: structural-typed fakes for both KalshiClient and EVScanner.
Real tests verify (1) scanner output is correctly turned into a
MonitorSignal, (2) all three filters (whitelist, price band, min EV)
reject correctly, (3) the loop survives a crashing consumer, (4)
get_markets / get_orderbook exceptions don't tank a scan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from tennis_edge.agent.monitor_bridge import (
    MonitorBridge,
    MonitorBridgeConfig,
    MonitorSignal,
    WHITELIST_ALL_TENNIS,
    WHITELIST_ATP_WTA_MAIN,
    _category_passes,
)
from tennis_edge.scanner import Opportunity


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeMarket:
    ticker: str


@dataclass
class FakeClient:
    """Minimal KalshiClient stand-in."""

    markets_by_series: dict[str, list[FakeMarket]] = field(default_factory=dict)
    orderbook_by_ticker: dict[str, object] = field(default_factory=dict)
    raise_get_markets_for: set[str] = field(default_factory=set)
    raise_get_orderbook_for: set[str] = field(default_factory=set)
    get_markets_calls: int = 0
    get_orderbook_calls: int = 0

    async def get_markets(
        self, series_ticker=None, event_ticker=None, status="open",
    ):
        self.get_markets_calls += 1
        if series_ticker in self.raise_get_markets_for:
            raise RuntimeError("simulated get_markets failure")
        return list(self.markets_by_series.get(series_ticker, []))

    async def get_orderbook(self, ticker: str):
        self.get_orderbook_calls += 1
        if ticker in self.raise_get_orderbook_for:
            raise RuntimeError("simulated get_orderbook failure")
        return self.orderbook_by_ticker.get(ticker)


@dataclass
class FakeScanner:
    """Returns canned Opportunity objects per ticker."""

    by_ticker: dict[str, Opportunity | None] = field(default_factory=dict)
    calls: int = 0

    def analyze_market_pair(self, market_yes, market_no, ob_yes, ob_no):
        self.calls += 1
        return self.by_ticker.get(market_yes.ticker)


def _opp(
    ticker: str = "KXATPMATCH-T1",
    category: str = "ATP Main",
    edge: float = 0.20,
    mid: float | None = 40.0,
    side: str = "yes",
    model_prob: float = 0.60,
) -> Opportunity:
    return Opportunity(
        ticker=ticker,
        player_name="Holmgren",
        opponent_name="Broady",
        match_title="Will Holmgren win the Holmgren vs Broady: Round Of 32 match?",
        round_info="Round Of 32",
        category=category,
        yes_bid=int(mid) - 1 if mid else None,
        yes_ask=int(mid) + 1 if mid else None,
        mid_price=mid,
        market_implied_prob=(mid or 0) / 100.0,
        model_prob=model_prob,
        model_confidence=abs(model_prob - 0.5),
        edge=edge,
        ev_per_dollar=0.30,
        kelly_fraction=0.05,
        recommended_side=side,
    )


async def _collect(signals: list[MonitorSignal]):
    async def cb(sig: MonitorSignal):
        signals.append(sig)
    return cb


# ---------------------------------------------------------------------------
# _category_passes helper
# ---------------------------------------------------------------------------


def test_category_passes_atp_main_with_main_whitelist():
    assert _category_passes("ATP Main", WHITELIST_ATP_WTA_MAIN) is True


def test_category_passes_wta_main_with_main_whitelist():
    assert _category_passes("WTA Main", WHITELIST_ATP_WTA_MAIN) is True


def test_category_blocks_challenger_with_main_whitelist():
    assert _category_passes("ATP Challenger", WHITELIST_ATP_WTA_MAIN) is False
    assert _category_passes("WTA Challenger", WHITELIST_ATP_WTA_MAIN) is False


def test_category_passes_challenger_with_full_whitelist():
    assert _category_passes("ATP Challenger", WHITELIST_ALL_TENNIS) is True


def test_category_blocks_other():
    assert _category_passes("Other", WHITELIST_ALL_TENNIS) is False


def test_category_blocks_unknown_label():
    assert _category_passes("NBA Playoffs", WHITELIST_ALL_TENNIS) is False


def test_category_handles_empty_label():
    assert _category_passes("", WHITELIST_ATP_WTA_MAIN) is False


# ---------------------------------------------------------------------------
# scan_once happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_once_emits_signal_above_thresholds():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20, mid=40.0),
    })
    signals: list[MonitorSignal] = []

    async def on_sig(s):
        signals.append(s)

    bridge = MonitorBridge(client, scanner, on_sig, MonitorBridgeConfig())
    n = await bridge.scan_once()

    assert n == 1
    assert len(signals) == 1
    sig = signals[0]
    assert sig.ticker == "T1"
    assert sig.category == "ATP Main"
    assert sig.market_yes_cents == 40
    assert sig.prematch_ev == pytest.approx(0.20)
    assert sig.recommended_side == "yes"


# ---------------------------------------------------------------------------
# Filter rejections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_once_drops_below_min_ev():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.10),  # below default 0.15
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig, MonitorBridgeConfig())

    assert await bridge.scan_once() == 0
    assert signals == []


@pytest.mark.asyncio
async def test_scan_once_drops_outside_price_band():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1"), FakeMarket("T2")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20, mid=5.0),    # below 10
        "T2": _opp(ticker="T2", edge=0.20, mid=95.0),   # above 90
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig, MonitorBridgeConfig())

    assert await bridge.scan_once() == 0


@pytest.mark.asyncio
async def test_scan_once_drops_when_mid_price_none():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20, mid=None),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig, MonitorBridgeConfig())

    assert await bridge.scan_once() == 0


@pytest.mark.asyncio
async def test_scan_once_drops_challenger_under_main_whitelist():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        # Edge OK, price OK, but category is Challenger.
        "T1": _opp(ticker="T1", category="ATP Challenger", edge=0.20),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(
        client, scanner, on_sig,
        MonitorBridgeConfig(series_whitelist=WHITELIST_ATP_WTA_MAIN),
    )

    assert await bridge.scan_once() == 0


@pytest.mark.asyncio
async def test_scan_once_passes_challenger_under_full_whitelist():
    client = FakeClient(markets_by_series={
        "KXATPCHALLENGERMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", category="ATP Challenger", edge=0.20),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(
        client, scanner, on_sig,
        MonitorBridgeConfig(series_whitelist=WHITELIST_ALL_TENNIS),
    )

    assert await bridge.scan_once() == 1


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_once_returns_zero_when_no_markets():
    bridge = MonitorBridge(
        FakeClient(), FakeScanner(),
        on_signal=(lambda s: asyncio.sleep(0)),
        config=MonitorBridgeConfig(),
    )
    assert await bridge.scan_once() == 0


@pytest.mark.asyncio
async def test_scan_once_returns_none_opportunity_silently():
    """Scanner returns None for unparseable / unknown markets — bridge
    must not error."""
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={"T1": None})
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig)

    assert await bridge.scan_once() == 0


@pytest.mark.asyncio
async def test_scan_once_get_markets_failure_is_isolated_per_series(caplog):
    """If get_markets fails for one series, the others still run."""
    client = FakeClient(
        markets_by_series={
            "KXATPMATCH": [FakeMarket("T1")],
            "KXWTAMATCH": [FakeMarket("T2")],
        },
        raise_get_markets_for={"KXATPMATCH"},
    )
    scanner = FakeScanner(by_ticker={
        "T2": _opp(ticker="T2", category="WTA Main", edge=0.20),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig)

    with caplog.at_level("WARNING"):
        n = await bridge.scan_once()
    assert n == 1
    assert any("get_markets(KXATPMATCH)" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_scan_once_orderbook_failure_does_not_block_scanner():
    """If orderbook fails for one ticker, we still run scanner with
    ob=None. Scanner falls back to last_price / market.yes_bid."""
    client = FakeClient(
        markets_by_series={"KXATPMATCH": [FakeMarket("T1")]},
        raise_get_orderbook_for={"T1"},
    )
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig)

    assert await bridge.scan_once() == 1
    assert scanner.calls == 1


@pytest.mark.asyncio
async def test_scan_once_consumer_exception_does_not_take_down_bridge(caplog):
    """A bug in on_signal must not stop the bridge from emitting
    subsequent signals. v2's safety story depends on the bridge
    being more reliable than its consumer."""
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1"), FakeMarket("T2")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20),
        "T2": _opp(ticker="T2", edge=0.21),
    })

    seen = []

    async def crashy(sig):
        seen.append(sig.ticker)
        raise RuntimeError("consumer bug")

    bridge = MonitorBridge(client, scanner, crashy)
    with caplog.at_level("ERROR"):
        n = await bridge.scan_once()
    # n is the count of SUCCESSFUL emissions (what stats track).
    # Both consumers crashed → 0 successful. But both were ATTEMPTED,
    # which is what `seen` proves: a crashing consumer on T1 must not
    # prevent T2 from also being attempted.
    assert n == 0
    assert seen == ["T1", "T2"]
    assert any("on_signal raised for T1" in r.message for r in caplog.records)
    assert any("on_signal raised for T2" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run() loop semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_loop_exits_promptly_on_stop():
    bridge = MonitorBridge(
        FakeClient(), FakeScanner(),
        on_signal=(lambda s: asyncio.sleep(0)),
        config=MonitorBridgeConfig(poll_interval_s=10.0),
    )

    async def stopper():
        await asyncio.sleep(0.05)
        bridge.request_stop()

    await asyncio.wait_for(
        asyncio.gather(bridge.run(), stopper()),
        timeout=2.0,
    )


@pytest.mark.asyncio
async def test_run_loop_continues_after_scan_exception(caplog):
    """If scan_once() crashes, run() logs and keeps iterating, not exits."""
    bridge = MonitorBridge(
        FakeClient(), FakeScanner(),
        on_signal=(lambda s: asyncio.sleep(0)),
        config=MonitorBridgeConfig(poll_interval_s=0.05),
    )

    crashes = {"n": 0}
    original_scan = bridge.scan_once

    async def crashy_scan():
        crashes["n"] += 1
        if crashes["n"] == 1:
            raise RuntimeError("first scan blew up")
        return await original_scan()

    bridge.scan_once = crashy_scan  # type: ignore[method-assign]

    async def stopper():
        # Let two scan attempts happen (one crash, one OK), then stop.
        await asyncio.sleep(0.15)
        bridge.request_stop()

    with caplog.at_level("ERROR"):
        await asyncio.wait_for(
            asyncio.gather(bridge.run(), stopper()),
            timeout=2.0,
        )

    assert crashes["n"] >= 2
    assert any("scan_once crashed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_count_scans_and_signals():
    client = FakeClient(markets_by_series={
        "KXATPMATCH": [FakeMarket("T1")],
    })
    scanner = FakeScanner(by_ticker={
        "T1": _opp(ticker="T1", edge=0.20),
    })
    signals = []
    async def on_sig(s): signals.append(s)
    bridge = MonitorBridge(client, scanner, on_sig)

    await bridge.scan_once()
    await bridge.scan_once()
    s = bridge.stats
    assert s["scans"] == 2
    assert s["signals_emitted"] == 2
