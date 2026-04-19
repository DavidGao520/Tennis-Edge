"""Tests for agent/settlement.py — counterfactual P&L poller.

Covers:
  - counterfactual_pnl math for every (recommendation, result) cell
  - Edge prices (0c, 100c) don't blow up the math
  - SettlementPoller.poll_once:
    * skips decisions already settled (idempotency)
    * skips markets not yet in terminal status
    * writes one SettlementRecord per decision
    * multiple decisions on same ticker settle together
    * void markets (status=settled, result empty) record outcome=void
    * get_market exception is logged and doesn't crash
  - Stop event short-circuits poll_once
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from tennis_edge.agent.decisions import (
    AgentDecision,
    DecisionLog,
    EvAnalysis,
)
from tennis_edge.agent.settlement import (
    SettlementConfig,
    SettlementPoller,
    counterfactual_pnl,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeMarket:
    ticker: str
    status: str = "active"
    result: str | None = None


@dataclass
class FakeExchange:
    markets: dict[str, FakeMarket] = field(default_factory=dict)
    raise_for: set[str] = field(default_factory=set)
    get_market_calls: int = 0

    async def get_market(self, ticker: str) -> FakeMarket:
        self.get_market_calls += 1
        if ticker in self.raise_for:
            raise RuntimeError("simulated network flake")
        if ticker not in self.markets:
            return FakeMarket(ticker=ticker, status="active", result=None)
        return self.markets[ticker]


def _analysis(recommendation: str = "BUY_YES") -> EvAnalysis:
    return EvAnalysis(
        edge_estimate=0.6,
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence="medium",
        reasoning="test",
        key_factors=["a"],
    )


def _decision(
    decision_id: str = "dec-X",
    ticker: str = "KXATPMATCH-X",
    market_yes_cents: int = 15,
    recommendation: str = "BUY_YES",
) -> AgentDecision:
    return AgentDecision(
        ts=datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc),
        run_id="test-run",
        decision_id=decision_id,
        ticker=ticker,
        model_pre_match=0.53,
        market_yes_cents=market_yes_cents,
        edge_at_decision=0.38,
        llm_provider="fake",
        llm_prompt_hash="a" * 16,
        llm_raw_output="{}",
        analysis=_analysis(recommendation),
        mode="shadow",
    )


# ---------------------------------------------------------------------------
# counterfactual_pnl math
# ---------------------------------------------------------------------------


def test_cpnl_buy_yes_wins_underpriced():
    """Buy YES at 15c, YES wins. $50 notional → 333 contracts @ 15c.
    Each pays $1, profit/contract = 85c, total profit = 333 * 0.85 = $283.05."""
    outcome, pnl = counterfactual_pnl("BUY_YES", 15, "yes", notional_usd=50.0)
    assert outcome == "won"
    assert pnl == pytest.approx(333 * 0.85, abs=0.01)


def test_cpnl_buy_yes_loses():
    """Buy YES @ 15c, NO wins → you lose the full $50 stake."""
    outcome, pnl = counterfactual_pnl("BUY_YES", 15, "no", notional_usd=50.0)
    assert outcome == "lost"
    assert pnl == pytest.approx(-50.0)


def test_cpnl_buy_no_wins():
    """Buy NO at effective 85c (YES=15), NO wins. $50 / 0.85 ≈ 58 contracts.
    Each pays $1, profit/contract = 15c, total = 58 * 0.15 = $8.70."""
    outcome, pnl = counterfactual_pnl("BUY_NO", 15, "no", notional_usd=50.0)
    assert outcome == "won"
    num_contracts = int(50 * 100 / 85)  # 58
    assert pnl == pytest.approx(num_contracts * 0.15, abs=0.01)


def test_cpnl_buy_no_loses():
    outcome, pnl = counterfactual_pnl("BUY_NO", 15, "yes", notional_usd=50.0)
    assert outcome == "lost"
    assert pnl == pytest.approx(-50.0)


def test_cpnl_skip_records_outcome_zero_pnl():
    out_yes, pnl_yes = counterfactual_pnl("SKIP", 40, "yes")
    out_no, pnl_no = counterfactual_pnl("SKIP", 40, "no")
    assert (out_yes, pnl_yes) == ("won", 0.0)
    assert (out_no, pnl_no) == ("lost", 0.0)


def test_cpnl_void_market_returns_void():
    for rec in ("BUY_YES", "BUY_NO", "SKIP"):
        outcome, pnl = counterfactual_pnl(rec, 30, "", notional_usd=50.0)
        assert outcome == "void"
        assert pnl == 0.0


def test_cpnl_unknown_result_treated_as_void():
    outcome, pnl = counterfactual_pnl("BUY_YES", 30, "some-other-string")
    assert outcome == "void"
    assert pnl == 0.0


def test_cpnl_degenerate_price_returns_zero_pnl():
    """Market_yes=0 means YES is priced at 0c — nonsensical fill price.
    Record outcome but zero P&L rather than dividing by zero."""
    out, pnl = counterfactual_pnl("BUY_YES", 0, "yes")
    assert out == "won"
    assert pnl == 0.0

    out, pnl = counterfactual_pnl("BUY_YES", 100, "yes")
    assert pnl == 0.0  # YES@100c has zero profit anyway

    out, pnl = counterfactual_pnl("BUY_NO", 100, "no")  # no_cents = 0
    assert pnl == 0.0


def test_cpnl_default_notional_is_fifty():
    """Phase 3C cap is $50; default should match."""
    _, pnl_a = counterfactual_pnl("BUY_YES", 15, "no")
    _, pnl_b = counterfactual_pnl("BUY_YES", 15, "no", notional_usd=50.0)
    assert pnl_a == pnl_b


def test_cpnl_unknown_recommendation_falls_back_to_skip(caplog):
    with caplog.at_level("WARNING"):
        out, pnl = counterfactual_pnl("HOLD_FOREVER", 40, "yes")
    assert out == "won"
    assert pnl == 0.0
    assert any("unknown recommendation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SettlementPoller
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_once_writes_settlement_for_resolved_market(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1",
                                  recommendation="BUY_YES", market_yes_cents=15))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="settled", result="yes"),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    written = await poller.poll_once()
    assert written == 1

    settlements = list(log.iter_settlements())
    assert len(settlements) == 1
    assert settlements[0].decision_id == "d-1"
    assert settlements[0].outcome == "won"
    assert settlements[0].realized_pnl > 0


@pytest.mark.asyncio
async def test_poll_once_skips_non_terminal_market(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1"))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="active", result=None),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    assert await poller.poll_once() == 0
    assert list(log.iter_settlements()) == []


@pytest.mark.asyncio
async def test_poll_once_is_idempotent_for_already_settled(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1"))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="settled", result="yes"),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    assert await poller.poll_once() == 1
    # Second call: decision already in settlements → no new writes,
    # and exchange is never queried again.
    calls_before = ex.get_market_calls
    assert await poller.poll_once() == 0
    assert ex.get_market_calls == calls_before
    assert len(list(log.iter_settlements())) == 1


@pytest.mark.asyncio
async def test_poll_once_handles_get_market_exception(tmp_path, caplog):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-bad", ticker="BAD"))
    log.append_decision(_decision(decision_id="d-good", ticker="GOOD"))

    ex = FakeExchange(
        markets={"GOOD": FakeMarket(ticker="GOOD", status="settled", result="yes")},
        raise_for={"BAD"},
    )
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    with caplog.at_level("WARNING"):
        written = await poller.poll_once()

    assert written == 1  # BAD skipped, GOOD settled
    ids = [s.decision_id for s in log.iter_settlements()]
    assert ids == ["d-good"]
    assert any("get_market(BAD)" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_poll_once_void_market(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1",
                                  recommendation="BUY_YES"))

    ex = FakeExchange(markets={
        # Market settled but result empty = void (match cancelled, etc).
        "T1": FakeMarket(ticker="T1", status="settled", result=""),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    await poller.poll_once()
    s = list(log.iter_settlements())[0]
    assert s.outcome == "void"
    assert s.realized_pnl == 0.0


@pytest.mark.asyncio
async def test_poll_once_multiple_decisions_same_ticker(tmp_path):
    """If two decisions exist on one ticker (cooldown=0 in test), one
    get_market call resolves both."""
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1"))
    log.append_decision(_decision(decision_id="d-2", ticker="T1"))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="settled", result="yes"),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    written = await poller.poll_once()
    assert written == 2
    assert ex.get_market_calls == 1  # deduped by ticker


@pytest.mark.asyncio
async def test_poll_once_skip_recommendation_settles_without_pnl(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1",
                                  recommendation="SKIP"))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="settled", result="no"),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    await poller.poll_once()
    s = list(log.iter_settlements())[0]
    assert s.outcome == "lost"  # market resolved NO
    assert s.realized_pnl == 0.0  # but we SKIPped, so no counterfactual


@pytest.mark.asyncio
async def test_poll_once_finalized_also_terminal(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="d-1", ticker="T1"))

    ex = FakeExchange(markets={
        "T1": FakeMarket(ticker="T1", status="finalized", result="yes"),
    })
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))

    assert await poller.poll_once() == 1


@pytest.mark.asyncio
async def test_poll_once_no_unresolved_returns_zero(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    ex = FakeExchange()
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.0))
    assert await poller.poll_once() == 0
    assert ex.get_market_calls == 0


@pytest.mark.asyncio
async def test_poll_once_stop_short_circuits(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    # Many unresolved decisions across many tickers.
    for i in range(5):
        log.append_decision(_decision(decision_id=f"d-{i}", ticker=f"T{i}"))

    ex = FakeExchange(markets={
        f"T{i}": FakeMarket(ticker=f"T{i}", status="settled", result="yes")
        for i in range(5)
    })
    # per_market_delay_s long enough that stop after first market matters.
    poller = SettlementPoller(log, ex, SettlementConfig(per_market_delay_s=0.5))

    async def stopper():
        await asyncio.sleep(0.05)
        poller.request_stop()

    await asyncio.gather(poller.poll_once(), stopper())
    settled = list(log.iter_settlements())
    assert len(settled) < 5  # stopped before finishing
    assert len(settled) >= 1  # but at least one went through


@pytest.mark.asyncio
async def test_run_loop_exits_on_stop(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    ex = FakeExchange()
    poller = SettlementPoller(
        log, ex,
        SettlementConfig(poll_interval_s=10.0, per_market_delay_s=0.0),
    )

    async def stopper():
        await asyncio.sleep(0.05)
        poller.request_stop()

    await asyncio.wait_for(
        asyncio.gather(poller.run(), stopper()), timeout=2.0,
    )
