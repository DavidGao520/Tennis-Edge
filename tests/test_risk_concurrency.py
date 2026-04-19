"""Concurrency tests for RiskManager.

Regression: previous split API (``check_trade`` then ``record_trade``)
had a TOCTOU race. Two candidates finishing LLM analysis at the same
moment could both pass the check before either recorded, producing
combined exposure greater than the configured cap.

These tests run many candidates through ``check_and_reserve`` in
parallel via ``asyncio.gather`` and assert that the final state never
violates the configured limits, and that exactly the expected number
of trades were admitted.
"""

from __future__ import annotations

import asyncio

import pytest

from tennis_edge.config import RiskConfig
from tennis_edge.strategy.risk import RiskManager
from tennis_edge.strategy.sizing import BetDecision


def _decision(ticker: str = "KXATPMATCH-TEST", amount: float = 25.0) -> BetDecision:
    return BetDecision(
        ticker=ticker,
        side="yes",
        model_prob=0.6,
        market_prob=0.5,
        edge=0.1,
        kelly_frac=0.05,
        bet_amount=amount,
        num_contracts=50,
    )


@pytest.mark.asyncio
async def test_concurrent_reserve_respects_per_market_cap():
    """Ten candidates race for a $50/market cap at $25 each. Only 2 should win."""
    config = RiskConfig(
        max_position_per_market=50.0,
        max_total_exposure=1000.0,
        daily_loss_limit=100.0,
    )
    mgr = RiskManager(config)

    async def attempt() -> tuple[bool, str]:
        return await mgr.check_and_reserve(_decision(amount=25.0))

    results = await asyncio.gather(*(attempt() for _ in range(10)))
    allowed = [r for r, _ in results if r]

    assert len(allowed) == 2, f"Expected 2 admitted, got {len(allowed)} — TOCTOU race"
    assert mgr.state.positions["KXATPMATCH-TEST"] == pytest.approx(50.0)
    assert mgr.state.total_exposure == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_concurrent_reserve_respects_total_exposure_cap():
    """Distinct tickers, total cap $200, $25 each → 8 admitted out of 20."""
    config = RiskConfig(
        max_position_per_market=1000.0,  # high so this isn't the binding cap
        max_total_exposure=200.0,
        daily_loss_limit=1000.0,
    )
    mgr = RiskManager(config)

    async def attempt(i: int) -> tuple[bool, str]:
        return await mgr.check_and_reserve(_decision(ticker=f"T{i}", amount=25.0))

    results = await asyncio.gather(*(attempt(i) for i in range(20)))
    allowed = [r for r, _ in results if r]

    assert len(allowed) == 8
    assert mgr.state.total_exposure == pytest.approx(200.0)
    assert len(mgr.state.positions) == 8


@pytest.mark.asyncio
async def test_release_unwinds_reservation():
    """Order failure path: reserve, release, second candidate should now fit."""
    config = RiskConfig(
        max_position_per_market=50.0,
        max_total_exposure=1000.0,
        daily_loss_limit=100.0,
    )
    mgr = RiskManager(config)
    d = _decision(amount=50.0)

    ok, _ = await mgr.check_and_reserve(d)
    assert ok
    assert mgr.state.total_exposure == pytest.approx(50.0)

    await mgr.release(d)
    assert mgr.state.total_exposure == pytest.approx(0.0)
    assert d.ticker not in mgr.state.positions

    # A fresh candidate should now be admitted at the full cap.
    ok2, _ = await mgr.check_and_reserve(_decision(amount=50.0))
    assert ok2


@pytest.mark.asyncio
async def test_release_is_idempotent():
    """Calling release without a prior reserve (or twice) is a no-op."""
    mgr = RiskManager(RiskConfig())
    d = _decision(amount=25.0)

    await mgr.release(d)  # never reserved
    assert mgr.state.total_exposure == 0.0

    await mgr.check_and_reserve(d)
    await mgr.release(d)
    await mgr.release(d)  # second release should not go negative
    assert mgr.state.total_exposure == 0.0


@pytest.mark.asyncio
async def test_release_only_unwinds_up_to_current_exposure():
    """If settlement already reduced exposure, release shouldn't go negative."""
    mgr = RiskManager(RiskConfig(max_position_per_market=100.0, max_total_exposure=1000.0))
    d = _decision(amount=25.0)

    await mgr.check_and_reserve(d)
    await mgr.record_settlement(d.ticker, pnl=-10.0)  # clears the position

    await mgr.release(d)
    assert mgr.state.total_exposure == 0.0
    assert mgr.state.daily_pnl == pytest.approx(-10.0)


@pytest.mark.asyncio
async def test_kill_switch_blocks_reservation():
    config = RiskConfig(kill_switch=True)
    mgr = RiskManager(config)

    ok, reason = await mgr.check_and_reserve(_decision())
    assert not ok
    assert "Kill switch" in reason
    assert mgr.state.total_exposure == 0.0


@pytest.mark.asyncio
async def test_daily_loss_limit_trips_kill_switch():
    config = RiskConfig(daily_loss_limit=50.0)
    mgr = RiskManager(config)

    # Losing settlement pushes us to the limit.
    await mgr.record_settlement("OTHER", pnl=-50.0)

    ok, reason = await mgr.check_and_reserve(_decision())
    assert not ok
    assert "Daily loss limit" in reason
    assert mgr.state.kill_switch_active is True


@pytest.mark.asyncio
async def test_reset_daily_clears_kill_switch_and_pnl():
    mgr = RiskManager(RiskConfig(daily_loss_limit=50.0))
    await mgr.record_settlement("OTHER", pnl=-60.0)
    await mgr.check_and_reserve(_decision())  # trips kill switch
    assert mgr.state.kill_switch_active is True

    await mgr.reset_daily()
    assert mgr.state.kill_switch_active is False
    assert mgr.state.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_per_market_cap_enforced_across_sequential_reserves():
    """Sequential (non-racing) reserves on same market accumulate correctly."""
    mgr = RiskManager(RiskConfig(max_position_per_market=50.0, max_total_exposure=1000.0))

    ok1, _ = await mgr.check_and_reserve(_decision(amount=30.0))
    ok2, _ = await mgr.check_and_reserve(_decision(amount=20.0))
    ok3, reason3 = await mgr.check_and_reserve(_decision(amount=1.0))  # $51 total

    assert ok1 and ok2
    assert not ok3
    assert "Per-market limit" in reason3
    assert mgr.state.positions["KXATPMATCH-TEST"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_summary_reports_state():
    mgr = RiskManager(RiskConfig(max_position_per_market=100.0, max_total_exposure=1000.0))
    await mgr.check_and_reserve(_decision(ticker="A", amount=25.0))
    await mgr.check_and_reserve(_decision(ticker="B", amount=30.0))

    s = mgr.summary()
    assert s["total_exposure"] == pytest.approx(55.0)
    assert s["active_positions"] == 2
    assert s["daily_pnl"] == 0.0
    assert s["kill_switch"] is False
