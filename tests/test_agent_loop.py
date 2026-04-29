"""Tests for the v2 AgentLoop (signal subscriber + grounded gates + executor).

Approach: structural fakes for everything injectable, including
the price_source callable that AgentLoop uses for the post-LLM edge
re-check. All paths driven via `on_signal()` and `drain_once()` so
there is no asyncio scheduling racing.

Coverage targets these critical paths from the eng review:
  - confidence=low → HARD reject
  - grounded_edge < min → HARD reject
  - post-LLM edge stale → HARD reject (v1 was soft)
  - place_order exception → risk.release called, no double charge
  - 3 consecutive order failures → safety counter trips kill switch
  - shadow vs auto mode both go through the same gate, only the
    injected ExchangeClient differs

Note: the v2 architecture decouples agent from any tick-logger DB.
AgentLoop pulls fresh prices from a callable (`price_source`) which
in production is `MonitorBridge.latest_price`. Tests inject a small
FakePriceSource dict.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from tennis_edge.agent.decisions import (
    AgentDecision,
    DecisionLog,
    EvAnalysis,
)
from tennis_edge.agent.llm import (
    FakeLLMProvider,
    LLMCallError,
    PromptContext,
)
from tennis_edge.agent.loop import (
    AgentLoop,
    AgentLoopConfig,
    Candidate,
)
from tennis_edge.agent.monitor_bridge import MonitorSignal
from tennis_edge.agent.safety import SafetyConfig, SafetyMonitor, SafetyState
from tennis_edge.config import RiskConfig
from tennis_edge.exchange.paper import PaperTradingEngine
from tennis_edge.exchange.schemas import OrderRequest, OrderResponse
from tennis_edge.strategy.risk import RiskManager


# ---------------------------------------------------------------------------
# Fake price source (replaces SQLite tick reader from v2.0)
# ---------------------------------------------------------------------------


class FakePriceSource:
    """Mimics MonitorBridge.latest_price.

    Stores per-ticker (yes_cents, monotonic_ts). Constructor accepts
    either fresh entries (age computed at lookup) or aged entries
    (stored age applied directly). Returns None for unknown tickers.
    """

    def __init__(self):
        self._prices: dict[str, tuple[int, float]] = {}

    def set(self, ticker: str, cents: int, age_s: float = 0.0) -> None:
        """Set price with age `age_s` seconds in the past."""
        self._prices[ticker] = (cents, time.monotonic() - age_s)

    def clear(self, ticker: str) -> None:
        self._prices.pop(ticker, None)

    def __call__(self, ticker: str) -> tuple[int, float] | None:
        cached = self._prices.get(ticker)
        if cached is None:
            return None
        cents, observed_at = cached
        return cents, time.monotonic() - observed_at


def _signal(
    ticker: str = "KXATPMATCH-T1",
    market_yes_cents: int = 15,
    model_prob: float = 0.50,
    side: str = "yes",
) -> MonitorSignal:
    return MonitorSignal(
        ticker=ticker,
        player_yes="Holmgren",
        player_no="Broady",
        category="ATP Main",
        market_yes_cents=market_yes_cents,
        model_prob=model_prob,
        market_prob=market_yes_cents / 100.0,
        prematch_ev=abs(model_prob - market_yes_cents / 100.0),
        recommended_side=side,
        detected_at=time.monotonic(),
    )


def _ev_analysis(
    edge_estimate: float = 0.55,
    recommendation: str = "BUY_YES",
    confidence: str = "high",
) -> EvAnalysis:
    return EvAnalysis(
        edge_estimate=edge_estimate,
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        reasoning="test reasoning",
        key_factors=["a", "b"],
    )


def _ctx_for_signal(sig: MonitorSignal) -> PromptContext:
    return PromptContext(
        ticker=sig.ticker,
        player_yes=sig.player_yes,
        player_no=sig.player_no,
        tournament="Test Open",
        surface="Hard",
        round_name="R32",
        best_of=3,
        model_pre_match=sig.model_prob,
        market_yes_cents=sig.market_yes_cents,
        yes_form_last10="?",
        no_form_last10="?",
        h2h_summary="?",
        yes_days_since_last_match=None,
        no_days_since_last_match=None,
    )


async def _build_ctx(sig: MonitorSignal) -> PromptContext:
    return _ctx_for_signal(sig)


def _make_loop(
    tmp_path: Path,
    *,
    llm: FakeLLMProvider | None = None,
    exchange=None,
    fresh_price_at: int | None = None,
    config_overrides: dict | None = None,
) -> tuple[
    AgentLoop, FakePriceSource, FakeLLMProvider, DecisionLog,
    SafetyMonitor, RiskManager,
]:
    """Build a wired AgentLoop with fakes for every dependency.

    Returns the loop, the price source (so tests can mutate it
    mid-test), and the other handles tests assert against.
    """
    price_source = FakePriceSource()
    if fresh_price_at is not None:
        price_source.set("KXATPMATCH-T1", fresh_price_at, age_s=0.0)

    safety = SafetyMonitor(SafetyConfig(control_dir=str(tmp_path / "ctrl")))
    decisions = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    risk = RiskManager(RiskConfig(
        max_position_per_market=200.0,
        max_total_exposure=10000.0,
        daily_loss_limit=10000.0,
    ))
    if exchange is None:
        exchange = PaperTradingEngine(initial_balance=10000.0)
    if llm is None:
        llm = FakeLLMProvider(analysis=_ev_analysis())

    cfg_kwargs: dict = dict(
        queue_max=20,
        cooldown_s=300.0,
        max_candidate_age_s=60.0,
        min_grounded_edge=0.10,
        stale_edge_hard_threshold=0.08,
        kelly_fraction=0.25,
        bankroll=1000.0,
        max_position_per_market=50.0,
        confidence_mult_high=1.0,
        confidence_mult_medium=0.5,
        confidence_mult_low=0.0,
        mode="auto",
    )
    if config_overrides:
        cfg_kwargs.update(config_overrides)

    loop = AgentLoop(
        config=AgentLoopConfig(**cfg_kwargs),
        safety=safety,
        llm=llm,
        decisions=decisions,
        risk=risk,
        exchange=exchange,
        prompt_builder=_build_ctx,
        price_source=price_source,
    )
    return loop, price_source, llm, decisions, safety, risk


# ---------------------------------------------------------------------------
# FakePriceSource (mirrors MonitorBridge.latest_price contract)
# ---------------------------------------------------------------------------


def test_price_source_returns_none_for_unknown_ticker():
    src = FakePriceSource()
    assert src("MISSING") is None


def test_price_source_returns_cents_and_age():
    src = FakePriceSource()
    src.set("T1", 42, age_s=0.0)
    out = src("T1")
    assert out is not None
    cents, age = out
    assert cents == 42
    assert 0.0 <= age < 0.5  # just-set should be near-zero age


def test_price_source_age_grows_with_time():
    src = FakePriceSource()
    src.set("T1", 50, age_s=10.0)
    out = src("T1")
    assert out is not None
    _, age = out
    # We injected age=10 → result should be ≥10s.
    assert age >= 10.0


def test_price_source_clear():
    src = FakePriceSource()
    src.set("T1", 50)
    src.clear("T1")
    assert src("T1") is None


# ---------------------------------------------------------------------------
# on_signal pre-LLM gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_signal_enqueues_when_running(tmp_path):
    loop, _, _, _, _, _ = _make_loop(tmp_path)
    await loop.on_signal(_signal())
    assert loop.queue.qsize() == 1


@pytest.mark.asyncio
async def test_on_signal_drops_when_safety_paused(tmp_path):
    loop, _, _, _, safety, _ = _make_loop(tmp_path)
    safety._state = SafetyState.PAUSED
    await loop.on_signal(_signal())
    assert loop.queue.qsize() == 0


@pytest.mark.asyncio
async def test_on_signal_drops_when_safety_killed(tmp_path):
    loop, _, _, _, safety, _ = _make_loop(tmp_path)
    safety._state = SafetyState.KILLED
    await loop.on_signal(_signal())
    assert loop.queue.qsize() == 0


@pytest.mark.asyncio
async def test_on_signal_cooldown_blocks_second(tmp_path):
    loop, *_ = _make_loop(tmp_path)
    await loop.on_signal(_signal(ticker="T1"))
    await loop.on_signal(_signal(ticker="T1"))
    assert loop.queue.qsize() == 1


@pytest.mark.asyncio
async def test_on_signal_queue_cap_drops_overflow(tmp_path, caplog):
    loop, *_ = _make_loop(tmp_path, config_overrides={"queue_max": 2, "cooldown_s": 0.0})
    for i in range(5):
        await loop.on_signal(_signal(ticker=f"T{i}"))
    with caplog.at_level("WARNING"):
        # Trigger one more by dropping cooldown on T0.
        loop._cooldown_until.clear()
        for i in range(5):
            await loop.on_signal(_signal(ticker=f"T{i}"))
    assert loop.queue.qsize() == 2
    assert any("queue full" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Worker — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_happy_path_executes_and_logs(tmp_path):
    """High-confidence BUY_YES with edge → places paper order, logs
    decision with executed=True and a real order_id."""
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
    ))
    loop, _, _, decisions, _, risk = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal(ticker="KXATPMATCH-T1", market_yes_cents=15))
    await loop.drain_once()

    rows = list(decisions.iter_decisions())
    assert len(rows) == 1
    d = rows[0]
    assert d.executed is True
    assert d.order_id is not None
    assert d.order_id.startswith("paper-")
    assert d.reject_reason is None
    assert risk.state.total_exposure > 0  # exposure reserved


# ---------------------------------------------------------------------------
# Worker — hard rejects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_hard_rejects_low_confidence(tmp_path):
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="low",
    ))
    loop, _, _, decisions, _, risk = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal(market_yes_cents=15))
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.order_id is None
    assert d.reject_reason == "confidence_low"
    assert risk.state.total_exposure == 0  # nothing reserved


@pytest.mark.asyncio
async def test_worker_hard_rejects_grounded_edge_below_min(tmp_path):
    """LLM says edge=0.18, market=15c → grounded_edge=0.03, below 0.10 min."""
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.18, recommendation="BUY_YES", confidence="high",
    ))
    loop, _, _, decisions, _, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal(market_yes_cents=15))
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.reject_reason == "grounded_edge_below_min"


@pytest.mark.asyncio
async def test_worker_hard_rejects_skip_recommendation(tmp_path):
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.50, recommendation="SKIP", confidence="medium",
    ))
    loop, _, _, decisions, _, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal(market_yes_cents=15))
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.reject_reason == "rec_skip"


@pytest.mark.asyncio
async def test_worker_hard_rejects_post_llm_edge_stale(tmp_path):
    """Signal market=15c, LLM ran on 0.55 → grounded_edge=0.40 OK.
    But latest tick now shows 50c → live edge=0.05, below 0.08 hard
    threshold. Must HARD reject (v1 was soft)."""
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
    ))
    loop, prices, _, decisions, _, _ = _make_loop(tmp_path, llm=llm)

    # Signal queued at 15c.
    await loop.on_signal(_signal(ticker="KXATPMATCH-T1", market_yes_cents=15))

    # Bridge's most recent observation is now 50c — market moved
    # against us during Gemini's think. Re-check at this price.
    prices.set("KXATPMATCH-T1", 50, age_s=0.0)

    await loop.drain_once()
    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.reject_reason == "edge_stale"
    assert d.edge_at_execution is not None
    assert abs(d.edge_at_execution) < 0.08


@pytest.mark.asyncio
async def test_worker_hard_rejects_when_no_recent_tick(tmp_path):
    """If the post-LLM re-check cannot find a tick, reject — never
    fill at an unknown price with real money."""
    llm = FakeLLMProvider(analysis=_ev_analysis())
    loop, _, _, decisions, _, _ = _make_loop(tmp_path, llm=llm)
    # No tick inserted at all.
    await loop.on_signal(_signal())
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.reject_reason == "no_recent_tick"


# ---------------------------------------------------------------------------
# Worker — risk gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_rejects_when_risk_check_fails(tmp_path):
    """Force a tight per-market cap. First Kelly-sized trade fits; a
    second trade on the same ticker doubles the reservation and
    blows past the cap."""
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
    ))
    loop, prices, _, decisions, _, risk = _make_loop(
        tmp_path, llm=llm,
        # Kelly bet ~$50 (Kelly_pct ≈ 0.118 × $1000 bankroll → cap $50).
        # Risk per-market cap = $60 → first $49.95 fits, second
        # would push to $99.90 > $60 → reject.
        config_overrides={"cooldown_s": 0.0},
    )
    prices.set("T1", 15)
    risk.config = RiskConfig(
        max_position_per_market=60.0,
        max_total_exposure=10000.0,
        daily_loss_limit=10000.0,
    )

    await loop.on_signal(_signal(ticker="T1"))
    await loop.drain_once()
    await loop.on_signal(_signal(ticker="T1"))
    await loop.drain_once()

    rows = list(decisions.iter_decisions())
    assert len(rows) == 2
    assert rows[0].executed is True
    assert rows[1].executed is False
    assert rows[1].reject_reason is not None
    assert rows[1].reject_reason.startswith("risk:")


# ---------------------------------------------------------------------------
# Worker — executor failures
# ---------------------------------------------------------------------------


class _RaisingExchange:
    """Always raises on place_order. Other ExchangeClient methods
    not used by AgentLoop."""

    raises_n: int = 999  # raise this many times

    def __init__(self, raises_n: int = 999):
        self.raises_n = raises_n
        self.calls = 0

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        self.calls += 1
        if self.calls <= self.raises_n:
            raise RuntimeError("simulated network flake")
        return OrderResponse(
            order_id="ok-after-retries", ticker=order.ticker,
            status="filled", side=order.side, action=order.action,
            count=order.count, yes_price=order.yes_price,
        )

    # AgentLoop only calls place_order; stubs not strictly needed.
    async def get_markets(self, **kw): return []
    async def get_market(self, ticker): return None
    async def get_orderbook(self, ticker): return None
    async def cancel_order(self, oid): pass
    async def get_positions(self): return []
    async def get_balance(self): return 0.0


@pytest.mark.asyncio
async def test_worker_release_on_place_order_exception(tmp_path):
    """When place_order raises, exposure must be released so the next
    candidate is not blocked by phantom reservation."""
    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
    ))
    ex = _RaisingExchange(raises_n=1)
    loop, _, _, decisions, _, risk = _make_loop(
        tmp_path, llm=llm, exchange=ex, fresh_price_at=15,
    )
    await loop.on_signal(_signal())
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.executed is False
    assert d.reject_reason == "order_failed"
    # Exposure was reserved then released → 0.
    assert risk.state.total_exposure == 0.0


@pytest.mark.asyncio
async def test_three_consecutive_order_failures_trip_kill(tmp_path):
    """Persistent place_order failures should trip the safety kill via
    the dedicated ORDER_CONSECUTIVE_FAILURES TripReason — distinct
    from LLM failures so post-mortem analytics can tell them apart."""
    from tennis_edge.agent.safety import TripReason

    llm = FakeLLMProvider(analysis=_ev_analysis(
        edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
    ))
    ex = _RaisingExchange(raises_n=999)
    # Need fresh price for each ticker we drain.
    loop, prices, _, _, safety, _ = _make_loop(
        tmp_path, llm=llm, exchange=ex, fresh_price_at=15,
        config_overrides={"cooldown_s": 0.0},
    )
    for i in range(3):
        prices.set(f"T{i}", 15)

    for i in range(3):
        await loop.on_signal(_signal(ticker=f"T{i}"))
        await loop.drain_once()

    assert safety.is_killed()
    assert safety.trip_event().reason is TripReason.ORDER_CONSECUTIVE_FAILURES


# ---------------------------------------------------------------------------
# LLM failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_llm_failure_recorded_no_decision_logged(tmp_path):
    llm = FakeLLMProvider(raise_exc=LLMCallError("boom"))
    loop, _, _, decisions, safety, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal())
    await loop.drain_once()

    assert safety.consecutive_llm_failures() == 1
    assert decisions.count_decisions() == 0


@pytest.mark.asyncio
async def test_three_llm_failures_kill_safety(tmp_path):
    llm = FakeLLMProvider(raise_exc=LLMCallError("flake"))
    loop, _, _, _, safety, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
        config_overrides={"cooldown_s": 0.0},
    )
    for i in range(3):
        await loop.on_signal(_signal(ticker=f"T{i}"))
        await loop.drain_once()
    assert safety.is_killed()


# ---------------------------------------------------------------------------
# Worker — freshness gate + safety state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_drops_stale_candidate(tmp_path):
    llm = FakeLLMProvider()
    loop, _, _, decisions, _, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
        config_overrides={"max_candidate_age_s": 0.001},
    )
    await loop.on_signal(_signal())
    await asyncio.sleep(0.01)
    await loop.drain_once()
    assert llm.call_count == 0
    assert decisions.count_decisions() == 0


@pytest.mark.asyncio
async def test_worker_skips_when_safety_not_running(tmp_path):
    llm = FakeLLMProvider()
    loop, _, _, decisions, safety, _ = _make_loop(tmp_path, llm=llm)
    await loop.on_signal(_signal())
    safety._state = SafetyState.PAUSED
    await loop.drain_once()
    assert llm.call_count == 0
    assert decisions.count_decisions() == 0


# ---------------------------------------------------------------------------
# Prompt builder None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_prompt_builder_none_skips_silently(tmp_path):
    llm = FakeLLMProvider()
    loop, _, _, decisions, safety, _ = _make_loop(tmp_path, llm=llm)

    async def none_builder(sig):
        return None
    loop.prompt_builder = none_builder

    await loop.on_signal(_signal())
    await loop.drain_once()
    assert llm.call_count == 0
    assert decisions.count_decisions() == 0
    # Not counted as LLM failure.
    assert safety.consecutive_llm_failures() == 0


# ---------------------------------------------------------------------------
# cooldown_remaining helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_remaining_reported(tmp_path):
    loop, *_ = _make_loop(tmp_path, config_overrides={"cooldown_s": 300.0})
    await loop.on_signal(_signal(ticker="T1"))
    rem = loop.cooldown_remaining("T1")
    assert 100.0 < rem <= 300.0
    assert loop.cooldown_remaining("other") == 0.0


# ---------------------------------------------------------------------------
# Decision log shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_log_carries_full_context(tmp_path):
    """AgentDecision must record provider, prompt hash, raw output,
    edge_at_execution, and run_id for downstream analytics."""
    llm = FakeLLMProvider(
        name="fake-grounded",
        analysis=_ev_analysis(
            edge_estimate=0.55, recommendation="BUY_YES", confidence="high",
        ),
    )
    loop, _, _, decisions, _, _ = _make_loop(
        tmp_path, llm=llm, fresh_price_at=15,
    )
    await loop.on_signal(_signal(market_yes_cents=15))
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.run_id == loop.run_id
    assert d.llm_provider == "fake-grounded"
    assert len(d.llm_prompt_hash) == 16
    assert d.analysis.recommendation == "BUY_YES"
    assert d.edge_at_execution is not None
