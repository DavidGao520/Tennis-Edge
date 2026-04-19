"""Tests for agent/loop.py — the Phase 3A shadow-trade spine.

The approach is deterministic: tests set up a real tiny SQLite
`market_ticks` DB, insert controlled rows, then call `tick_once()` and
`drain_once()` to drive the loop step-by-step. No real LLM, no real
asyncio scheduling races.

Covers:
  - Tick reader: latest-per-ticker, cursor advances, no duplicates
  - Edge threshold filter
  - Unknown ticker (no Glicko anchor) filter
  - Cooldown per ticker
  - Queue cap enforcement
  - Pre-dequeue freshness gate (max_candidate_age_s)
  - Skip when paused / killed
  - LLM failure path: record_llm_failure called
  - LLM success: decision written, record_llm_success called
  - Post-LLM edge stale → reject_reason="edge_stale"
  - Post-LLM edge OK → no reject
  - Context builder returns None → skip without counting failure
  - Shadow mode never sets executed=True
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from tennis_edge.agent.decisions import DecisionLog, EvAnalysis
from tennis_edge.agent.llm import (
    FakeLLMProvider,
    LLMCallError,
    PromptContext,
)
from tennis_edge.agent.loop import AgentLoop, AgentLoopConfig, MarketTickReader
from tennis_edge.agent.safety import SafetyConfig, SafetyMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick_db(path: Path) -> None:
    """Create an empty market_ticks table matching the real schema."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE market_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            ts INTEGER NOT NULL,
            yes_bid INTEGER, yes_ask INTEGER,
            last_price INTEGER, volume INTEGER,
            received_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_tick(
    path: Path,
    ticker: str,
    received_at: int,
    yes_bid: int | None = None,
    yes_ask: int | None = None,
    last_price: int | None = None,
    volume: int = 1,
) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO market_ticks (ticker, ts, yes_bid, yes_ask, "
        "last_price, volume, received_at) VALUES (?,?,?,?,?,?,?)",
        (ticker, received_at, yes_bid, yes_ask, last_price, volume, received_at),
    )
    conn.commit()
    conn.close()


def _ctx_builder_for(
    player_yes: str = "Holmgren",
    player_no: str = "Broady",
) -> callable:
    """Build a PromptContext factory that returns a valid context."""

    def build(ticker: str, model_prob: float, market_yes_cents: int):
        return PromptContext(
            ticker=ticker,
            player_yes=player_yes,
            player_no=player_no,
            tournament="Test Open",
            surface="Clay",
            round_name="R32",
            best_of=3,
            model_pre_match=model_prob,
            market_yes_cents=market_yes_cents,
            yes_form_last10="7-3",
            no_form_last10="4-6",
            h2h_summary="even",
            yes_days_since_last_match=3,
            no_days_since_last_match=2,
        )

    return build


def _make_loop(
    tmp_path: Path,
    model_prob: float = 0.53,
    llm: FakeLLMProvider | None = None,
    ctx_builder=None,
    **cfg_overrides,
) -> tuple[AgentLoop, Path, FakeLLMProvider, DecisionLog, SafetyMonitor]:
    db = tmp_path / "ticks.db"
    _make_tick_db(db)
    safety = SafetyMonitor(SafetyConfig(control_dir=str(tmp_path / "ctrl")))
    prov = llm or FakeLLMProvider()
    decisions = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    cfg_kwargs = dict(
        min_edge=0.08,
        cooldown_s=300.0,
        queue_max=20,
        max_candidate_age_s=60.0,
        tick_poll_interval_s=0.01,
        mode="shadow",
    )
    cfg_kwargs.update(cfg_overrides)
    loop = AgentLoop(
        config=AgentLoopConfig(**cfg_kwargs),
        db_path=db,
        safety=safety,
        llm=prov,
        decisions=decisions,
        model_prob_fn=lambda ticker: model_prob,
        context_builder=ctx_builder or _ctx_builder_for(),
    )
    return loop, db, prov, decisions, safety


# ---------------------------------------------------------------------------
# Tick reader
# ---------------------------------------------------------------------------


def test_reader_returns_latest_per_ticker(tmp_path):
    db = tmp_path / "ticks.db"
    _make_tick_db(db)
    now = int(time.time())
    # Reader's cursor initializes to "now" at construction; insert
    # strictly newer rows so the reader sees them.
    r = MarketTickReader(db)

    _insert_tick(db, "T1", now + 1, yes_bid=40, yes_ask=42)
    _insert_tick(db, "T1", now + 2, yes_bid=45, yes_ask=47)  # newer
    _insert_tick(db, "T2", now + 3, yes_bid=30, yes_ask=32)

    rows = r.latest_per_ticker()
    by_tick = {row.ticker: row for row in rows}
    assert set(by_tick) == {"T1", "T2"}
    assert by_tick["T1"].yes_bid == 45  # latest, not first
    assert by_tick["T2"].yes_ask == 32


def test_reader_cursor_advances(tmp_path):
    db = tmp_path / "ticks.db"
    _make_tick_db(db)
    r = MarketTickReader(db)
    now = int(time.time())

    _insert_tick(db, "T1", now + 10, last_price=50)
    first = r.latest_per_ticker()
    assert len(first) == 1

    # No new inserts → second poll returns nothing.
    second = r.latest_per_ticker()
    assert second == []

    # New tick after cursor → visible again.
    _insert_tick(db, "T1", now + 20, last_price=60)
    third = r.latest_per_ticker()
    assert len(third) == 1
    assert third[0].last_price == 60


def test_reader_price_cents_prefers_last_price(tmp_path):
    from tennis_edge.agent.loop import _TickRow
    row = _TickRow(ticker="T", received_at=0, yes_bid=40, yes_ask=50, last_price=45)
    assert MarketTickReader.price_cents(row) == 45


def test_reader_price_cents_falls_back_to_mid(tmp_path):
    from tennis_edge.agent.loop import _TickRow
    row = _TickRow(ticker="T", received_at=0, yes_bid=40, yes_ask=50, last_price=None)
    assert MarketTickReader.price_cents(row) == 45


def test_reader_price_cents_none_when_empty(tmp_path):
    from tennis_edge.agent.loop import _TickRow
    row = _TickRow(ticker="T", received_at=0, yes_bid=None, yes_ask=None, last_price=None)
    assert MarketTickReader.price_cents(row) is None


def test_reader_missing_db_returns_empty(tmp_path):
    r = MarketTickReader(tmp_path / "nope.db")
    assert r.latest_per_ticker() == []
    assert r.latest_for_ticker("X") is None


# ---------------------------------------------------------------------------
# Enqueue gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_skipped_below_edge_threshold(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.52)
    # Market 50c → edge = 0.02, below default 0.08 threshold.
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=50)
    await loop.tick_once()
    assert loop.queue.empty()


@pytest.mark.asyncio
async def test_enqueue_skipped_for_unknown_ticker(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.53)
    # Override model_prob_fn to say "unknown".
    loop.model_prob_fn = lambda ticker: None
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    assert loop.queue.empty()


@pytest.mark.asyncio
async def test_enqueue_skipped_when_tick_has_no_prices(tmp_path):
    loop, db, *_ = _make_loop(tmp_path)
    _insert_tick(db, "T1", int(time.time()) + 10)  # all prices None
    await loop.tick_once()
    assert loop.queue.empty()


@pytest.mark.asyncio
async def test_enqueue_accepts_above_edge_threshold(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.53)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    assert loop.queue.qsize() == 1


@pytest.mark.asyncio
async def test_cooldown_blocks_second_enqueue(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.53, cooldown_s=300.0)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    assert loop.queue.qsize() == 1

    _insert_tick(db, "T1", int(time.time()) + 20, last_price=14)
    await loop.tick_once()
    assert loop.queue.qsize() == 1  # second one dropped by cooldown


@pytest.mark.asyncio
async def test_cooldown_zero_allows_immediate_requeue(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.53, cooldown_s=0.0)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    _insert_tick(db, "T1", int(time.time()) + 20, last_price=14)
    await loop.tick_once()
    assert loop.queue.qsize() == 2


@pytest.mark.asyncio
async def test_queue_cap_drops_overflow(tmp_path, caplog):
    loop, db, *_ = _make_loop(
        tmp_path, model_prob=0.53, queue_max=2, cooldown_s=0.0,
    )
    for i in range(5):
        _insert_tick(db, f"T{i}", int(time.time()) + 10 + i, last_price=15)
    with caplog.at_level("WARNING"):
        await loop.tick_once()
    assert loop.queue.qsize() == 2
    assert any("queue full" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Worker behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_skips_when_safety_paused(tmp_path):
    loop, db, prov, decisions, safety = _make_loop(tmp_path, model_prob=0.53)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    assert loop.queue.qsize() == 1

    # Flip to PAUSED (not KILLED — still is_running()==False).
    from tennis_edge.agent.safety import SafetyState
    safety._state = SafetyState.PAUSED

    handled = await loop.drain_once()
    assert handled  # drained, but not acted on
    assert prov.call_count == 0
    assert decisions.count_decisions() == 0


@pytest.mark.asyncio
async def test_worker_drops_stale_candidate(tmp_path):
    loop, db, prov, decisions, _ = _make_loop(
        tmp_path, model_prob=0.53, max_candidate_age_s=0.001,
    )
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    # Guarantee the candidate is past max age.
    await asyncio.sleep(0.01)
    await loop.drain_once()
    assert prov.call_count == 0
    assert decisions.count_decisions() == 0


@pytest.mark.asyncio
async def test_context_builder_none_skips_without_llm_failure(tmp_path):
    loop, db, prov, decisions, safety = _make_loop(tmp_path, model_prob=0.53)
    loop.context_builder = lambda *a, **kw: None
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    await loop.drain_once()
    assert prov.call_count == 0
    assert decisions.count_decisions() == 0
    # Not counted as LLM failure — the LLM was never called.
    assert safety.consecutive_llm_failures() == 0


@pytest.mark.asyncio
async def test_worker_happy_path_logs_decision(tmp_path):
    loop, db, prov, decisions, safety = _make_loop(tmp_path, model_prob=0.53)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    await loop.drain_once()

    assert prov.call_count == 1
    assert decisions.count_decisions() == 1
    loaded = list(decisions.iter_decisions())[0]
    assert loaded.ticker == "T1"
    assert loaded.mode == "shadow"
    assert loaded.executed is False
    assert loaded.analysis.recommendation == "BUY_YES"


@pytest.mark.asyncio
async def test_worker_llm_failure_counts_on_safety(tmp_path):
    prov = FakeLLMProvider(raise_exc=LLMCallError("flake"))
    loop, db, _, decisions, safety = _make_loop(
        tmp_path, model_prob=0.53, llm=prov,
    )
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    await loop.drain_once()

    assert safety.consecutive_llm_failures() == 1
    assert decisions.count_decisions() == 0  # no log on failure


@pytest.mark.asyncio
async def test_worker_three_llm_failures_trip_kill_switch(tmp_path):
    prov = FakeLLMProvider(raise_exc=LLMCallError("flake"))
    loop, db, _, _, safety = _make_loop(
        tmp_path, model_prob=0.53, llm=prov, cooldown_s=0.0,
    )
    for i in range(3):
        _insert_tick(db, f"T{i}", int(time.time()) + 10 + i, last_price=15)
    await loop.tick_once()
    for _ in range(3):
        await loop.drain_once()

    assert safety.is_killed()


@pytest.mark.asyncio
async def test_post_llm_edge_stale_rejects(tmp_path):
    """LLM returns; market has since moved to kill the edge."""
    analysis = EvAnalysis(
        edge_estimate=0.6, recommendation="BUY_YES",
        confidence="medium", reasoning="r", key_factors=["a"],
    )
    prov = FakeLLMProvider(analysis=analysis)
    loop, db, _, decisions, _ = _make_loop(
        tmp_path, model_prob=0.53, llm=prov, stale_edge_threshold=0.08,
    )

    # Entry edge: market 15c → edge 0.38. Good.
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    assert loop.queue.qsize() == 1

    # Before the worker re-checks, market rockets to 50c → edge = 0.03,
    # below 0.08 threshold.
    _insert_tick(db, "T1", int(time.time()) + 30, last_price=50)

    await loop.drain_once()
    d = list(decisions.iter_decisions())[0]
    assert d.reject_reason == "edge_stale"
    assert d.edge_at_execution is not None
    assert abs(d.edge_at_execution) < 0.08
    # Edge-stale decisions are still logged (shadow analytics want them).
    assert d.executed is False


@pytest.mark.asyncio
async def test_post_llm_edge_holds_no_reject(tmp_path):
    loop, db, _, decisions, _ = _make_loop(tmp_path, model_prob=0.53)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    # Market drifts but edge still strong.
    _insert_tick(db, "T1", int(time.time()) + 30, last_price=20)
    await loop.drain_once()

    d = list(decisions.iter_decisions())[0]
    assert d.reject_reason is None
    assert d.edge_at_execution is not None
    assert d.edge_at_execution > 0.08


@pytest.mark.asyncio
async def test_worker_skips_when_killed_between_enqueue_and_dequeue(tmp_path):
    loop, db, prov, decisions, safety = _make_loop(tmp_path, model_prob=0.53)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    # Kill safety after enqueue but before dequeue.
    await safety.record_llm_failure(LLMCallError("a"))
    await safety.record_llm_failure(LLMCallError("b"))
    await safety.record_llm_failure(LLMCallError("c"))  # trips
    assert safety.is_killed()

    await loop.drain_once()
    assert prov.call_count == 0
    assert decisions.count_decisions() == 0


@pytest.mark.asyncio
async def test_shadow_mode_never_sets_executed(tmp_path):
    loop, db, _, decisions, _ = _make_loop(tmp_path, model_prob=0.53)
    for i in range(3):
        _insert_tick(db, f"T{i}", int(time.time()) + 10 + i, last_price=15)
    await loop.tick_once()
    for _ in range(3):
        loop.config  # noqa — just keeping intent clear
        await loop.drain_once()

    for d in decisions.iter_decisions():
        assert d.executed is False
        assert d.order_id is None
        assert d.mode == "shadow"


@pytest.mark.asyncio
async def test_cooldown_remaining_reported(tmp_path):
    loop, db, *_ = _make_loop(tmp_path, model_prob=0.53, cooldown_s=300.0)
    _insert_tick(db, "T1", int(time.time()) + 10, last_price=15)
    await loop.tick_once()
    rem = loop.cooldown_remaining("T1")
    assert 100.0 < rem <= 300.0
    assert loop.cooldown_remaining("other") == 0.0
