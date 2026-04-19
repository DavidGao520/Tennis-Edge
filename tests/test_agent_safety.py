"""Tests for agent/safety.py — five kill switches + file-flag IPC.

Covers the full truth table of each switch plus the interactions the
watchdog_loop has to handle correctly (pause short-circuit, flatten
takes precedence, trip is idempotent and KILLED is terminal).

External deps (WebSocket, SQLite, budget tracker, risk manager) are
stubbed with minimal duck-typed fakes that match the Protocols in
safety.py.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tennis_edge.agent.llm import LLMCallError
from tennis_edge.agent.safety import (
    SafetyConfig,
    SafetyMonitor,
    SafetyState,
    TripReason,
    clear_flatten_flag,
    clear_pause_flag,
    touch_flatten_flag,
    touch_pause_flag,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeWS:
    msg: float | None = 0.0
    conn: float | None = 0.0

    def seconds_since_last_message(self) -> float | None:
        return self.msg

    def seconds_since_last_connect(self) -> float | None:
        return self.conn


@dataclass
class FakeBudget:
    remaining: dict[str, float] = field(default_factory=dict)

    def remaining_usd(self, provider: str) -> float:
        return self.remaining.get(provider, 999.0)


@dataclass
class _RiskState:
    daily_pnl: float = 0.0


@dataclass
class FakeRisk:
    state: _RiskState = field(default_factory=_RiskState)


def _mk_tick_db(path: Path, max_received_at: int | None) -> None:
    """Create a tick DB with a controllable max(received_at) row."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, ts INTEGER,
            yes_bid INTEGER, yes_ask INTEGER,
            last_price INTEGER, volume INTEGER,
            received_at INTEGER NOT NULL
        )
        """
    )
    if max_received_at is not None:
        conn.execute(
            "INSERT INTO market_ticks (ticker, ts, yes_bid, yes_ask, "
            "last_price, volume, received_at) VALUES (?,?,?,?,?,?,?)",
            ("T", max_received_at, 50, 52, 51, 1, max_received_at),
        )
    conn.commit()
    conn.close()


def _cfg(tmp_path: Path, **overrides) -> SafetyConfig:
    base = dict(control_dir=str(tmp_path / "ctrl"))
    base.update(overrides)
    return SafetyConfig(**base)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_starts_running(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    assert m.is_running()
    assert not m.is_paused()
    assert not m.is_killed()
    assert m.trip_event() is None
    assert m.consecutive_llm_failures() == 0


# ---------------------------------------------------------------------------
# LLM consecutive-failure switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_failures_trip_at_threshold(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, max_consecutive_llm_failures=3))
    for i in range(2):
        await m.record_llm_failure(LLMCallError(f"flake {i}"))
        assert m.is_running()
    await m.record_llm_failure(LLMCallError("final"))
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.LLM_CONSECUTIVE_FAILURES


@pytest.mark.asyncio
async def test_llm_success_resets_counter(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, max_consecutive_llm_failures=3))
    await m.record_llm_failure(LLMCallError("1"))
    await m.record_llm_failure(LLMCallError("2"))
    assert m.consecutive_llm_failures() == 2

    await m.record_llm_success()
    assert m.consecutive_llm_failures() == 0

    # Now two more failures should not trip (would have been 4 before reset).
    await m.record_llm_failure(LLMCallError("3"))
    await m.record_llm_failure(LLMCallError("4"))
    assert m.is_running()


# ---------------------------------------------------------------------------
# WebSocket switches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_reconnect_starvation_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, ws_reconnect_stale_s=60))
    ws = FakeWS(msg=1.0, conn=120.0)  # 2 minutes since last successful connect
    await m.check_ws(ws, live_match_fn=lambda: False)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.WS_RECONNECT_STARVATION


@pytest.mark.asyncio
async def test_ws_never_connected_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    ws = FakeWS(msg=None, conn=None)
    await m.check_ws(ws, live_match_fn=lambda: False)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.WS_RECONNECT_STARVATION


@pytest.mark.asyncio
async def test_ws_stale_but_no_live_match_does_not_trip(tmp_path):
    """Quiet nights have no ticker traffic by design. Must not trip."""
    m = SafetyMonitor(_cfg(tmp_path, ws_message_stale_s=60))
    ws = FakeWS(msg=3600.0, conn=1.0)  # 1 hour silent, but link fresh
    await m.check_ws(ws, live_match_fn=lambda: False)
    assert m.is_running()


@pytest.mark.asyncio
async def test_ws_stale_with_live_match_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, ws_message_stale_s=60))
    ws = FakeWS(msg=90.0, conn=1.0)
    await m.check_ws(ws, live_match_fn=lambda: True)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.WS_STALE_WITH_LIVE_MATCH


@pytest.mark.asyncio
async def test_ws_check_accepts_async_live_match_fn(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, ws_message_stale_s=60))
    ws = FakeWS(msg=90.0, conn=1.0)

    async def is_live():
        return True

    await m.check_ws(ws, live_match_fn=is_live)
    assert m.is_killed()


@pytest.mark.asyncio
async def test_ws_fresh_does_not_trip(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    ws = FakeWS(msg=5.0, conn=5.0)
    await m.check_ws(ws, live_match_fn=lambda: True)
    assert m.is_running()


# ---------------------------------------------------------------------------
# Tick-logger stale switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_logger_fresh_does_not_trip(tmp_path):
    db = tmp_path / "t.db"
    _mk_tick_db(db, max_received_at=int(time.time()) - 5)
    m = SafetyMonitor(_cfg(tmp_path))
    await m.check_tick_logger(db)
    assert m.is_running()


@pytest.mark.asyncio
async def test_tick_logger_stale_trips(tmp_path):
    db = tmp_path / "t.db"
    _mk_tick_db(db, max_received_at=int(time.time()) - 300)
    m = SafetyMonitor(_cfg(tmp_path, tick_logger_stale_s=60))
    await m.check_tick_logger(db)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.TICK_LOGGER_STALE


@pytest.mark.asyncio
async def test_tick_logger_empty_table_trips(tmp_path):
    db = tmp_path / "t.db"
    _mk_tick_db(db, max_received_at=None)
    m = SafetyMonitor(_cfg(tmp_path))
    await m.check_tick_logger(db)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.TICK_LOGGER_STALE


@pytest.mark.asyncio
async def test_tick_logger_missing_db_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    await m.check_tick_logger(tmp_path / "does-not-exist.db")
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.TICK_LOGGER_STALE


@pytest.mark.asyncio
async def test_tick_logger_missing_table_trips(tmp_path):
    db = tmp_path / "t.db"
    # Create DB but not the market_ticks table.
    sqlite3.connect(db).close()
    m = SafetyMonitor(_cfg(tmp_path))
    await m.check_tick_logger(db)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.TICK_LOGGER_STALE


# ---------------------------------------------------------------------------
# Budget switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    bt = FakeBudget(remaining={"gemini": 0.0, "claude": 5.0})
    await m.check_budget(bt, ["gemini", "claude"])
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_budget_has_remaining_does_not_trip(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    bt = FakeBudget(remaining={"gemini": 5.0, "claude": 10.0})
    await m.check_budget(bt, ["gemini", "claude"])
    assert m.is_running()


# ---------------------------------------------------------------------------
# Daily P&L switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_loss_limit_trips(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, daily_loss_limit_usd=200.0))
    risk = FakeRisk(state=_RiskState(daily_pnl=-250.0))
    await m.check_daily_pnl(risk)
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.DAILY_LOSS_LIMIT


@pytest.mark.asyncio
async def test_daily_loss_within_limit_does_not_trip(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path, daily_loss_limit_usd=200.0))
    risk = FakeRisk(state=_RiskState(daily_pnl=-150.0))
    await m.check_daily_pnl(risk)
    assert m.is_running()


@pytest.mark.asyncio
async def test_daily_positive_pnl_does_not_trip(tmp_path):
    m = SafetyMonitor(_cfg(tmp_path))
    risk = FakeRisk(state=_RiskState(daily_pnl=500.0))
    await m.check_daily_pnl(risk)
    assert m.is_running()


# ---------------------------------------------------------------------------
# File-flag IPC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_flag_sets_paused(tmp_path):
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path))
    touch_pause_flag(ctrl)
    await m.check_user_flags()
    assert m.is_paused()


@pytest.mark.asyncio
async def test_pause_flag_cleared_resumes(tmp_path):
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path))
    touch_pause_flag(ctrl)
    await m.check_user_flags()
    assert m.is_paused()

    clear_pause_flag(ctrl)
    await m.check_user_flags()
    assert m.is_running()


@pytest.mark.asyncio
async def test_flatten_flag_trips(tmp_path):
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path))
    touch_flatten_flag(ctrl)
    await m.check_user_flags()
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.USER_FLATTEN


@pytest.mark.asyncio
async def test_flatten_flag_beats_pause_flag(tmp_path):
    """Both flags present: flatten takes precedence."""
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path))
    touch_pause_flag(ctrl)
    touch_flatten_flag(ctrl)
    await m.check_user_flags()
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.USER_FLATTEN


def test_clear_flag_tolerates_missing_file(tmp_path):
    """Daemon post-flatten cleanup must not raise if flag already gone."""
    ctrl = tmp_path / "ctrl"
    ctrl.mkdir()
    # Must not raise.
    clear_pause_flag(ctrl)
    clear_flatten_flag(ctrl)


# ---------------------------------------------------------------------------
# Trip idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_trip_wins(tmp_path):
    """Multiple simultaneous kill switches → only the first is recorded."""
    m = SafetyMonitor(_cfg(tmp_path))
    ws = FakeWS(msg=None, conn=300.0)  # triggers WS_RECONNECT_STARVATION
    await m.check_ws(ws, live_match_fn=lambda: False)
    first_reason = m.trip_event().reason

    # Try to trip via budget; state must stay KILLED with the first reason.
    await m.check_budget(FakeBudget(remaining={"p": 0.0}), ["p"])
    assert m.is_killed()
    assert m.trip_event().reason is first_reason


@pytest.mark.asyncio
async def test_killed_is_terminal_user_flag_has_no_effect(tmp_path):
    """Clearing the pause flag cannot resurrect a KILLED daemon."""
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path))
    await m.record_llm_failure(LLMCallError("1"))
    await m.record_llm_failure(LLMCallError("2"))
    await m.record_llm_failure(LLMCallError("3"))
    assert m.is_killed()

    touch_pause_flag(ctrl)
    clear_pause_flag(ctrl)
    await m.check_user_flags()
    assert m.is_killed()  # cannot go RUNNING


# ---------------------------------------------------------------------------
# watchdog_loop integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_loop_exits_when_killed(tmp_path):
    """Watchdog loop should return promptly once a switch trips."""
    m = SafetyMonitor(_cfg(tmp_path, ws_reconnect_stale_s=60))
    ws = FakeWS(msg=1.0, conn=999.0)   # will trip reconnect starvation
    db = tmp_path / "t.db"
    _mk_tick_db(db, max_received_at=int(time.time()))
    bt = FakeBudget(remaining={"g": 5.0})
    risk = FakeRisk()

    await asyncio.wait_for(
        m.watchdog_loop(
            ws=ws,
            db_path=db,
            budget=bt,
            providers=["g"],
            risk=risk,
            live_match_fn=lambda: True,
            interval_s=0.01,
        ),
        timeout=2.0,
    )
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.WS_RECONNECT_STARVATION


@pytest.mark.asyncio
async def test_watchdog_loop_skips_checks_when_paused(tmp_path):
    """While paused, kill-switch checks must not run — otherwise a user
    pausing the agent during a known-bad condition (e.g. market close)
    would trip anyway and convert a reversible pause into a terminal kill."""
    ctrl = tmp_path / "ctrl"
    m = SafetyMonitor(_cfg(tmp_path, ws_reconnect_stale_s=60))
    # Set up conditions that WOULD trip if checked.
    ws = FakeWS(msg=None, conn=999.0)
    db = tmp_path / "t.db"
    _mk_tick_db(db, max_received_at=int(time.time()) - 9999)
    bt = FakeBudget(remaining={"g": 0.0})
    risk = FakeRisk(state=_RiskState(daily_pnl=-1000.0))

    # Put the monitor into PAUSED before the loop observes anything.
    touch_pause_flag(ctrl)

    async def stopper():
        # Give the loop two ticks in paused state, then flip to killed
        # externally so the loop exits.
        await asyncio.sleep(0.05)
        assert m.is_paused()  # proof the trips did not fire
        clear_pause_flag(ctrl)
        touch_flatten_flag(ctrl)  # triggers USER_FLATTEN on next tick

    await asyncio.wait_for(
        asyncio.gather(
            m.watchdog_loop(
                ws=ws, db_path=db, budget=bt, providers=["g"],
                risk=risk, live_match_fn=lambda: True, interval_s=0.01,
            ),
            stopper(),
        ),
        timeout=2.0,
    )
    assert m.is_killed()
    assert m.trip_event().reason is TripReason.USER_FLATTEN
