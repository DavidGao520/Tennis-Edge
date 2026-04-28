"""Phase 2 Agent safety module.

Five kill switches + a file-flag IPC for user-initiated pause/flatten.
The daemon in Lane H polls these; no Q/K TTY keystroke plumbing lives
here (Phase 3B problem when the agent embeds in the monitor TUI).

                     SafetyMonitor state
                       ┌──────────┐
                       │ RUNNING  │
                       └────┬─────┘
        user pauses ────►   │       ◄──── user resumes
                       ┌────▼─────┐
                       │  PAUSED  │
                       └────┬─────┘
        any trip  ────►     │      ◄── irreversible: daemon exits
                       ┌────▼─────┐
                       │  KILLED  │
                       └──────────┘

Kill switches (any one flips RUNNING → KILLED):

  1. LLM_CONSECUTIVE_FAILURES     3 in a row, any LLMError subclass
  2. WS_RECONNECT_STARVATION      last_connect_ts > 60s (link down)
  3. WS_STALE_WITH_LIVE_MATCH     last_message_ts > 60s AND live match
  4. TICK_LOGGER_STALE            max(received_at) > 60s old (no writer)
  5. BUDGET_EXCEEDED              any provider remaining_usd <= 0
  6. DAILY_LOSS_LIMIT             risk manager daily_pnl <= -limit
  7. USER_FLATTEN (via flag)      data/agent_control/flatten exists

User pause (reversible):

  USER_PAUSE (via flag)           data/agent_control/pause exists

IPC: lockless flag files. CLI writes (touch), daemon reads (Path.exists).
Flatten flag is consumed: daemon reads, closes positions, deletes flag,
then kills itself. Pause flag is read every watchdog tick and can be
cleared by the user to resume.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from .llm import LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyConfig:
    """Tunables for kill switches.

    Defaults chosen per Phase 2 eng review:
      - 60s WS staleness (matches the original plan's disconnect rule)
      - 60s tick-logger staleness
      - 3 consecutive LLM failures (plan constant)
      - $200 daily loss floor (plan constant)
    """

    max_consecutive_llm_failures: int = 3
    ws_reconnect_stale_s: float = 60.0
    ws_message_stale_s: float = 60.0
    tick_logger_stale_s: float = 60.0
    daily_loss_limit_usd: float = 200.0
    control_dir: str = "data/agent_control"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SafetyState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    KILLED = "killed"


class TripReason(Enum):
    LLM_CONSECUTIVE_FAILURES = "llm_consecutive_failures"
    WS_RECONNECT_STARVATION = "ws_reconnect_starvation"
    WS_STALE_WITH_LIVE_MATCH = "ws_stale_with_live_match"
    TICK_LOGGER_STALE = "tick_logger_stale"
    BUDGET_EXCEEDED = "budget_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    USER_FLATTEN = "user_flatten"
    ORDER_CONSECUTIVE_FAILURES = "order_consecutive_failures"


@dataclass(frozen=True)
class TripEvent:
    ts: datetime
    reason: TripReason
    detail: str


# ---------------------------------------------------------------------------
# Structural typing for injected deps
# ---------------------------------------------------------------------------


class _WSProto(Protocol):
    """Subset of KalshiWebSocket that the watchdog reads."""

    def seconds_since_last_message(self) -> float | None: ...
    def seconds_since_last_connect(self) -> float | None: ...


class _BudgetProto(Protocol):
    def remaining_usd(self, provider: str) -> float: ...


class _RiskProto(Protocol):
    @property
    def state(self) -> object: ...  # has .daily_pnl float


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SafetyMonitor:
    """Owns agent state + kill-switch evaluation.

    The call pattern is:

      mon = SafetyMonitor(config)

      # On every LLM attempt:
      try:
          result = await provider.analyze(ctx)
          await mon.record_llm_success()
      except LLMError as e:
          await mon.record_llm_failure(e)
          if not mon.is_running():
              break

      # Periodic watchdog (spawned once):
      await mon.watchdog_loop(ws=ws, db=db, budget=bt,
                              providers=["gemini-3.1-pro-preview"],
                              risk=risk_mgr, live_match_fn=is_match_live,
                              interval_s=30.0)

    State is asyncio-lock guarded. is_running() / is_paused() are
    lock-free scalar reads; readers can poll at arbitrary frequency
    without contending with the watchdog.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config
        self._state = SafetyState.RUNNING
        self._trip: TripEvent | None = None
        self._llm_failures = 0
        self._lock = asyncio.Lock()

        self.control_dir = Path(config.control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.pause_flag = self.control_dir / "pause"
        self.flatten_flag = self.control_dir / "flatten"

    # ---- state ----

    def state(self) -> SafetyState:
        return self._state

    def is_running(self) -> bool:
        return self._state is SafetyState.RUNNING

    def is_paused(self) -> bool:
        return self._state is SafetyState.PAUSED

    def is_killed(self) -> bool:
        return self._state is SafetyState.KILLED

    def trip_event(self) -> TripEvent | None:
        return self._trip

    # ---- LLM counter ----

    async def record_llm_success(self) -> None:
        """Reset the consecutive-failure counter."""
        async with self._lock:
            self._llm_failures = 0

    async def record_llm_failure(self, err: LLMError) -> None:
        """Increment and trip at max_consecutive_llm_failures."""
        async with self._lock:
            self._llm_failures += 1
            count = self._llm_failures
        if count >= self.config.max_consecutive_llm_failures:
            await self._trip_once(
                TripReason.LLM_CONSECUTIVE_FAILURES,
                f"{count} consecutive LLM failures, last: {type(err).__name__}: {err}",
            )

    def consecutive_llm_failures(self) -> int:
        return self._llm_failures

    async def kill(self, reason: TripReason, detail: str) -> None:
        """Public hook for callers (e.g. AgentLoop's order-failure path)
        that need to flip state to KILLED with a specific TripReason
        without going through the LLM-failure rail.

        Idempotent: first trip wins. KILLED is terminal regardless of
        which switch fired.
        """
        await self._trip_once(reason, detail)

    # ---- user-initiated control ----

    async def check_user_flags(self) -> None:
        """Reconcile state against file flags. Call from watchdog_loop.

        pause flag:    RUNNING ⇆ PAUSED (reversible — delete flag to resume)
        flatten flag:  any state → trip USER_FLATTEN (the daemon then
                       closes positions and exits; deleting the flag
                       itself is the daemon's responsibility post-flatten)
        """
        if self.flatten_flag.exists() and not self.is_killed():
            await self._trip_once(
                TripReason.USER_FLATTEN,
                f"flatten flag present at {self.flatten_flag}",
            )
            return
        if self.is_killed():
            return

        # Pause is reversible. Re-check every tick so the user can flip
        # back and forth by touching / deleting the flag file.
        flag_exists = self.pause_flag.exists()
        async with self._lock:
            if flag_exists and self._state is SafetyState.RUNNING:
                logger.info("safety: user pause flag set → PAUSED")
                self._state = SafetyState.PAUSED
            elif not flag_exists and self._state is SafetyState.PAUSED:
                logger.info("safety: user pause flag cleared → RUNNING")
                self._state = SafetyState.RUNNING

    # ---- individual kill switches ----

    async def check_ws(
        self,
        ws: _WSProto,
        live_match_fn: Callable[[], bool] | Callable[[], Awaitable[bool]],
    ) -> None:
        """Two sub-checks: reconnect starvation, then message staleness
        conditional on a live match."""
        # Reconnect starvation. None means "never connected" — same bad.
        since_connect = ws.seconds_since_last_connect()
        if since_connect is None or since_connect > self.config.ws_reconnect_stale_s:
            await self._trip_once(
                TripReason.WS_RECONNECT_STARVATION,
                f"seconds_since_last_connect={since_connect}",
            )
            return

        # Message staleness only matters if a match is live. Support
        # both sync and async predicates.
        result = live_match_fn()
        if asyncio.iscoroutine(result):
            is_live = await result
        else:
            is_live = bool(result)
        if not is_live:
            return

        since_msg = ws.seconds_since_last_message()
        if since_msg is None or since_msg > self.config.ws_message_stale_s:
            await self._trip_once(
                TripReason.WS_STALE_WITH_LIVE_MATCH,
                f"live match but seconds_since_last_message={since_msg}",
            )

    async def check_tick_logger(self, db_path: str | os.PathLike[str]) -> None:
        """Trip if no tick-logger write within tick_logger_stale_s.

        SQLite is opened in read-only mode via URI so a missing DB or
        missing market_ticks table does not create anything. Missing
        table or empty DB counts as stale.
        """
        uri = f"file:{os.fspath(db_path)}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        except sqlite3.OperationalError as e:
            await self._trip_once(
                TripReason.TICK_LOGGER_STALE,
                f"cannot open tick DB read-only: {e}",
            )
            return

        try:
            cur = conn.execute("SELECT MAX(received_at) FROM market_ticks")
            row = cur.fetchone()
        except sqlite3.OperationalError as e:
            # Table missing, DB corrupt, etc. Treat as stale.
            await self._trip_once(
                TripReason.TICK_LOGGER_STALE,
                f"market_ticks query failed: {e}",
            )
            return
        finally:
            conn.close()

        max_received = row[0] if row else None
        if max_received is None:
            await self._trip_once(
                TripReason.TICK_LOGGER_STALE,
                "market_ticks empty — tick logger never wrote",
            )
            return

        now = int(time.time())
        age_s = now - int(max_received)
        if age_s > self.config.tick_logger_stale_s:
            await self._trip_once(
                TripReason.TICK_LOGGER_STALE,
                f"max(received_at) age={age_s}s",
            )

    async def check_budget(
        self,
        budget: _BudgetProto,
        providers: list[str],
    ) -> None:
        """Trip if any configured provider has hit its monthly cap."""
        for p in providers:
            if budget.remaining_usd(p) <= 0:
                await self._trip_once(
                    TripReason.BUDGET_EXCEEDED,
                    f"provider {p} over monthly cap",
                )
                return

    async def check_daily_pnl(self, risk: _RiskProto) -> None:
        """Trip if realized daily P&L hit the floor.

        Reads `risk.state.daily_pnl` directly. The Lane B risk manager
        already trips its own internal kill switch at this threshold;
        mirroring the decision into SafetyMonitor ensures the agent
        daemon exits instead of silently rejecting every candidate.
        """
        daily = float(getattr(risk.state, "daily_pnl", 0.0))
        if daily <= -self.config.daily_loss_limit_usd:
            await self._trip_once(
                TripReason.DAILY_LOSS_LIMIT,
                f"daily_pnl={daily:.2f} <= -{self.config.daily_loss_limit_usd:.2f}",
            )

    # ---- watchdog ----

    async def watchdog_loop(
        self,
        *,
        ws: _WSProto,
        db_path: str | os.PathLike[str],
        budget: _BudgetProto,
        providers: list[str],
        risk: _RiskProto,
        live_match_fn: Callable[[], bool] | Callable[[], Awaitable[bool]],
        interval_s: float = 30.0,
    ) -> None:
        """Run all periodic checks until the monitor is KILLED.

        Called once from the daemon; lives alongside the LLM worker
        and the DB-tail reader. Exits cleanly when any kill switch
        trips — the daemon checks is_killed() on its own schedule.
        """
        logger.info("safety: watchdog started, interval=%.0fs", interval_s)
        try:
            while not self.is_killed():
                await self.check_user_flags()
                if self.is_killed():
                    break

                # User pause short-circuits the other checks so we do
                # not trip while the user is intentionally holding state.
                if not self.is_paused():
                    await self.check_budget(budget, providers)
                    if self.is_killed():
                        break
                    await self.check_daily_pnl(risk)
                    if self.is_killed():
                        break
                    await self.check_ws(ws, live_match_fn)
                    if self.is_killed():
                        break
                    await self.check_tick_logger(db_path)
                    if self.is_killed():
                        break

                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    # ---- internal ----

    async def _trip_once(self, reason: TripReason, detail: str) -> None:
        """Idempotent trip: first cause wins, subsequent calls are no-ops.

        KILLED is terminal. If we are already KILLED we keep the first
        trip event so the daemon shutdown log is deterministic.
        """
        async with self._lock:
            if self._state is SafetyState.KILLED:
                return
            self._state = SafetyState.KILLED
            self._trip = TripEvent(
                ts=datetime.now(timezone.utc),
                reason=reason,
                detail=detail,
            )
        logger.error("safety KILLED: %s — %s", reason.value, detail)


# ---------------------------------------------------------------------------
# Flag-file IPC (CLI writers + daemon readers)
# ---------------------------------------------------------------------------


def touch_pause_flag(control_dir: str | os.PathLike[str]) -> Path:
    """CLI helper: create the pause flag so the running daemon pauses."""
    p = Path(control_dir) / "pause"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def clear_pause_flag(control_dir: str | os.PathLike[str]) -> None:
    """CLI helper: delete the pause flag so the daemon resumes."""
    p = Path(control_dir) / "pause"
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def touch_flatten_flag(control_dir: str | os.PathLike[str]) -> Path:
    """CLI helper: create the flatten flag. The daemon picks it up on
    the next watchdog tick, trips USER_FLATTEN, closes positions, and
    then deletes the flag itself before exiting."""
    p = Path(control_dir) / "flatten"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def clear_flatten_flag(control_dir: str | os.PathLike[str]) -> None:
    """Daemon helper: called after flatten is complete so the next
    daemon start does not immediately re-trip."""
    p = Path(control_dir) / "flatten"
    try:
        p.unlink()
    except FileNotFoundError:
        pass
