"""Phase 2 Agent main loop. Shadow mode only (Phase 3A).

Pipeline
========
                                                       (every ~2s)
    tick_logger ──► market_ticks ──► MarketTickReader ─┐
    (Mac mini)      (SQLite tail)                      │
                                                       ▼
                                      ┌─────────────────────────┐
                                      │  edge = model - market  │
                                      │  cooldown per ticker    │
                                      │  bounded queue cap      │
                                      └────────────┬────────────┘
                                                   ▼
                                           CandidateQueue
                                                   │
                                                   ▼
                                      ┌─────────────────────────┐
                                      │    LLM worker (1x)      │
                                      │    is_running? skip     │
                                      │    context_builder()    │
                                      │    provider.analyze()   │
                                      │    err: record_failure  │
                                      │    ok:  record_success  │
                                      │    re-check edge stale  │
                                      │    log AgentDecision    │
                                      └─────────────────────────┘

                         watchdog_loop runs in parallel
                         (budget / P&L / WS / tick-logger / flags)

Phase 3A: mode="shadow" — executor is never invoked, every decision
appended with executed=False. Phase 3B/3C will add the execute step.

Key invariants
==============
1. Single LLM worker coroutine. At most one in-flight call at a time.
   LLM latency (5-15s on Gemini 3.x thinking) dominates decision
   freshness; parallelism would only make stale-edge worse.

2. Per-ticker cooldown (default 5 min). Same market cannot enter the
   queue again during cooldown window.

3. Bounded queue cap (default 20). Overflow drops the new candidate
   with a log line — never block the reader, never starve the worker.

4. Candidate freshness gate at dequeue. If a candidate sat in the
   queue longer than max_candidate_age_s, drop without LLM call.

5. Post-LLM edge re-validation. Re-read latest tick after the LLM
   returns; if |edge| collapsed below threshold, log the decision
   with reject_reason="edge_stale" — NOT executed — but still
   append to JSONL so shadow analytics can measure LLM latency's
   cost in missed signals.

6. Every LLM error routes through safety.record_llm_failure so the
   3x-consecutive kill switch from Lane 1D can count.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from .decisions import AgentDecision, DecisionLog, prompt_hash
from .llm import (
    BudgetExceeded,
    LLMError,
    LLMProvider,
    PROMPT_TEMPLATE_V1,
    PromptContext,
)
from .safety import SafetyMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLoopConfig:
    """Tunables. Defaults chosen per Phase 2 eng review.

    min_edge               — abs(model - market) below this is dropped
                             pre-queue. 0.08 = 8 percentage points,
                             higher than Phase 1 scanner's 3% so we
                             only burn LLM dollars on real signals.
    cooldown_s             — per-ticker cooldown. 300 = 5 min.
    queue_max              — hard cap on in-flight candidates.
    max_candidate_age_s    — at dequeue, drop anything older.
    tick_poll_interval_s   — how often MarketTickReader polls the DB.
    stale_edge_threshold   — |edge| below this after LLM is "edge stale".
                             Set to min_edge by default; operator can
                             widen it if they want to capture near-miss
                             signals in the log for analysis.
    mode                   — "shadow" (Phase 3A), "human_in_loop" (3B),
                             "auto" (3C). Only shadow wired today.
    """

    min_edge: float = 0.08
    cooldown_s: float = 300.0
    queue_max: int = 20
    max_candidate_age_s: float = 60.0
    tick_poll_interval_s: float = 2.0
    stale_edge_threshold: float | None = None  # None → use min_edge
    mode: str = "shadow"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One queued trade opportunity."""

    decision_id: str
    ticker: str
    model_prob: float
    market_yes_cents: int
    edge_at_decision: float
    enqueued_at: float  # time.monotonic()


# ---------------------------------------------------------------------------
# Injected deps (structural typing)
# ---------------------------------------------------------------------------


class _ModelProbFn(Protocol):
    """Returns the pre-match P(YES wins) for a ticker, or None if the
    ticker is unknown (e.g. unrated player)."""

    def __call__(self, ticker: str) -> float | None: ...


class _ContextBuilder(Protocol):
    """Builds the LLM prompt context for a ticker.

    Typically queries the DB for player names, tournament, form, H2H,
    rest days, etc. Returns None if the ticker cannot be enriched
    (missing player data, unknown tournament) — the loop treats that
    as a skip without counting it as an LLM failure.
    """

    def __call__(
        self, ticker: str, model_prob: float, market_yes_cents: int
    ) -> PromptContext | None: ...


# ---------------------------------------------------------------------------
# Tick reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TickRow:
    ticker: str
    received_at: int
    yes_bid: int | None
    yes_ask: int | None
    last_price: int | None


class MarketTickReader:
    """Tails market_ticks. Returns the latest row per ticker since cursor.

    Opens SQLite read-only so a stuck reader cannot block the tick
    logger's writer. The cursor is an integer received_at; the first
    read initializes it to "now" so we do not flood the queue with
    every historical tick on startup.
    """

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = Path(db_path)
        self._cursor: int = int(time.time())

    def latest_per_ticker(self) -> list[_TickRow]:
        """One row per ticker, the most recent since cursor."""
        uri = f"file:{os.fspath(self.db_path)}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        except sqlite3.OperationalError as e:
            logger.warning("tick reader: cannot open %s: %s", self.db_path, e)
            return []

        try:
            cur = conn.execute(
                """
                SELECT ticker, MAX(received_at) AS received_at,
                       yes_bid, yes_ask, last_price
                FROM (
                    SELECT ticker, received_at, yes_bid, yes_ask, last_price
                    FROM market_ticks
                    WHERE received_at > ?
                    ORDER BY received_at DESC
                )
                GROUP BY ticker
                """,
                (self._cursor,),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("tick reader: query failed: %s", e)
            return []
        finally:
            conn.close()

        if not rows:
            return []

        # Advance cursor to newest received_at we saw. Any later insert
        # this same second will be picked up next poll (acceptable —
        # we are not running HFT).
        max_ts = max(int(r[1]) for r in rows)
        self._cursor = max_ts

        out: list[_TickRow] = []
        for r in rows:
            out.append(_TickRow(
                ticker=r[0],
                received_at=int(r[1]),
                yes_bid=r[2],
                yes_ask=r[3],
                last_price=r[4],
            ))
        return out

    def latest_for_ticker(self, ticker: str) -> _TickRow | None:
        """Point-lookup used for post-LLM edge re-check. Does not
        advance the cursor (that belongs to the tailing poll)."""
        uri = f"file:{os.fspath(self.db_path)}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        except sqlite3.OperationalError:
            return None
        try:
            cur = conn.execute(
                """
                SELECT ticker, received_at, yes_bid, yes_ask, last_price
                FROM market_ticks
                WHERE ticker = ?
                ORDER BY received_at DESC
                LIMIT 1
                """,
                (ticker,),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        return _TickRow(
            ticker=row[0], received_at=int(row[1]),
            yes_bid=row[2], yes_ask=row[3], last_price=row[4],
        )

    @staticmethod
    def price_cents(row: _TickRow) -> int | None:
        """Best-effort YES price in cents from a tick row.

        Preference order: last_price → mid of bid/ask → ask → bid.
        Used for edge calc only. Not a fill price.
        """
        if row.last_price is not None:
            return int(row.last_price)
        if row.yes_bid is not None and row.yes_ask is not None:
            return int((row.yes_bid + row.yes_ask) / 2)
        if row.yes_ask is not None:
            return int(row.yes_ask)
        if row.yes_bid is not None:
            return int(row.yes_bid)
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Orchestrates reader, queue, LLM worker. Shadow mode only."""

    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        db_path: str | os.PathLike[str],
        safety: SafetyMonitor,
        llm: LLMProvider,
        decisions: DecisionLog,
        model_prob_fn: _ModelProbFn,
        context_builder: _ContextBuilder,
        run_id: str | None = None,
    ):
        self.config = config
        self.safety = safety
        self.llm = llm
        self.decisions = decisions
        self.model_prob_fn = model_prob_fn
        self.context_builder = context_builder
        self.reader = MarketTickReader(db_path)
        self.queue: asyncio.Queue[Candidate] = asyncio.Queue(maxsize=config.queue_max)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

        self._cooldown_until: dict[str, float] = {}
        self._stop = asyncio.Event()

    # ---- public control ----

    def request_stop(self) -> None:
        """Clean shutdown from outside (signal handler, CLI, tests)."""
        self._stop.set()

    def _stale_threshold(self) -> float:
        return (
            self.config.stale_edge_threshold
            if self.config.stale_edge_threshold is not None
            else self.config.min_edge
        )

    # ---- run ----

    async def run(self) -> None:
        """Main entry. Runs reader + worker until safety KILLED or stop()."""
        logger.info("agent loop starting run_id=%s mode=%s", self.run_id, self.config.mode)
        reader_task = asyncio.create_task(self._reader_loop(), name="reader")
        worker_task = asyncio.create_task(self._worker_loop(), name="worker")

        try:
            await self._wait_until_done()
        finally:
            reader_task.cancel()
            worker_task.cancel()
            for t in (reader_task, worker_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("agent loop stopped run_id=%s", self.run_id)

    async def _wait_until_done(self) -> None:
        """Sleep until stop requested or safety kills us."""
        while not self._stop.is_set():
            if self.safety.is_killed():
                logger.info("agent loop: safety KILLED, exiting")
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

    # ---- reader ----

    async def _reader_loop(self) -> None:
        try:
            while True:
                if self.safety.is_killed():
                    return
                if not self.safety.is_paused():
                    try:
                        rows = self.reader.latest_per_ticker()
                    except Exception:
                        logger.exception("reader: unexpected failure")
                        rows = []
                    for row in rows:
                        self._maybe_enqueue(row)
                await asyncio.sleep(self.config.tick_poll_interval_s)
        except asyncio.CancelledError:
            return

    def _maybe_enqueue(self, row: _TickRow) -> None:
        """Decide whether this tick deserves a queue slot."""
        now = time.monotonic()
        cooldown_end = self._cooldown_until.get(row.ticker, 0.0)
        if cooldown_end > now:
            return  # cooldown active, skip silently

        price = MarketTickReader.price_cents(row)
        if price is None:
            return

        model_prob = self.model_prob_fn(row.ticker)
        if model_prob is None:
            return  # unknown ticker, no Glicko anchor

        market_prob = price / 100.0
        edge = model_prob - market_prob
        if abs(edge) < self.config.min_edge:
            return

        candidate = Candidate(
            decision_id=f"dec-{uuid.uuid4().hex[:12]}",
            ticker=row.ticker,
            model_prob=model_prob,
            market_yes_cents=price,
            edge_at_decision=edge,
            enqueued_at=now,
        )

        try:
            self.queue.put_nowait(candidate)
        except asyncio.QueueFull:
            logger.warning(
                "queue full (%d), dropping candidate ticker=%s edge=%.3f",
                self.config.queue_max, row.ticker, edge,
            )
            return

        # Start the cooldown window the moment we enqueue. Prevents a
        # burst of rapid ticks for the same market from spamming the
        # queue while the first candidate still awaits the LLM.
        self._cooldown_until[row.ticker] = now + self.config.cooldown_s

    # ---- worker ----

    async def _worker_loop(self) -> None:
        try:
            while True:
                candidate = await self.queue.get()
                try:
                    await self._handle_candidate(candidate)
                except Exception:
                    logger.exception(
                        "worker: unexpected failure handling %s", candidate.ticker,
                    )
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            return

    async def _handle_candidate(self, c: Candidate) -> None:
        # Check monitor state fresh — may have tripped since enqueue.
        if not self.safety.is_running():
            logger.info(
                "worker: safety not running (%s), dropping %s",
                self.safety.state().value, c.ticker,
            )
            return

        # Freshness gate. Candidate may have sat in the queue while the
        # LLM worked through earlier items.
        age = time.monotonic() - c.enqueued_at
        if age > self.config.max_candidate_age_s:
            logger.info(
                "worker: dropping stale candidate ticker=%s age=%.1fs",
                c.ticker, age,
            )
            return

        ctx = self.context_builder(c.ticker, c.model_prob, c.market_yes_cents)
        if ctx is None:
            # Context builder couldn't enrich. Not an LLM failure (no
            # call made), but worth logging. Do NOT call record_llm_*.
            logger.info("worker: no context for %s, skipping", c.ticker)
            return

        # ---- LLM call ----
        try:
            result = await self.llm.analyze(ctx)
        except BudgetExceeded as e:
            # Budget is its own kill switch in safety; we still count
            # this as an LLM failure so the 3x counter registers.
            logger.warning("LLM budget exceeded: %s", e)
            await self.safety.record_llm_failure(e)
            return
        except LLMError as e:
            logger.warning("LLM failure (%s): %s", type(e).__name__, e)
            await self.safety.record_llm_failure(e)
            return
        except Exception as e:
            # Defensive: anything not in the LLMError hierarchy is a
            # bug we want to know about, but still count as failure so
            # a persistent bug eventually trips the kill switch.
            logger.exception("LLM unexpected error")
            await self.safety.record_llm_failure(LLMError(f"unexpected: {e}"))
            return

        await self.safety.record_llm_success()

        # ---- post-LLM edge re-check ----
        reject_reason: str | None = None
        edge_at_exec: float | None = None
        latest = self.reader.latest_for_ticker(c.ticker)
        if latest is not None:
            latest_price = MarketTickReader.price_cents(latest)
            if latest_price is not None:
                edge_at_exec = c.model_prob - latest_price / 100.0
                if abs(edge_at_exec) < self._stale_threshold():
                    reject_reason = "edge_stale"

        # ---- log decision ----
        # Note: in Phase 3A shadow, executed is always False regardless
        # of reject_reason. reject_reason=None just means "would have
        # traded in 3B/3C".
        ph = prompt_hash(PROMPT_TEMPLATE_V1, {
            "ticker": ctx.ticker,
            "player_yes": ctx.player_yes,
            "player_no": ctx.player_no,
            "model_pre_match": ctx.model_pre_match,
            "market_yes_cents": ctx.market_yes_cents,
        })

        decision = AgentDecision(
            ts=datetime.now(timezone.utc),
            run_id=self.run_id,
            decision_id=c.decision_id,
            ticker=c.ticker,
            model_pre_match=c.model_prob,
            market_yes_cents=c.market_yes_cents,
            edge_at_decision=c.edge_at_decision,
            llm_provider=result.provider,
            llm_prompt_hash=ph,
            llm_raw_output=result.raw_output,
            analysis=result.analysis,
            screenshot_paths=[],  # 3A: no screenshots
            mode=self.config.mode,  # type: ignore[arg-type]
            executed=False,        # 3A always logs-only
            order_id=None,
            edge_at_execution=edge_at_exec,
            reject_reason=reject_reason,
        )
        self.decisions.append_decision(decision)
        logger.info(
            "decision logged: %s edge=%.3f → %.3f rec=%s%s",
            c.ticker, c.edge_at_decision,
            edge_at_exec if edge_at_exec is not None else float("nan"),
            result.analysis.recommendation,
            f" REJECT:{reject_reason}" if reject_reason else "",
        )

    # ---- helpers for tests ----

    async def tick_once(self) -> None:
        """One-shot reader poll. Used by tests to drive deterministically."""
        rows = self.reader.latest_per_ticker()
        for row in rows:
            self._maybe_enqueue(row)

    async def drain_once(self) -> bool:
        """Handle exactly one candidate if present; return True if one
        was processed. Non-blocking."""
        try:
            c = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        try:
            await self._handle_candidate(c)
        finally:
            self.queue.task_done()
        return True

    def cooldown_remaining(self, ticker: str) -> float:
        """For tests: seconds left on a ticker cooldown, 0 if none."""
        end = self._cooldown_until.get(ticker, 0.0)
        return max(0.0, end - time.monotonic())
