"""Phase 3 v2 Agent main loop.

Replaces the v1 DB-tail reader. Now subscribes to MonitorBridge signals
via async callback, runs each candidate through grounded Gemini, and
calls `exchange.place_order` (paper or live) after a multi-stage hard
gate. The whole point of v2 is to fix v1's silent failure modes:
agent now refuses to trade against settled markets and refuses to
trade with stale edge.

Pipeline
========

       MonitorBridge                                   (every 15s)
            │
            ▼
    on_signal(sig)
            │
            ├── safety paused or killed? drop
            ├── cooldown active? drop
            ├── queue full? drop
            └── enqueue Candidate
                            │
                            ▼
                    LLM worker (1x)
                            │
                            ├── safety not running? drop
                            ├── candidate stale (>60s queued)? drop
                            ├── prompt_builder(sig) returns None? skip
                            ├── Gemini grounded analyze
                            │       LLMError? record_llm_failure, no order
                            │       parse error? record_llm_failure, no order
                            ├── confidence == "low"? HARD reject
                            ├── grounded edge < min? HARD reject
                            ├── post-LLM edge re-check < hard? HARD reject
                            ├── Kelly size + confidence multiplier
                            ├── risk.check_and_reserve fails? log, no order
                            ├── exchange.place_order(client_order_id=decision_id)
                            │       exception? release reservation, log
                            └── append AgentDecision

Hard rejects (vs v1 soft):
- v1 logged "edge_stale" but still left executed=False open in shadow.
  v2 also logs reject_reason but the executor was never going to run
  in those branches; in live mode, the order absolutely does not go
  to Kalshi.

Mode flag (`shadow` / `auto`) controls whether the executor branch
fires at all. In `shadow` we still call place_order on the paper
engine -- that is what `--executor paper` is for. The semantic
difference vs v1 is that we log a real `order_id` instead of
hardcoding executed=False.
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

from ..exchange.base import ExchangeClient
from ..exchange.schemas import OrderRequest
from ..strategy.risk import RiskManager
from .decisions import AgentDecision, DecisionLog, EvAnalysis, prompt_hash
from .llm import (
    BudgetExceeded,
    LLMError,
    LLMProvider,
    PROMPT_TEMPLATE_GROUNDED_V1,
    PromptContext,
)
from .monitor_bridge import MonitorSignal
from .safety import SafetyMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLoopConfig:
    """Tunables for the v2 main loop.

    queue_max
        Hard cap on in-flight candidates. Overflow drops with warning.

    cooldown_s
        Per-ticker cooldown window after a signal enters the queue.
        Prevents the same market from spamming the queue while the
        first candidate awaits the LLM.

    max_candidate_age_s
        At dequeue, drop anything older. Protects against burst
        signals during a Gemini stall.

    min_grounded_edge
        Gemini's `edge_estimate` minus market price below this in
        absolute terms means SKIP. Default 0.10 = 10pp. Below this
        the trade is not worth real money even if confidence is high.

    stale_edge_hard_threshold
        After LLM returns, re-read the latest tick. If |edge| against
        the most recent market price is below this, HARD reject the
        order. Default 0.08. Tighter than min_grounded_edge to catch
        markets that moved during Gemini's 18-30s think.

    kelly_fraction
        Fraction of full Kelly to bet (variance survival). Default
        0.25 matches Phase 1 PositionSizer.

    bankroll
        Used as the base for Kelly sizing. Capped at
        max_position_per_market regardless. Default $1000 keeps tests
        deterministic; CLI sets this from KalshiClient balance.

    max_position_per_market
        Hard $ cap per ticker. Default $50 matches v2 plan.

    confidence_mult_*
        Kelly multiplier per Gemini confidence bucket. low always =
        0.0 (no trade). medium = 0.5, high = 1.0. Plan flagged these
        as seed values requiring recalibration after 50 settled
        trades.

    mode
        "shadow" or "auto". Both call exchange.place_order. The
        difference is in which exchange is wired (paper vs live)
        and whether shadow analytics treats the result as
        counterfactual or realized -- handled at the analytics
        layer, not here.
    """

    queue_max: int = 20
    cooldown_s: float = 300.0
    max_candidate_age_s: float = 60.0

    min_grounded_edge: float = 0.10
    stale_edge_hard_threshold: float = 0.08

    kelly_fraction: float = 0.25
    bankroll: float = 1000.0
    max_position_per_market: float = 50.0

    confidence_mult_high: float = 1.0
    confidence_mult_medium: float = 0.5
    confidence_mult_low: float = 0.0

    mode: str = "shadow"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One queued analysis attempt."""

    decision_id: str
    signal: MonitorSignal
    enqueued_at: float  # time.monotonic()


# ---------------------------------------------------------------------------
# Prompt builder protocol
# ---------------------------------------------------------------------------


PromptBuilder = Callable[[MonitorSignal], Awaitable["PromptContext | None"]]
"""Build a PromptContext from a MonitorSignal. May fetch additional
metadata (market title, tournament, surface) over the network.
Returns None to skip this candidate without recording an LLM failure
(e.g., Kalshi metadata fetch failed)."""


# ---------------------------------------------------------------------------
# Tick re-check helper (no DB tailing — query-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TickRow:
    ticker: str
    received_at: int
    yes_bid: int | None
    yes_ask: int | None
    last_price: int | None


class _TickReader:
    """Read-only point lookup against market_ticks. v2 uses this only
    for the post-LLM edge re-check; the v1 tailing reader is gone.

    Single-method API keeps the surface minimal: callers query
    `latest_for_ticker(ticker)` and either get a row or None.
    """

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = Path(db_path)

    def latest_for_ticker(self, ticker: str) -> _TickRow | None:
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
        """Best-effort YES price: last → mid → ask → bid → None."""
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
    """v2 agent. Subscribes to MonitorBridge, runs grounded Gemini,
    sizes via Kelly, executes via injected ExchangeClient."""

    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        safety: SafetyMonitor,
        llm: LLMProvider,
        decisions: DecisionLog,
        risk: RiskManager,
        exchange: ExchangeClient,
        prompt_builder: PromptBuilder,
        tick_db_path: str | os.PathLike[str],
        run_id: str | None = None,
    ):
        self.config = config
        self.safety = safety
        self.llm = llm
        self.decisions = decisions
        self.risk = risk
        self.exchange = exchange
        self.prompt_builder = prompt_builder
        self.tick_reader = _TickReader(tick_db_path)
        self.queue: asyncio.Queue[Candidate] = asyncio.Queue(maxsize=config.queue_max)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

        self._cooldown_until: dict[str, float] = {}
        self._stop = asyncio.Event()
        self._consecutive_order_failures = 0

    # ---- public control ----

    def request_stop(self) -> None:
        self._stop.set()

    # ---- pub/sub endpoint (called by MonitorBridge) ----

    async def on_signal(self, sig: MonitorSignal) -> None:
        """MonitorBridge entry point. Applies pre-LLM gates and queues
        the candidate. Idempotent per ticker via cooldown."""
        if self.safety.is_killed():
            return
        if self.safety.is_paused():
            return

        now = time.monotonic()
        cooldown_end = self._cooldown_until.get(sig.ticker, 0.0)
        if cooldown_end > now:
            return

        candidate = Candidate(
            decision_id=f"dec-{uuid.uuid4().hex[:12]}",
            signal=sig,
            enqueued_at=now,
        )
        try:
            self.queue.put_nowait(candidate)
        except asyncio.QueueFull:
            logger.warning(
                "agent: queue full (%d), dropping signal ticker=%s ev=%.3f",
                self.config.queue_max, sig.ticker, sig.prematch_ev,
            )
            return

        # Cooldown starts at enqueue time, not LLM return — prevents
        # the same market from spamming the queue while the first
        # candidate awaits Gemini.
        self._cooldown_until[sig.ticker] = now + self.config.cooldown_s

    # ---- run ----

    async def run(self) -> None:
        """Spawn the LLM worker. Exits on stop or safety kill."""
        logger.info(
            "agent loop v2 starting run_id=%s mode=%s",
            self.run_id, self.config.mode,
        )
        worker = asyncio.create_task(self._worker_loop(), name="agent-worker")

        try:
            await self._wait_until_done()
        finally:
            worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass
            logger.info("agent loop v2 stopped run_id=%s", self.run_id)

    async def _wait_until_done(self) -> None:
        while not self._stop.is_set():
            if self.safety.is_killed():
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

    # ---- worker ----

    async def _worker_loop(self) -> None:
        try:
            while True:
                candidate = await self.queue.get()
                try:
                    await self._handle_candidate(candidate)
                except Exception:
                    logger.exception(
                        "agent: unexpected failure handling %s",
                        candidate.signal.ticker,
                    )
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            return

    async def _handle_candidate(self, c: Candidate) -> None:
        sig = c.signal

        # Safety state may have flipped since enqueue.
        if not self.safety.is_running():
            logger.info(
                "agent worker: safety=%s, dropping %s",
                self.safety.state().value, sig.ticker,
            )
            return

        # Freshness gate: candidate may have queued behind a slow LLM.
        age = time.monotonic() - c.enqueued_at
        if age > self.config.max_candidate_age_s:
            logger.info(
                "agent worker: dropping stale candidate ticker=%s age=%.1fs",
                sig.ticker, age,
            )
            return

        ctx = await self.prompt_builder(sig)
        if ctx is None:
            # Metadata enrichment failed; not an LLM error since we
            # never called the LLM.
            logger.info("agent worker: no prompt context for %s, skipping",
                        sig.ticker)
            return

        # ---- LLM ----
        try:
            result = await self.llm.analyze(ctx)
        except BudgetExceeded as e:
            logger.warning("agent worker: budget exceeded: %s", e)
            await self.safety.record_llm_failure(e)
            return
        except LLMError as e:
            logger.warning("agent worker: LLM failure (%s): %s", type(e).__name__, e)
            await self.safety.record_llm_failure(e)
            return
        except Exception as e:
            logger.exception("agent worker: unexpected LLM error")
            await self.safety.record_llm_failure(LLMError(f"unexpected: {e}"))
            return

        await self.safety.record_llm_success()

        analysis = result.analysis

        # ---- decision gates (each gate may produce a logged-but-not-executed record) ----

        reject_reason = self._check_grounded_edge(analysis, sig.market_yes_cents)

        edge_at_exec: float | None = None
        if reject_reason is None:
            edge_at_exec, stale_reject = self._post_llm_edge_check(
                analysis, sig.ticker,
            )
            if stale_reject is not None:
                reject_reason = stale_reject

        # ---- size + risk reservation ----

        order_request: OrderRequest | None = None
        order_id: str | None = None

        if reject_reason is None:
            sized = self._size_kelly(
                analysis, sig.market_yes_cents, c.decision_id,
            )
            if sized is None:
                reject_reason = "kelly_zero"
            else:
                order_request = sized
                # Risk manager check_and_reserve is the last hard gate
                # before committing any capital.
                ok, reason = await self.risk.check_and_reserve(
                    self._risk_decision_for(order_request, sig),
                )
                if not ok:
                    reject_reason = f"risk: {reason}"
                    order_request = None

        # ---- execute ----

        if order_request is not None:
            order_id = await self._place_order_safely(
                order_request, sig, analysis, c.decision_id,
            )
            if order_id is None:
                reject_reason = "order_failed"

        # ---- log ----

        ph = prompt_hash(PROMPT_TEMPLATE_GROUNDED_V1, {
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
            ticker=sig.ticker,
            model_pre_match=sig.model_prob,
            market_yes_cents=sig.market_yes_cents,
            edge_at_decision=sig.model_prob - sig.market_prob,
            llm_provider=result.provider,
            llm_prompt_hash=ph,
            llm_raw_output=result.raw_output,
            analysis=analysis,
            screenshot_paths=[],
            mode=self.config.mode,  # type: ignore[arg-type]
            executed=(order_id is not None),
            order_id=order_id,
            edge_at_execution=edge_at_exec,
            reject_reason=reject_reason,
        )
        self.decisions.append_decision(decision)
        logger.info(
            "decision: %s rec=%s conf=%s ev_est=%.3f edge_exec=%s exec=%s%s",
            sig.ticker, analysis.recommendation, analysis.confidence,
            analysis.edge_estimate,
            f"{edge_at_exec:.3f}" if edge_at_exec is not None else "n/a",
            order_id or "no",
            f" reject={reject_reason}" if reject_reason else "",
        )

    # ---- decision-gate helpers ----

    def _check_grounded_edge(
        self, analysis: EvAnalysis, market_yes_cents: int,
    ) -> str | None:
        """Hard reject if Gemini's claimed edge or confidence is too low."""
        if analysis.confidence == "low":
            return "confidence_low"

        market_prob = market_yes_cents / 100.0

        # For SKIP, edge is 0 by definition.
        if analysis.recommendation == "SKIP":
            return "rec_skip"

        if analysis.recommendation == "BUY_YES":
            grounded_edge = analysis.edge_estimate - market_prob
        else:  # BUY_NO
            grounded_edge = (1.0 - analysis.edge_estimate) - (1.0 - market_prob)

        if grounded_edge < self.config.min_grounded_edge:
            return "grounded_edge_below_min"
        return None

    def _post_llm_edge_check(
        self, analysis: EvAnalysis, ticker: str,
    ) -> tuple[float | None, str | None]:
        """Re-read latest tick. Hard reject if edge collapsed.

        Returns (edge_at_execution, reject_reason). edge_at_execution
        may be non-None even when we reject — useful for shadow analytics.
        """
        latest = self.tick_reader.latest_for_ticker(ticker)
        if latest is None:
            # No fresh tick. Be conservative — reject rather than fill
            # at a possibly stale price. tick-logger gap is also a
            # safety concern that the watchdog handles separately.
            return None, "no_recent_tick"

        latest_price = self.tick_reader.price_cents(latest)
        if latest_price is None:
            return None, "no_recent_price"

        market_prob = latest_price / 100.0
        if analysis.recommendation == "BUY_YES":
            edge_live = analysis.edge_estimate - market_prob
        else:  # BUY_NO
            edge_live = (1.0 - analysis.edge_estimate) - (1.0 - market_prob)

        if abs(edge_live) < self.config.stale_edge_hard_threshold:
            return edge_live, "edge_stale"

        return edge_live, None

    def _size_kelly(
        self, analysis: EvAnalysis, market_yes_cents: int, decision_id: str,
    ) -> OrderRequest | None:
        """Kelly + confidence multiplier + per-market cap. None to skip."""
        if analysis.recommendation == "BUY_YES":
            side = "yes"
            cost_per_cents = market_yes_cents
            true_p = analysis.edge_estimate
        elif analysis.recommendation == "BUY_NO":
            side = "no"
            cost_per_cents = 100 - market_yes_cents
            true_p = 1.0 - analysis.edge_estimate
        else:
            return None  # SKIP

        if cost_per_cents <= 0 or cost_per_cents >= 100:
            return None
        cost_prob = cost_per_cents / 100.0

        # Fractional Kelly: f = (p - cost_prob) / (1 - cost_prob)
        raw_kelly = (true_p - cost_prob) / (1.0 - cost_prob)
        if raw_kelly <= 0:
            return None

        conf_mult = {
            "high": self.config.confidence_mult_high,
            "medium": self.config.confidence_mult_medium,
            "low": self.config.confidence_mult_low,
        }[analysis.confidence]
        if conf_mult <= 0:
            return None

        fraction = raw_kelly * self.config.kelly_fraction * conf_mult
        bet_amount = min(
            fraction * self.config.bankroll,
            self.config.max_position_per_market,
        )

        num_contracts = max(1, int(bet_amount * 100 / cost_per_cents))
        # Recompute actual bet from integer contracts so risk reservation
        # matches the wire amount.
        actual_bet_dollars = num_contracts * cost_per_cents / 100.0
        if actual_bet_dollars <= 0:
            return None

        # Phase 3A: limit orders at the observed market price.
        # client_order_id is the decision_id — same UUID survives any
        # network-timeout retry so Kalshi treats it as the same order.
        return OrderRequest(
            ticker=analysis.recommendation,  # placeholder, overwritten below
            action="buy",
            side=side,  # type: ignore[arg-type]
            type="limit",
            count=num_contracts,
            yes_price=market_yes_cents,  # YES limit price in cents
            client_order_id=decision_id,
        )

    def _risk_decision_for(self, order: OrderRequest, sig: MonitorSignal):
        """Adapt OrderRequest to the BetDecision shape RiskManager needs.

        Uses `sig.ticker` rather than `order.ticker` because the
        OrderRequest carries a placeholder ticker until
        `_place_order_safely` rewrites it. Risk tracking must use the
        authoritative market identifier from the signal, not the
        placeholder, or exposure attribution drifts to the wrong key.
        """
        from ..strategy.sizing import BetDecision

        cost_per_cents = (
            sig.market_yes_cents if order.side == "yes"
            else 100 - sig.market_yes_cents
        )
        bet_amount = (order.count or 0) * cost_per_cents / 100.0
        return BetDecision(
            ticker=sig.ticker,
            side=order.side,
            model_prob=sig.model_prob,
            market_prob=sig.market_prob,
            edge=sig.model_prob - sig.market_prob,
            kelly_frac=0.0,  # not used in the check
            bet_amount=bet_amount,
            num_contracts=order.count or 0,
        )

    # ---- executor ----

    async def _place_order_safely(
        self,
        order: OrderRequest,
        sig: MonitorSignal,
        analysis: EvAnalysis,
        decision_id: str,
    ) -> str | None:
        """Wrap exchange.place_order. On failure, release the risk
        reservation and bump the consecutive-failure counter.

        Returns the order_id on success, None on failure.
        """
        # The OrderRequest from _size_kelly carries the recommendation
        # in `ticker` as a placeholder; fix it here so the executor
        # actually targets the market.
        order = order.model_copy(update={"ticker": sig.ticker})

        try:
            resp = await self.exchange.place_order(order)
        except Exception:
            logger.exception(
                "agent worker: place_order failed ticker=%s side=%s",
                sig.ticker, order.side,
            )
            await self.risk.release(self._risk_decision_for(order, sig))
            self._consecutive_order_failures += 1
            if self._consecutive_order_failures >= 3:
                # Persistent execution failures are a separate failure
                # mode from LLM problems. Use the dedicated
                # ORDER_CONSECUTIVE_FAILURES TripReason so post-mortem
                # analytics can distinguish "the LLM was flaky" from
                # "Kalshi was rejecting our orders".
                from .safety import TripReason as _TripReason
                await self.safety.kill(
                    _TripReason.ORDER_CONSECUTIVE_FAILURES,
                    f"3 consecutive place_order failures (last ticker={sig.ticker})",
                )
            return None

        self._consecutive_order_failures = 0
        return resp.order_id or None

    # ---- testing helpers ----

    async def drain_once(self) -> bool:
        """Process exactly one queued candidate (non-blocking).

        Used by tests to drive the loop deterministically.
        """
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
        end = self._cooldown_until.get(ticker, 0.0)
        return max(0.0, end - time.monotonic())
