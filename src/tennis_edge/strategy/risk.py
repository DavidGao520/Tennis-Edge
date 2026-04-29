"""Risk management: position limits, exposure caps, kill-switch.

Phase 2: concurrency-safe. The Agent (Option 3) runs multiple candidates
through an LLM queue; two candidates can finish analysis simultaneously
and both call into the risk manager before either has placed an order.
The previous split API (``check_trade`` followed by ``record_trade``) had
a TOCTOU race that let combined exposure exceed the configured cap.

The atomic API is ``check_and_reserve`` — check limits and reserve
exposure under a single lock. If the downstream order placement fails,
call ``release`` to unwind the reservation. Idempotent.

               check_and_reserve                release
                       │                           │
                       ▼                           ▼
                ┌─────────────┐            ┌──────────────┐
                │ acquire lock│            │ acquire lock │
                │ validate    │            │ subtract     │
                │ record      │            │ exposure     │
                │ release lock│            │ release lock │
                └─────────────┘            └──────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..config import RiskConfig
from .sizing import BetDecision

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Tracks current risk exposure."""
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> exposure
    kill_switch_active: bool = False


class RiskManager:
    """Pre-trade validation and risk controls.

    The public API is async because the expected caller (agent executor)
    runs inside an asyncio event loop and may have multiple candidates
    racing. Use ``check_and_reserve`` to atomically validate-and-record,
    ``release`` to unwind a reservation if the order fails, and
    ``record_settlement`` when a market resolves.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()
        self._lock = asyncio.Lock()

    async def check_and_reserve(self, decision: BetDecision) -> tuple[bool, str]:
        """Atomically validate limits and reserve exposure.

        Returns:
            (allowed, reason). When allowed, exposure is already recorded.
            Caller must invoke ``release(decision)`` if the order then
            fails to execute.
        """
        async with self._lock:
            ok, reason = self._check_unlocked(decision)
            if ok:
                self._record_unlocked(decision)
            return ok, reason

    async def release(self, decision: BetDecision) -> None:
        """Undo a reservation whose downstream order failed.

        Idempotent: safe to call if nothing was reserved or if the
        ticker's exposure has already been reduced below ``bet_amount``
        (only unwinds up to the currently-held exposure).
        """
        async with self._lock:
            current = self.state.positions.get(decision.ticker, 0.0)
            unwind = min(decision.bet_amount, current)
            if unwind <= 0:
                return
            remaining = current - unwind
            # Small float drift at zero: drop the entry so ``active_positions``
            # in summary() stays accurate.
            if remaining <= 1e-9:
                self.state.positions.pop(decision.ticker, None)
            else:
                self.state.positions[decision.ticker] = remaining
            self.state.total_exposure -= unwind

    async def record_settlement(self, ticker: str, pnl: float) -> None:
        """Update risk state after a market settles."""
        async with self._lock:
            exposure = self.state.positions.pop(ticker, 0.0)
            self.state.total_exposure -= exposure
            self.state.daily_pnl += pnl

    async def reset_daily(self) -> None:
        """Reset daily PnL counter. Call at start of each trading day."""
        async with self._lock:
            self.state.daily_pnl = 0.0
            self.state.kill_switch_active = False

    def summary(self) -> dict:
        """Point-in-time snapshot. Read-only, lock-free."""
        return {
            "total_exposure": self.state.total_exposure,
            "daily_pnl": self.state.daily_pnl,
            "active_positions": len(self.state.positions),
            "kill_switch": self.state.kill_switch_active,
        }

    # --- internal helpers; must be called under self._lock ---

    def _check_unlocked(self, decision: BetDecision) -> tuple[bool, str]:
        if self.state.kill_switch_active or self.config.kill_switch:
            return False, "Kill switch is active"

        if self.state.daily_pnl <= -self.config.daily_loss_limit:
            self.state.kill_switch_active = True
            logger.warning("KILL SWITCH: Daily loss limit hit (%.2f)", self.state.daily_pnl)
            return False, f"Daily loss limit reached: {self.state.daily_pnl:.2f}"

        current_position = self.state.positions.get(decision.ticker, 0.0)
        new_position = current_position + decision.bet_amount
        if new_position > self.config.max_position_per_market:
            return False, (
                f"Per-market limit: {new_position:.2f} > {self.config.max_position_per_market:.2f}"
            )

        new_total = self.state.total_exposure + decision.bet_amount
        if new_total > self.config.max_total_exposure:
            return False, (
                f"Total exposure limit: {new_total:.2f} > {self.config.max_total_exposure:.2f}"
            )

        return True, "OK"

    def _record_unlocked(self, decision: BetDecision) -> None:
        self.state.total_exposure += decision.bet_amount
        current = self.state.positions.get(decision.ticker, 0.0)
        self.state.positions[decision.ticker] = current + decision.bet_amount
