"""Risk management: position limits, exposure caps, kill-switch."""

from __future__ import annotations

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
    """Pre-trade validation and risk controls."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    def check_trade(self, decision: BetDecision) -> tuple[bool, str]:
        """Validate whether a trade is allowed given current risk state.

        Returns:
            (allowed, reason)
        """
        if self.state.kill_switch_active or self.config.kill_switch:
            return False, "Kill switch is active"

        # Check daily loss limit
        if self.state.daily_pnl <= -self.config.daily_loss_limit:
            self.state.kill_switch_active = True
            logger.warning("KILL SWITCH: Daily loss limit hit (%.2f)", self.state.daily_pnl)
            return False, f"Daily loss limit reached: {self.state.daily_pnl:.2f}"

        # Check per-market position limit
        current_position = self.state.positions.get(decision.ticker, 0.0)
        new_position = current_position + decision.bet_amount
        if new_position > self.config.max_position_per_market:
            return False, (
                f"Per-market limit: {new_position:.2f} > {self.config.max_position_per_market:.2f}"
            )

        # Check total exposure cap
        new_total = self.state.total_exposure + decision.bet_amount
        if new_total > self.config.max_total_exposure:
            return False, (
                f"Total exposure limit: {new_total:.2f} > {self.config.max_total_exposure:.2f}"
            )

        return True, "OK"

    def record_trade(self, decision: BetDecision) -> None:
        """Update risk state after a trade executes."""
        self.state.total_exposure += decision.bet_amount
        current = self.state.positions.get(decision.ticker, 0.0)
        self.state.positions[decision.ticker] = current + decision.bet_amount

    def record_settlement(self, ticker: str, pnl: float) -> None:
        """Update risk state after a market settles."""
        exposure = self.state.positions.pop(ticker, 0.0)
        self.state.total_exposure -= exposure
        self.state.daily_pnl += pnl

    def reset_daily(self) -> None:
        """Reset daily PnL counter (call at start of each trading day)."""
        self.state.daily_pnl = 0.0
        self.state.kill_switch_active = False

    def summary(self) -> dict:
        return {
            "total_exposure": self.state.total_exposure,
            "daily_pnl": self.state.daily_pnl,
            "active_positions": len(self.state.positions),
            "kill_switch": self.state.kill_switch_active,
        }
