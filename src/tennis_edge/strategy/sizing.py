"""Position sizing with bankroll constraints."""

from __future__ import annotations

from dataclasses import dataclass

from .kelly import edge, fractional_kelly


@dataclass
class BetDecision:
    ticker: str
    side: str  # "yes" or "no"
    model_prob: float
    market_prob: float
    edge: float
    kelly_frac: float
    bet_amount: float  # in dollars
    num_contracts: int


class PositionSizer:
    """Determine bet size using fractional Kelly with constraints."""

    def __init__(
        self,
        bankroll: float,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.05,
        min_edge: float = 0.03,
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge

    def size(
        self,
        model_prob: float,
        market_price_cents: int,
        ticker: str = "",
    ) -> BetDecision | None:
        """Calculate position size for a potential trade.

        Args:
            model_prob: Model's estimated probability of YES outcome.
            market_price_cents: Kalshi YES price in cents (1-99).
            ticker: Market ticker for reference.

        Returns:
            BetDecision if edge >= min_edge, else None.
        """
        market_prob = market_price_cents / 100.0

        raw_edge = edge(model_prob, market_prob)
        abs_edge = abs(raw_edge)

        if abs_edge < self.min_edge:
            return None

        kelly = fractional_kelly(model_prob, market_prob, self.kelly_fraction)

        if kelly > 0:
            side = "yes"
            cost_per_contract = market_price_cents  # cents
        else:
            side = "no"
            cost_per_contract = 100 - market_price_cents  # cents
            kelly = abs(kelly)

        # Cap at max_bet_fraction
        kelly = min(kelly, self.max_bet_fraction)

        bet_amount = self.bankroll * kelly
        # Each Kalshi contract costs the price in cents
        num_contracts = max(1, int(bet_amount * 100 / cost_per_contract))

        # Recalculate actual bet amount
        actual_bet = num_contracts * cost_per_contract / 100.0

        return BetDecision(
            ticker=ticker,
            side=side,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=raw_edge,
            kelly_frac=kelly,
            bet_amount=actual_bet,
            num_contracts=num_contracts,
        )

    def update_bankroll(self, new_bankroll: float) -> None:
        self.bankroll = new_bankroll
