"""Paper trading engine: simulates order execution locally."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import ExchangeClient
from .schemas import Market, Orderbook, OrderRequest, OrderResponse, Position

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    ticker: str
    side: str  # "yes" or "no"
    count: int
    avg_price: float  # cents
    cost: float  # total cost in dollars


@dataclass
class PaperFill:
    order_id: str
    ticker: str
    side: str
    count: int
    price: float  # cents
    pnl: float = 0.0


class PaperTradingEngine(ExchangeClient):
    """Simulate order execution without hitting Kalshi."""

    def __init__(self, initial_balance: float = 1000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: dict[str, PaperPosition] = {}
        self.fills: list[PaperFill] = []
        self._order_id_counter = 0
        self._markets: dict[str, Market] = {}

    async def __aenter__(self) -> PaperTradingEngine:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    def register_market(self, market: Market) -> None:
        """Register a market for paper trading simulation."""
        self._markets[market.ticker] = market

    async def get_markets(
        self, series_ticker: str | None = None, status: str = "open"
    ) -> list[Market]:
        return list(self._markets.values())

    async def get_market(self, ticker: str) -> Market:
        if ticker in self._markets:
            return self._markets[ticker]
        return Market(ticker=ticker, status="unknown")

    async def get_orderbook(self, ticker: str) -> Orderbook:
        return Orderbook()

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate fill at specified price."""
        self._order_id_counter += 1
        order_id = f"paper-{self._order_id_counter}"

        if order.side == "yes":
            price = order.yes_price or 50
            cost_per_contract = price / 100.0
        else:
            price = 100 - (order.yes_price or 50) if order.yes_price else 50
            cost_per_contract = price / 100.0

        total_cost = cost_per_contract * order.count

        if total_cost > self.balance:
            logger.warning("Insufficient balance: need $%.2f, have $%.2f", total_cost, self.balance)
            return OrderResponse(
                order_id=order_id, ticker=order.ticker, status="rejected",
                side=order.side, action=order.action, count=0,
            )

        self.balance -= total_cost

        # Update position
        key = f"{order.ticker}_{order.side}"
        if key in self.positions:
            pos = self.positions[key]
            pos.count += order.count
            pos.cost += total_cost
            pos.avg_price = (pos.cost / pos.count) * 100
        else:
            self.positions[key] = PaperPosition(
                ticker=order.ticker,
                side=order.side,
                count=order.count,
                avg_price=float(price),
                cost=total_cost,
            )

        fill = PaperFill(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            count=order.count,
            price=float(price),
        )
        self.fills.append(fill)

        logger.info(
            "PAPER TRADE: %s %d %s @ %d cents ($%.2f)",
            order.ticker, order.count, order.side, price, total_cost,
        )

        return OrderResponse(
            order_id=order_id, ticker=order.ticker, status="filled",
            side=order.side, action=order.action, count=order.count,
            yes_price=order.yes_price, remaining_count=0,
        )

    async def cancel_order(self, order_id: str) -> None:
        pass  # Paper orders fill immediately

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                ticker=pos.ticker,
                market_exposure=pos.cost,
                total_traded=pos.count,
            )
            for pos in self.positions.values()
        ]

    async def get_balance(self) -> float:
        return self.balance

    def settle_market(self, ticker: str, result: str) -> float:
        """Settle positions on a market. Returns PnL.

        Args:
            result: "yes" or "no" — the actual outcome.
        """
        total_pnl = 0.0

        for side in ["yes", "no"]:
            key = f"{ticker}_{side}"
            pos = self.positions.pop(key, None)
            if not pos:
                continue

            if side == result:
                # Won: payout is $1 per contract
                payout = pos.count * 1.0
                pnl = payout - pos.cost
            else:
                # Lost: contract is worthless
                pnl = -pos.cost

            total_pnl += pnl
            self.balance += max(0, pos.cost + pnl)

        return total_pnl

    @property
    def portfolio_value(self) -> float:
        """Current balance + estimated position value."""
        pos_value = sum(p.cost for p in self.positions.values())
        return self.balance + pos_value

    @property
    def total_pnl(self) -> float:
        return self.portfolio_value - self.initial_balance
