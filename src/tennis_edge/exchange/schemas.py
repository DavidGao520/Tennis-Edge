"""Pydantic v2 models for Kalshi API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Market(BaseModel):
    ticker: str
    event_ticker: str = ""
    title: str | None = None
    subtitle: str | None = None
    status: str = ""
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    close_time: datetime | None = None
    result: str | None = None

    @property
    def mid_price(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


class OrderRequest(BaseModel):
    ticker: str
    action: Literal["buy", "sell"] = "buy"
    side: Literal["yes", "no"] = "yes"
    type: Literal["limit", "market"] = "limit"
    count: int = 1
    yes_price: int | None = None  # cents, for limit orders
    expiration_ts: int | None = None


class OrderResponse(BaseModel):
    order_id: str = ""
    ticker: str = ""
    status: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int | None = None
    remaining_count: int = 0
    created_time: datetime | None = None


class OrderbookLevel(BaseModel):
    price: int  # cents
    quantity: int


class Orderbook(BaseModel):
    yes: list[OrderbookLevel] = Field(default_factory=list)
    no: list[OrderbookLevel] = Field(default_factory=list)

    @property
    def best_yes_bid(self) -> int | None:
        return max((l.price for l in self.yes), default=None)

    @property
    def best_yes_ask(self) -> int | None:
        # Derive from NO bids: ask_yes = 100 - best_bid_no
        if self.no:
            best_no_bid = max(l.price for l in self.no)
            return 100 - best_no_bid
        return None


class Position(BaseModel):
    ticker: str = ""
    market_exposure: float = 0.0
    total_traded: int = 0
    resting_orders_count: int = 0
    realized_pnl: float = 0.0


class Fill(BaseModel):
    trade_id: str = ""
    ticker: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    created_time: datetime | None = None
