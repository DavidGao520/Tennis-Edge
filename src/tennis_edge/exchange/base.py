"""Abstract exchange client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import Market, OrderRequest, OrderResponse, Orderbook, Position


class ExchangeClient(ABC):
    """Abstract base class for exchange clients (live, paper, mock)."""

    @abstractmethod
    async def get_markets(
        self, series_ticker: str | None = None, status: str = "open"
    ) -> list[Market]: ...

    @abstractmethod
    async def get_market(self, ticker: str) -> Market: ...

    @abstractmethod
    async def get_orderbook(self, ticker: str) -> Orderbook: ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def get_balance(self) -> float: ...

    async def __aenter__(self) -> ExchangeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass
