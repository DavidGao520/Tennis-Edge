"""Async Kalshi REST client implementing ExchangeClient ABC."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

import httpx

from ..config import KalshiConfig
from .auth import KalshiAuth
from .base import ExchangeClient
from .schemas import (
    Fill,
    Market,
    Orderbook,
    OrderbookLevel,
    OrderRequest,
    OrderResponse,
    Position,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.15  # ~7 req/s to stay under 10 req/s limit


class KalshiClient(ExchangeClient):
    """Async Kalshi REST API v2 client."""

    def __init__(self, config: KalshiConfig, auth: KalshiAuth | None = None):
        self.config = config
        self.auth = auth
        self._base_url = config.effective_base_url
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> KalshiClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with KalshiClient(...)'")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json: dict | None = None,
        auth_required: bool = False,
    ) -> dict:
        """Make an HTTP request with retry and auth."""
        headers = {}
        if auth_required and self.auth:
            # Sign with full path (base path from URL + endpoint path)
            sign_path = "/trade-api/v2" + path
            headers = self.auth.sign_request(method.upper(), sign_path)

        import asyncio

        for attempt in range(MAX_RETRIES):
            try:
                resp = await self.client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json,
                    headers=headers,
                )

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited, waiting %ds", wait)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        return {}

    # --- Public endpoints ---

    async def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str = "open",
    ) -> list[Market]:
        """Fetch markets, optionally filtered by series/event ticker."""
        params: dict[str, str] = {"status": status, "limit": "100"}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker

        data = await self._request("GET", "/markets", params=params)
        markets_raw = data.get("markets", [])

        return [
            Market(
                ticker=m.get("ticker", ""),
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title"),
                subtitle=m.get("yes_sub_title"),
                status=m.get("status", ""),
                yes_bid=m.get("yes_bid"),
                yes_ask=m.get("yes_ask"),
                no_bid=m.get("no_bid"),
                no_ask=m.get("no_ask"),
                last_price=m.get("last_price"),
                volume=m.get("volume"),
                open_interest=m.get("open_interest"),
                close_time=m.get("close_time"),
                result=m.get("result"),
            )
            for m in markets_raw
        ]

    async def get_market(self, ticker: str) -> Market:
        data = await self._request("GET", f"/markets/{ticker}")
        m = data.get("market", data)
        return Market(
            ticker=m.get("ticker", ticker),
            event_ticker=m.get("event_ticker", ""),
            title=m.get("title"),
            status=m.get("status", ""),
            yes_bid=m.get("yes_bid"),
            yes_ask=m.get("yes_ask"),
            last_price=m.get("last_price"),
            volume=m.get("volume"),
        )

    async def get_orderbook(self, ticker: str) -> Orderbook:
        data = await self._request("GET", f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook", data)
        return Orderbook(
            yes=[OrderbookLevel(price=l[0], quantity=l[1]) for l in ob.get("yes", [])],
            no=[OrderbookLevel(price=l[0], quantity=l[1]) for l in ob.get("no", [])],
        )

    # --- Authenticated endpoints ---

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        data = await self._request(
            "POST",
            "/portfolio/orders",
            json=order.model_dump(exclude_none=True),
            auth_required=True,
        )
        o = data.get("order", data)
        return OrderResponse(
            order_id=o.get("order_id", ""),
            ticker=o.get("ticker", order.ticker),
            status=o.get("status", ""),
            side=o.get("side", order.side),
            action=o.get("action", order.action),
            count=o.get("count", order.count),
            yes_price=o.get("yes_price"),
            remaining_count=o.get("remaining_count", 0),
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._request("DELETE", f"/portfolio/orders/{order_id}", auth_required=True)

    async def get_positions(self) -> list[Position]:
        data = await self._request("GET", "/portfolio/positions", auth_required=True)
        positions = data.get("market_positions", [])
        return [
            Position(
                ticker=p.get("ticker", ""),
                market_exposure=p.get("market_exposure", 0),
                total_traded=p.get("total_traded", 0),
                realized_pnl=p.get("realized_pnl", 0),
            )
            for p in positions
        ]

    async def get_balance(self) -> float:
        data = await self._request("GET", "/portfolio/balance", auth_required=True)
        return data.get("balance", 0) / 100.0  # cents to dollars

    async def get_fills(self) -> list[Fill]:
        data = await self._request("GET", "/portfolio/fills", auth_required=True)
        fills = data.get("fills", [])
        return [
            Fill(
                trade_id=f.get("trade_id", ""),
                ticker=f.get("ticker", ""),
                side=f.get("side", ""),
                action=f.get("action", ""),
                count=f.get("count", 0),
                yes_price=f.get("yes_price", 0),
            )
            for f in fills
        ]
