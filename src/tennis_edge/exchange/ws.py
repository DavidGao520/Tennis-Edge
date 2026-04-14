"""Kalshi WebSocket client for real-time market data streaming.

Channels:
  - ticker: bid/ask for ALL markets, pushed on every change
  - orderbook_delta: orderbook snapshots + deltas for subscribed markets
  - trade: trade executions for subscribed markets
  - fill: your own fill notifications (private)
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import certifi
import websockets

from .auth import KalshiAuth

logger = logging.getLogger(__name__)

WS_URL_PROD = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"
WS_SIGN_PATH = "/trade-api/ws/v2"

HEARTBEAT_INTERVAL = 30  # seconds
RECONNECT_DELAY = 5


@dataclass
class TickerUpdate:
    """Real-time ticker data for a single market."""
    ticker: str
    yes_bid: int | None = None   # cents
    yes_ask: int | None = None   # cents
    no_bid: int | None = None
    no_ask: int | None = None
    last_price: int | None = None
    volume: int | None = None
    ts: int = 0


@dataclass
class OrderbookSnapshot:
    """Full orderbook snapshot for a market."""
    ticker: str
    yes_bids: list[list[int]] = field(default_factory=list)  # [[price, qty], ...]
    no_bids: list[list[int]] = field(default_factory=list)


@dataclass
class TradeUpdate:
    """A trade execution on a market."""
    ticker: str
    side: str
    yes_price: int = 0
    no_price: int = 0
    count: int = 0
    ts: int = 0


@dataclass
class FillUpdate:
    """Your own fill notification."""
    ticker: str
    order_id: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    ts: int = 0


# Callback types
TickerCallback = Callable[[TickerUpdate], Coroutine[Any, Any, None]]
OrderbookCallback = Callable[[OrderbookSnapshot], Coroutine[Any, Any, None]]
TradeCallback = Callable[[TradeUpdate], Coroutine[Any, Any, None]]
FillCallback = Callable[[FillUpdate], Coroutine[Any, Any, None]]


class KalshiWebSocket:
    """Async WebSocket client for Kalshi real-time data.

    Usage:
        ws = KalshiWebSocket(auth, on_ticker=my_handler)
        ws.subscribe_ticker()
        ws.subscribe_orderbook(["KXATPMATCH-26APR14-ALC"])
        await ws.connect()  # blocks, reconnects on failure
    """

    def __init__(
        self,
        auth: KalshiAuth,
        use_demo: bool = False,
        on_ticker: TickerCallback | None = None,
        on_orderbook: OrderbookCallback | None = None,
        on_trade: TradeCallback | None = None,
        on_fill: FillCallback | None = None,
        on_connect: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ):
        self.auth = auth
        self.ws_url = WS_URL_DEMO if use_demo else WS_URL_PROD
        self.on_ticker = on_ticker
        self.on_orderbook = on_orderbook
        self.on_trade = on_trade
        self.on_fill = on_fill
        self.on_connect = on_connect

        self._ws: Any = None
        self._cmd_id = 0
        self._subscriptions: list[dict] = []
        self._running = False

    def subscribe_ticker(self) -> None:
        """Subscribe to ticker updates for ALL markets."""
        self._subscriptions.append({"channels": ["ticker"]})

    def subscribe_orderbook(self, market_tickers: list[str]) -> None:
        """Subscribe to orderbook deltas for specific markets."""
        self._subscriptions.append({
            "channels": ["orderbook_delta"],
            "market_tickers": market_tickers,
        })

    def subscribe_trades(self, market_tickers: list[str]) -> None:
        """Subscribe to trade feed for specific markets."""
        self._subscriptions.append({
            "channels": ["trade"],
            "market_tickers": market_tickers,
        })

    def subscribe_fills(self) -> None:
        """Subscribe to your own fill notifications."""
        self._subscriptions.append({"channels": ["fill"]})

    async def connect(self) -> None:
        """Connect and stream. Reconnects on failure."""
        self._running = True

        while self._running:
            try:
                await self._connect_and_stream()
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning("WebSocket disconnected: %s. Reconnecting in %ds...", e, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as e:
                logger.error("WebSocket error: %s. Reconnecting in %ds...", e, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_stream(self) -> None:
        """Single connection attempt."""
        # Build auth headers
        headers = self.auth.sign_request("GET", WS_SIGN_PATH)

        logger.info("Connecting to %s", self.ws_url)

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        async with websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=HEARTBEAT_INTERVAL,
            ping_timeout=10,
            ssl=ssl_ctx,
        ) as ws:
            self._ws = ws
            logger.info("WebSocket connected")

            if self.on_connect:
                await self.on_connect()

            # Send subscriptions
            for sub in self._subscriptions:
                await self._send_subscribe(sub)

            # Message loop
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON: %s", raw[:200])

    async def _send_subscribe(self, params: dict) -> None:
        """Send a subscription command."""
        self._cmd_id += 1
        cmd = {
            "id": self._cmd_id,
            "cmd": "subscribe",
            "params": params,
        }
        await self._ws.send(json.dumps(cmd))
        logger.info("Subscribed: %s", params.get("channels"))

    async def subscribe_orderbook_live(self, market_tickers: list[str]) -> None:
        """Subscribe to orderbook while already connected."""
        if self._ws:
            params = {"channels": ["orderbook_delta"], "market_tickers": market_tickers}
            await self._send_subscribe(params)
            self._subscriptions.append(params)

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming messages to callbacks."""
        msg_type = msg.get("type", "")

        if msg_type == "ticker":
            if self.on_ticker:
                data = msg.get("msg", {})

                # Parse dollar strings to cents (int)
                def to_cents(val) -> int | None:
                    if val is None:
                        return None
                    try:
                        return int(round(float(val) * 100))
                    except (ValueError, TypeError):
                        return None

                update = TickerUpdate(
                    ticker=data.get("market_ticker", ""),
                    yes_bid=to_cents(data.get("yes_bid_dollars")),
                    yes_ask=to_cents(data.get("yes_ask_dollars")),
                    last_price=to_cents(data.get("price_dollars")),
                    volume=int(float(data.get("volume_fp", 0) or 0)),
                    ts=data.get("ts", 0),
                )
                await self.on_ticker(update)

        elif msg_type == "orderbook_snapshot":
            if self.on_orderbook:
                data = msg.get("msg", {})
                snapshot = OrderbookSnapshot(
                    ticker=data.get("market_ticker", ""),
                    yes_bids=data.get("yes", []),
                    no_bids=data.get("no", []),
                )
                await self.on_orderbook(snapshot)

        elif msg_type == "orderbook_delta":
            if self.on_orderbook:
                data = msg.get("msg", {})
                snapshot = OrderbookSnapshot(
                    ticker=data.get("market_ticker", ""),
                    yes_bids=data.get("yes", []),
                    no_bids=data.get("no", []),
                )
                await self.on_orderbook(snapshot)

        elif msg_type == "trade":
            if self.on_trade:
                data = msg.get("msg", {})
                trade = TradeUpdate(
                    ticker=data.get("market_ticker", ""),
                    side=data.get("side", ""),
                    yes_price=data.get("yes_price", 0),
                    no_price=data.get("no_price", 0),
                    count=data.get("count", 0),
                    ts=data.get("ts", 0),
                )
                await self.on_trade(trade)

        elif msg_type == "fill":
            if self.on_fill:
                data = msg.get("msg", {})
                fill = FillUpdate(
                    ticker=data.get("market_ticker", ""),
                    order_id=data.get("order_id", ""),
                    side=data.get("side", ""),
                    action=data.get("action", ""),
                    count=data.get("count", 0),
                    yes_price=data.get("yes_price", 0),
                    ts=data.get("ts", 0),
                )
                await self.on_fill(fill)

        elif msg_type == "subscribed":
            logger.info("Subscription confirmed: %s", msg.get("msg", {}).get("channel"))

        elif msg_type == "error":
            logger.error("WebSocket error: code=%s msg=%s",
                        msg.get("code"), msg.get("msg"))
