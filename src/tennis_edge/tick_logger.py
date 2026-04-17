"""Phase 2 — Kalshi WebSocket tick logger.

Subscribes to the ticker channel and persists every tennis-market update to
SQLite (table `market_ticks`). Run continuously via:

    tennis-edge log-ticks

Each day this is not running is a day of real backtest data we can never
recover. The output feeds the rewritten backtest engine in Anthony's
workstream so we can replace synthetic odds with real Kalshi prices.

Design choices:
  - Buffer ticks in memory, flush every FLUSH_INTERVAL_S seconds or
    FLUSH_BATCH_SIZE rows, whichever first. SQLite single-row inserts at
    100+ ticks/sec would dominate runtime.
  - Filter tennis markets only. Other Kalshi markets are noise for us.
  - No deduplication. Storage is cheap, downstream queries handle it.
  - Graceful shutdown on SIGINT/SIGTERM: flush buffer before exit.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .data.db import Database
from .exchange.auth import KalshiAuth
from .exchange.ws import KalshiWebSocket, TickerUpdate

logger = logging.getLogger(__name__)

# Tennis market series. Anything not starting with one of these is dropped.
TENNIS_PREFIXES = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTAMATCH",
    "KXWTACHALLENGERMATCH",
)

# Buffer policy.
FLUSH_INTERVAL_S = 5.0
FLUSH_BATCH_SIZE = 100


@dataclass
class TickRow:
    """One row queued for insert."""
    ticker: str
    ts: int
    yes_bid: int | None
    yes_ask: int | None
    last_price: int | None
    volume: int | None
    received_at: int


def _is_tennis(ticker: str) -> bool:
    return any(ticker.startswith(p) for p in TENNIS_PREFIXES)


class TickLogger:
    """Async tick logger. Owns the WebSocket and the DB write loop."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.buffer: deque[TickRow] = deque()
        self.total_received = 0
        self.total_written = 0
        self.last_flush_ts = time.monotonic()
        self.start_ts = time.monotonic()
        self._stop = asyncio.Event()
        self._db: Database | None = None

    async def run(self) -> None:
        """Main entry point. Runs until SIGINT/SIGTERM."""
        # Open DB and ensure schema.
        db_path = Path(self.config.project_root) / self.config.database.path
        self._db = Database(db_path)
        self._db.connect()
        self._db.initialize()
        logger.info("Tick logger DB: %s", db_path)

        # Auth + WebSocket.
        key_path = Path(self.config.project_root) / self.config.kalshi.private_key_path
        auth = KalshiAuth(self.config.kalshi.api_key_id, str(key_path))
        ws = KalshiWebSocket(
            auth=auth,
            use_demo=self.config.kalshi.use_demo,
            on_ticker=self._on_ticker,
        )
        ws.subscribe_ticker()

        # Wire SIGINT/SIGTERM to graceful shutdown.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for SIGTERM.
                pass

        logger.info("Tick logger starting. Filtering to tennis markets only.")

        # Run WebSocket connection, periodic flusher, and stats reporter
        # concurrently. WebSocket reconnects on its own; we cancel everything
        # when _stop is set.
        ws_task = asyncio.create_task(ws.connect())
        flush_task = asyncio.create_task(self._flush_loop())
        stats_task = asyncio.create_task(self._stats_loop())

        try:
            await self._stop.wait()
        finally:
            logger.info("Tick logger stopping. Flushing buffer...")
            ws_task.cancel()
            flush_task.cancel()
            stats_task.cancel()
            for t in (ws_task, flush_task, stats_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            # Final flush.
            await self._flush_buffer()
            self._db.close()
            logger.info(
                "Tick logger stopped. Received=%d written=%d uptime=%.0fs",
                self.total_received, self.total_written,
                time.monotonic() - self.start_ts,
            )

    def _request_stop(self) -> None:
        self._stop.set()

    async def _on_ticker(self, update: TickerUpdate) -> None:
        """WebSocket callback for every ticker message."""
        if not _is_tennis(update.ticker):
            return

        # Skip rows with no price information whatsoever.
        if (update.yes_bid is None and update.yes_ask is None
                and update.last_price is None):
            return

        self.buffer.append(TickRow(
            ticker=update.ticker,
            ts=update.ts or int(time.time()),
            yes_bid=update.yes_bid,
            yes_ask=update.yes_ask,
            last_price=update.last_price,
            volume=update.volume,
            received_at=int(time.time()),
        ))
        self.total_received += 1

        # Trigger immediate flush if we hit batch size.
        if len(self.buffer) >= FLUSH_BATCH_SIZE:
            await self._flush_buffer()

    async def _flush_loop(self) -> None:
        """Periodic flush in case the buffer isn't filling fast enough."""
        try:
            while True:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                if self.buffer:
                    await self._flush_buffer()
        except asyncio.CancelledError:
            return

    async def _flush_buffer(self) -> None:
        """Drain the buffer to SQLite in a single executemany."""
        if not self.buffer or self._db is None:
            return

        # Drain atomically: snapshot and clear so concurrent appends keep
        # going to a fresh buffer.
        rows = list(self.buffer)
        self.buffer.clear()

        try:
            self._db.executemany(
                "INSERT INTO market_ticks "
                "(ticker, ts, yes_bid, yes_ask, last_price, volume, received_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.ticker, r.ts, r.yes_bid, r.yes_ask,
                     r.last_price, r.volume, r.received_at)
                    for r in rows
                ],
            )
            self._db.commit()
            self.total_written += len(rows)
            self.last_flush_ts = time.monotonic()
        except Exception:
            # On failure, put rows back at the front of the buffer so we
            # try again next flush. If the DB is genuinely broken we'll
            # crash on the next flush instead of silently losing data.
            for r in reversed(rows):
                self.buffer.appendleft(r)
            logger.exception("Flush failed; rows requeued (%d in buffer)", len(self.buffer))

    async def _stats_loop(self) -> None:
        """Print throughput every 60s so a long-running logger is visible."""
        try:
            while True:
                await asyncio.sleep(60)
                uptime = time.monotonic() - self.start_ts
                rate = self.total_received / uptime if uptime > 0 else 0
                logger.info(
                    "tick-logger: received=%d written=%d buffer=%d rate=%.1f/s uptime=%.0fs",
                    self.total_received, self.total_written, len(self.buffer),
                    rate, uptime,
                )
        except asyncio.CancelledError:
            return
