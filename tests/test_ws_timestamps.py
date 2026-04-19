"""Timestamp exposure tests for KalshiWebSocket.

Phase 2 agent safety watchdog needs two health signals from the
WebSocket client:

  last_message_ts  — monotonic time of the most recent decoded frame.
                     Cleared to None on disconnect so a stale pre-
                     disconnect reading cannot mask an outage.

  last_connect_ts  — monotonic time of the most recent successful
                     upgrade. Preserved across disconnects so we can
                     detect reconnect-loop starvation.

These tests drive the logic directly without opening a real socket.
``_handle_message`` is called with synthetic frames; connect/disconnect
state is exercised by setting the private attributes the way the
connect loop would.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tennis_edge.exchange.ws import KalshiWebSocket, TickerUpdate


def _ws(on_ticker=None) -> KalshiWebSocket:
    """Construct a client without any real auth or network wiring.

    The constructor only stores ``auth`` for later use inside
    ``_connect_and_stream``, which we do not exercise here.
    """
    return KalshiWebSocket(auth=None, on_ticker=on_ticker)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_initial_state_is_none():
    ws = _ws()
    assert ws.last_message_ts is None
    assert ws.last_connect_ts is None
    assert ws.seconds_since_last_message() is None
    assert ws.seconds_since_last_connect() is None


@pytest.mark.asyncio
async def test_handle_message_updates_last_message_ts():
    ws = _ws()
    before = time.monotonic()
    await ws._handle_message({"type": "subscribed", "msg": {"channel": "ticker"}})
    after = time.monotonic()

    assert ws.last_message_ts is not None
    assert before <= ws.last_message_ts <= after


@pytest.mark.asyncio
async def test_ticker_frame_fires_callback_and_updates_ts():
    received: list[TickerUpdate] = []

    async def handler(update: TickerUpdate) -> None:
        received.append(update)

    ws = _ws(on_ticker=handler)
    frame = {
        "type": "ticker",
        "msg": {
            "market_ticker": "KXATPMATCH-TEST",
            "yes_bid_dollars": "0.52",
            "yes_ask_dollars": "0.55",
            "price_dollars": "0.53",
            "volume_fp": 1234,
            "ts": 1_700_000_000,
        },
    }
    await ws._handle_message(frame)

    assert len(received) == 1
    assert received[0].ticker == "KXATPMATCH-TEST"
    assert received[0].yes_bid == 52
    assert ws.last_message_ts is not None


@pytest.mark.asyncio
async def test_error_frame_still_counts_as_alive():
    """A server-side error frame is still proof TCP is flowing."""
    ws = _ws()
    await ws._handle_message({"type": "error", "code": 42, "msg": "bad sub"})
    assert ws.last_message_ts is not None


@pytest.mark.asyncio
async def test_seconds_since_last_message_grows():
    ws = _ws()
    await ws._handle_message({"type": "subscribed", "msg": {}})
    t1 = ws.seconds_since_last_message()
    assert t1 is not None
    assert t1 >= 0.0

    await asyncio.sleep(0.02)
    t2 = ws.seconds_since_last_message()
    assert t2 is not None
    assert t2 > t1


@pytest.mark.asyncio
async def test_last_message_cleared_on_simulated_disconnect():
    """The reconnect loop clears last_message_ts. Simulate the clear and
    confirm seconds_since_last_message returns None."""
    ws = _ws()
    await ws._handle_message({"type": "subscribed", "msg": {}})
    assert ws.last_message_ts is not None

    # This is what connect() does on (ConnectionClosed, ConnectionError,
    # OSError) and on generic Exception. Matching that behavior keeps
    # the test coupled to the invariant, not the implementation shape.
    ws.last_message_ts = None

    assert ws.seconds_since_last_message() is None


@pytest.mark.asyncio
async def test_last_connect_ts_survives_disconnect():
    """Reconnect loop clears last_message_ts but leaves last_connect_ts
    so the watchdog can detect reconnect-loop starvation."""
    ws = _ws()
    ws.last_connect_ts = time.monotonic()
    ws.last_message_ts = time.monotonic()

    # Simulate disconnect path.
    ws.last_message_ts = None

    assert ws.last_connect_ts is not None
    assert ws.seconds_since_last_connect() is not None
    assert ws.seconds_since_last_message() is None


@pytest.mark.asyncio
async def test_seconds_since_last_connect_grows_with_time():
    ws = _ws()
    ws.last_connect_ts = time.monotonic()

    t1 = ws.seconds_since_last_connect()
    await asyncio.sleep(0.02)
    t2 = ws.seconds_since_last_connect()

    assert t1 is not None and t2 is not None
    assert t2 > t1


@pytest.mark.asyncio
async def test_each_message_advances_timestamp():
    """Back-to-back frames each move last_message_ts forward (monotonic)."""
    ws = _ws()
    await ws._handle_message({"type": "subscribed", "msg": {}})
    ts1 = ws.last_message_ts

    await asyncio.sleep(0.01)
    await ws._handle_message({"type": "subscribed", "msg": {}})
    ts2 = ws.last_message_ts

    assert ts1 is not None and ts2 is not None
    assert ts2 >= ts1


@pytest.mark.asyncio
async def test_watchdog_shape_healthy_then_stale():
    """End-to-end watchdog logic the safety module will implement.

    Rule: link is unhealthy if
      - seconds_since_last_connect is None, OR
      - seconds_since_last_connect > 60 (reconnect starvation), OR
      - a live match exists AND (seconds_since_last_message is None OR > 60)

    The safety module will own the 'live match exists' check. This test
    just proves the timestamps behave the way that watchdog expects.
    """
    ws = _ws()

    # Pre-connect: no connect timestamp means unhealthy regardless.
    assert ws.seconds_since_last_connect() is None

    # After a connect + one message: everything is fresh.
    ws.last_connect_ts = time.monotonic()
    await ws._handle_message({"type": "subscribed", "msg": {}})
    assert ws.seconds_since_last_connect() is not None
    assert ws.seconds_since_last_message() is not None

    # Backdate last_message_ts by 90s. seconds_since_last_message now >60,
    # which the watchdog will treat as stale if a match is live.
    ws.last_message_ts = time.monotonic() - 90.0
    stale = ws.seconds_since_last_message()
    assert stale is not None and stale > 60.0

    # last_connect_ts stayed fresh: reconnect loop is not starving.
    fresh = ws.seconds_since_last_connect()
    assert fresh is not None and fresh < 60.0
