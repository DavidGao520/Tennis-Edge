"""Tests for OrderRequest.client_order_id idempotency plumbing.

Phase 3 v2: AgentLoop sets `client_order_id = decision.decision_id`
so a network-timeout retry never produces a double fill. Verifies
the field round-trips correctly and that KalshiClient sends it on
the wire when set.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tennis_edge.config import KalshiConfig
from tennis_edge.exchange.client import KalshiClient
from tennis_edge.exchange.paper import PaperTradingEngine
from tennis_edge.exchange.schemas import OrderRequest


# ---------------------------------------------------------------------------
# OrderRequest schema
# ---------------------------------------------------------------------------


def test_order_request_default_client_order_id_is_none():
    req = OrderRequest(ticker="T1", count=1, yes_price=50)
    assert req.client_order_id is None


def test_order_request_accepts_client_order_id():
    req = OrderRequest(
        ticker="T1", count=1, yes_price=50,
        client_order_id="dec-abcdef123456",
    )
    assert req.client_order_id == "dec-abcdef123456"


def test_model_dump_excludes_client_order_id_when_none():
    req = OrderRequest(ticker="T1", count=1, yes_price=50)
    blob = req.model_dump(exclude_none=True)
    assert "client_order_id" not in blob


def test_model_dump_includes_client_order_id_when_set():
    req = OrderRequest(
        ticker="T1", count=1, yes_price=50,
        client_order_id="dec-xyz",
    )
    blob = req.model_dump(exclude_none=True)
    assert blob["client_order_id"] == "dec-xyz"


def test_round_trip_via_json():
    req = OrderRequest(
        ticker="T1", count=10, yes_price=42, side="yes",
        client_order_id="dec-roundtrip-abc123",
    )
    raw = req.model_dump_json()
    parsed = OrderRequest.model_validate_json(raw)
    assert parsed.client_order_id == "dec-roundtrip-abc123"
    assert parsed.ticker == "T1"
    assert parsed.count == 10


# ---------------------------------------------------------------------------
# KalshiClient sends client_order_id over the wire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kalshi_client_sends_client_order_id_in_body(tmp_path):
    """Verify the field is in the POST body when set. We use respx to
    mock the HTTP layer so no real Kalshi auth or RSA key is needed."""
    cfg = KalshiConfig(
        base_url="https://api.elections.kalshi.com/trade-api/v2",
        demo_base_url="https://api.elections.kalshi.com/trade-api/v2",
        use_demo=True,  # KalshiConfig.effective_base_url picks demo
        paper_mode=False,
    )

    captured_bodies: list[dict] = []

    def _record(request: httpx.Request) -> httpx.Response:
        # Capture the JSON body so we can assert on it.
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"order": {
                "order_id": "kalshi-001",
                "ticker": "T1",
                "status": "resting",
                "side": "yes",
                "action": "buy",
                "count": 1,
                "yes_price": 50,
                "remaining_count": 0,
            }},
        )

    async with respx.mock(base_url=cfg.effective_base_url) as router:
        router.post("/portfolio/orders").mock(side_effect=_record)
        # auth=None → skip signing path; KalshiClient._request only signs
        # if auth_required AND self.auth is not None.
        async with KalshiClient(cfg, auth=None) as client:
            req = OrderRequest(
                ticker="T1", count=1, yes_price=50, side="yes",
                client_order_id="dec-test-12345",
            )
            await client.place_order(req)

    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert body["client_order_id"] == "dec-test-12345"
    assert body["ticker"] == "T1"


@pytest.mark.asyncio
async def test_kalshi_client_omits_client_order_id_when_none(tmp_path):
    """When client_order_id is None, it MUST NOT appear in the JSON
    body (exclude_none=True). Sending null could trigger validation
    errors on Kalshi's side."""
    cfg = KalshiConfig(use_demo=True)

    captured_bodies: list[dict] = []

    def _record(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"order": {
            "order_id": "kalshi-001", "ticker": "T1", "status": "resting",
            "side": "yes", "action": "buy", "count": 1,
            "yes_price": 50, "remaining_count": 0,
        }})

    async with respx.mock(base_url=cfg.effective_base_url) as router:
        router.post("/portfolio/orders").mock(side_effect=_record)
        async with KalshiClient(cfg, auth=None) as client:
            req = OrderRequest(ticker="T1", count=1, yes_price=50)
            await client.place_order(req)

    assert "client_order_id" not in captured_bodies[0]


# ---------------------------------------------------------------------------
# Paper engine accepts client_order_id silently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_engine_accepts_client_order_id():
    """Paper has no network so idempotency is moot, but the field
    must not error out the place_order call. Validates the AgentLoop
    can send the same OrderRequest shape to either backend."""
    eng = PaperTradingEngine(initial_balance=1000.0)
    req = OrderRequest(
        ticker="T1", count=10, yes_price=50, side="yes",
        client_order_id="dec-paper-test",
    )
    resp = await eng.place_order(req)
    # Paper assigns its own order_id (counter-based); the existence
    # of client_order_id must not crash the simulator.
    assert resp.order_id.startswith("paper-")
    assert resp.ticker == "T1"
