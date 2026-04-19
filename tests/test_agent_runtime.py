"""Tests for agent/runtime.py.

Scope: pure parsing + helper logic + market-cache behavior. The full
model-prediction path (FeatureBuilder + LogisticPredictor) is covered
by the existing scanner tests and is not re-tested here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from tennis_edge.agent.runtime import (
    MarketCache,
    _infer_round_best_of,
    _infer_surface,
    _tournament_from_market,
    parse_market_title,
)
from tennis_edge.exchange.schemas import Market


# ---------------------------------------------------------------------------
# parse_market_title
# ---------------------------------------------------------------------------


def test_parse_title_canonical():
    t = "Will Carlos Alcaraz win the Alcaraz vs Virtanen: Round Of 32 match?"
    p = parse_market_title(t)
    assert p is not None
    yes, no, rnd = p
    assert yes == "Carlos Alcaraz"
    assert no == "Virtanen"
    assert "32" in rnd


def test_parse_title_yes_is_second_name():
    """YES can match either side of the 'X vs Y' split."""
    t = "Will Kei Nishikori win the Smith vs Nishikori: Final match?"
    p = parse_market_title(t)
    assert p is not None
    yes, no, _ = p
    assert yes == "Kei Nishikori"
    assert no == "Smith"


def test_parse_title_empty_returns_none():
    assert parse_market_title("") is None


def test_parse_title_unparseable_returns_none():
    assert parse_market_title("random text with no structure") is None


def test_parse_title_no_match_returns_none():
    """Title has the 'Will X win' shape but YES name matches neither side."""
    t = "Will Ghost Player win the A vs B: Final match?"
    assert parse_market_title(t) is None


# ---------------------------------------------------------------------------
# _infer_surface
# ---------------------------------------------------------------------------


def test_infer_surface_clay_from_title():
    m = Market(ticker="T", title="Will X win the Madrid Open match?")
    assert _infer_surface(m) == "Clay"


def test_infer_surface_grass_from_wimbledon():
    m = Market(ticker="T", title="Will X win the Wimbledon Championships match?")
    assert _infer_surface(m) == "Grass"


def test_infer_surface_defaults_to_hard():
    m = Market(ticker="T", title="Will X win the Brisbane International match?")
    assert _infer_surface(m) == "Hard"


def test_infer_surface_handles_missing_title():
    m = Market(ticker="T", title=None)
    assert _infer_surface(m) == "Hard"


# ---------------------------------------------------------------------------
# _infer_round_best_of
# ---------------------------------------------------------------------------


def test_infer_round_r32():
    r, bo = _infer_round_best_of("Round Of 32")
    assert r == "R32"
    assert bo == 3


def test_infer_round_sf():
    r, _ = _infer_round_best_of("Semifinal")
    assert r == "SF"


def test_infer_round_qf():
    r, _ = _infer_round_best_of("Quarterfinal")
    assert r == "QF"


def test_infer_round_final_not_semifinal():
    """'Final' alone is F; 'Semifinal' contains 'Final' but must map to SF."""
    r_final, _ = _infer_round_best_of("Final")
    r_sf, _ = _infer_round_best_of("Semifinal")
    assert r_final == "F"
    assert r_sf == "SF"


def test_infer_round_r16():
    r, _ = _infer_round_best_of("Round Of 16")
    assert r == "R16"


def test_infer_round_default_on_unknown():
    r, _ = _infer_round_best_of("Qualifier")
    assert r == "R32"


# ---------------------------------------------------------------------------
# _tournament_from_market
# ---------------------------------------------------------------------------


def test_tournament_from_title():
    m = Market(
        ticker="T", event_ticker="EVT",
        title="Will Alcaraz win the Alcaraz vs Virtanen: R32 match?",
    )
    assert _tournament_from_market(m) == "Alcaraz vs Virtanen: R32"


def test_tournament_falls_back_to_event_ticker():
    m = Market(ticker="T", event_ticker="EVT", title=None)
    assert _tournament_from_market(m) == "EVT"


# ---------------------------------------------------------------------------
# MarketCache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_cache_caches():
    client = AsyncMock()
    client.get_market.return_value = Market(ticker="T1", title="test")
    cache = MarketCache(client, ttl_s=60.0)

    m1 = await cache.get("T1")
    m2 = await cache.get("T1")
    assert m1 is m2
    assert client.get_market.call_count == 1


@pytest.mark.asyncio
async def test_market_cache_handles_network_error_returns_none(caplog):
    client = AsyncMock()
    client.get_market.side_effect = RuntimeError("network down")
    cache = MarketCache(client, ttl_s=60.0)

    with caplog.at_level("WARNING"):
        result = await cache.get("T1")
    assert result is None
    assert any("failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_market_cache_ttl_expiry_refetches():
    client = AsyncMock()
    client.get_market.return_value = Market(ticker="T1", title="test")
    cache = MarketCache(client, ttl_s=0.01)

    await cache.get("T1")
    await asyncio.sleep(0.02)
    await cache.get("T1")
    assert client.get_market.call_count == 2


@pytest.mark.asyncio
async def test_market_cache_coalesces_concurrent_requests():
    """Two concurrent get()s for the same ticker should share one
    underlying fetch."""
    event = asyncio.Event()

    async def slow_fetch(ticker: str):
        await event.wait()
        return Market(ticker=ticker, title="ok")

    client = AsyncMock()
    client.get_market.side_effect = slow_fetch
    cache = MarketCache(client, ttl_s=60.0)

    t1 = asyncio.create_task(cache.get("T1"))
    t2 = asyncio.create_task(cache.get("T1"))
    await asyncio.sleep(0.01)  # let both start
    event.set()
    m1, m2 = await asyncio.gather(t1, t2)

    assert m1 is m2
    assert client.get_market.call_count == 1
