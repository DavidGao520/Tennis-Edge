"""Tests for agent/llm.py.

Scope: everything except the actual Gemini network call.
GeminiProvider is tested for construction/config-wiring only; its
.analyze() is exercised via FakeLLMProvider which preserves the same
contract (budget reserve → record, EvAnalysis return, exception
hierarchy).

Covers:
  - PricingRates.cost math
  - BudgetTracker: reserve/record/persistence/month rollover/atomic write
  - BudgetExceeded pre-flight reject
  - FakeLLMProvider happy path
  - FakeLLMProvider budget wiring
  - LLMError hierarchy so the 3x kill switch can catch the right types
  - build_prompt substitution
  - GeminiProvider missing-API-key path
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tennis_edge.agent.decisions import EvAnalysis
from tennis_edge.agent.llm import (
    BudgetExceeded,
    BudgetState,
    BudgetTracker,
    FakeLLMProvider,
    GeminiProvider,
    LLMCallError,
    LLMError,
    LLMOutputError,
    PricingRates,
    PromptContext,
    build_prompt,
)


# ---------------------------------------------------------------------------
# PricingRates
# ---------------------------------------------------------------------------


def test_pricing_rates_zero_tokens_zero_cost():
    r = PricingRates(1.0, 2.0, 3.0)
    assert r.cost(0, 0, 0) == 0.0


def test_pricing_rates_math():
    r = PricingRates(input_per_1m_usd=1.0, output_per_1m_usd=2.0, thinking_per_1m_usd=5.0)
    # 1M input → $1, 500k output → $1, 100k thinking → $0.50
    assert r.cost(1_000_000, 500_000, 100_000) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_reserve_blocks_over_cap(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"p": 1.0})
    await bt.record("p", 0.90)
    # Under cap.
    await bt.reserve("p", 0.05)
    # Over cap.
    with pytest.raises(BudgetExceeded):
        await bt.reserve("p", 0.20)


@pytest.mark.asyncio
async def test_budget_record_updates_counters(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"p": 10.0})
    await bt.record("p", 0.12, input_tokens=100, output_tokens=50, thinking_tokens=200)
    snap = bt.snapshot("p")
    assert snap.total_cost_usd == pytest.approx(0.12)
    assert snap.call_count == 1
    assert snap.input_tokens == 100
    assert snap.output_tokens == 50
    assert snap.thinking_tokens == 200


@pytest.mark.asyncio
async def test_budget_remaining_usd(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"p": 10.0})
    await bt.record("p", 3.0)
    assert bt.remaining_usd("p") == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_budget_persists_across_instances(tmp_path):
    path = tmp_path / "b.json"
    bt1 = BudgetTracker(path, {"p": 5.0})
    await bt1.record("p", 1.25, input_tokens=10, output_tokens=20, thinking_tokens=30)

    bt2 = BudgetTracker(path, {"p": 5.0})
    assert bt2.snapshot("p").total_cost_usd == pytest.approx(1.25)
    assert bt2.snapshot("p").input_tokens == 10


@pytest.mark.asyncio
async def test_budget_state_file_is_valid_json(tmp_path):
    path = tmp_path / "b.json"
    bt = BudgetTracker(path, {"p": 5.0})
    await bt.record("p", 0.50, input_tokens=5, output_tokens=6)

    data = json.loads(path.read_text())
    assert data["providers"]["p"]["total_cost_usd"] == pytest.approx(0.50)
    assert "month_key" in data


@pytest.mark.asyncio
async def test_budget_tolerates_corrupt_state_file(tmp_path, caplog):
    path = tmp_path / "b.json"
    path.write_text("not valid json {{{")
    with caplog.at_level("WARNING"):
        bt = BudgetTracker(path, {"p": 5.0})
    # Starts fresh instead of raising.
    assert bt.snapshot("p").total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_budget_month_rollover_resets(tmp_path):
    """When the state file is from a previous month, state resets."""
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "month_key": "2020-01",
        "providers": {"p": {"total_cost_usd": 99.0, "call_count": 1,
                            "input_tokens": 0, "output_tokens": 0,
                            "thinking_tokens": 0}},
    }))
    bt = BudgetTracker(path, {"p": 5.0})
    assert bt.snapshot("p").total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_budget_reserve_does_not_mutate_state(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"p": 10.0})
    await bt.reserve("p", 1.0)
    assert bt.snapshot("p").total_cost_usd == 0.0
    assert bt.snapshot("p").call_count == 0


@pytest.mark.asyncio
async def test_budget_no_cap_means_infinity(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {})  # no cap for any provider
    # Should not raise.
    await bt.reserve("anything", 999.0)


# ---------------------------------------------------------------------------
# FakeLLMProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_provider_returns_valid_analysis(tmp_path):
    p = FakeLLMProvider()
    ctx = PromptContext(
        ticker="T", player_yes="A", player_no="B", tournament="Test",
        surface="Hard", round_name="R32", best_of=3,
        model_pre_match=0.55, market_yes_cents=40,
        yes_form_last10="7-3", no_form_last10="5-5",
        h2h_summary="even", yes_days_since_last_match=3,
        no_days_since_last_match=2,
    )
    result = await p.analyze(ctx)
    assert isinstance(result.analysis, EvAnalysis)
    assert p.call_count == 1


@pytest.mark.asyncio
async def test_fake_provider_wires_into_budget(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"fake": 1.0})
    p = FakeLLMProvider(
        name="fake", cost_usd=0.02, input_tokens=100,
        output_tokens=50, thinking_tokens=25, budget=bt,
    )
    ctx = _min_ctx()
    await p.analyze(ctx)
    snap = bt.snapshot("fake")
    assert snap.total_cost_usd == pytest.approx(0.02)
    assert snap.call_count == 1
    assert snap.input_tokens == 100
    assert snap.thinking_tokens == 25


@pytest.mark.asyncio
async def test_fake_provider_raises_budget_exceeded_when_over_cap(tmp_path):
    bt = BudgetTracker(tmp_path / "b.json", {"fake": 0.01})
    p = FakeLLMProvider(name="fake", cost_usd=0.05, budget=bt)
    with pytest.raises(BudgetExceeded):
        await p.analyze(_min_ctx())
    # No record written if reserve failed.
    assert bt.snapshot("fake").call_count == 0


@pytest.mark.asyncio
async def test_fake_provider_can_raise_custom_error(tmp_path):
    p = FakeLLMProvider(raise_exc=LLMCallError("boom"))
    with pytest.raises(LLMCallError):
        await p.analyze(_min_ctx())


@pytest.mark.asyncio
async def test_fake_provider_error_does_not_record_cost(tmp_path):
    """If the call fails after reserve, record must not be called."""
    bt = BudgetTracker(tmp_path / "b.json", {"fake": 1.0})
    p = FakeLLMProvider(
        name="fake", cost_usd=0.02, budget=bt,
        raise_exc=LLMCallError("network flake"),
    )
    with pytest.raises(LLMCallError):
        await p.analyze(_min_ctx())
    # Reserve succeeded but record did not run — no cost logged.
    assert bt.snapshot("fake").total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_llm_error_hierarchy():
    """Safety watchdog catches LLMError; all three subclasses inherit."""
    assert issubclass(LLMCallError, LLMError)
    assert issubclass(LLMOutputError, LLMError)
    assert issubclass(BudgetExceeded, LLMError)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_build_prompt_substitutes_every_field():
    ctx = PromptContext(
        ticker="KXATP-X", player_yes="Holmgren", player_no="Broady",
        tournament="Madrid Open", surface="Clay", round_name="R32",
        best_of=3, model_pre_match=0.53, market_yes_cents=15,
        yes_form_last10="7-3 last 10", no_form_last10="4-6 last 10",
        h2h_summary="Holmgren 3-1 on clay",
        yes_days_since_last_match=4, no_days_since_last_match=1,
        extra_notes="NOTE: ankle taping observed.",
    )
    text = build_prompt(ctx)
    assert "Holmgren" in text
    assert "Broady" in text
    assert "Madrid Open" in text
    assert "Clay" in text
    assert "0.530" in text or "0.53" in text
    assert "15c" in text
    assert "0.150" in text
    assert "7-3 last 10" in text
    assert "ankle taping" in text
    assert "4d since last match" in text


def test_build_prompt_handles_none_rest_days():
    ctx = PromptContext(
        ticker="T", player_yes="A", player_no="B", tournament="X",
        surface="Hard", round_name="F", best_of=3, model_pre_match=0.5,
        market_yes_cents=50, yes_form_last10="", no_form_last10="",
        h2h_summary="", yes_days_since_last_match=None,
        no_days_since_last_match=None,
    )
    text = build_prompt(ctx)
    assert "unknown" in text


# ---------------------------------------------------------------------------
# GeminiProvider — construction only; network calls not exercised.
# ---------------------------------------------------------------------------


def test_gemini_provider_requires_api_key(tmp_path, monkeypatch):
    # Ensure no ambient env var leaks in.
    monkeypatch.delenv("TENNIS_EDGE_GEMINI_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    bt = BudgetTracker(tmp_path / "b.json", {"gemini-3.1-pro-preview": 50.0})
    with pytest.raises(RuntimeError, match="TENNIS_EDGE_GEMINI_KEY"):
        GeminiProvider(
            model="gemini-3.1-pro-preview",
            rates=PricingRates(1.0, 2.0, 3.0),
            budget=bt,
            api_key=None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _min_ctx() -> PromptContext:
    return PromptContext(
        ticker="T", player_yes="A", player_no="B", tournament="X",
        surface="Hard", round_name="F", best_of=3, model_pre_match=0.5,
        market_yes_cents=50, yes_form_last10="", no_form_last10="",
        h2h_summary="", yes_days_since_last_match=None,
        no_days_since_last_match=None,
    )
