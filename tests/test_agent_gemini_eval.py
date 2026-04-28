"""Eval suite for the grounded Gemini prompt.

Required gate before merging any change to PROMPT_TEMPLATE_GROUNDED_V1.

These tests make REAL Gemini API calls. They are excluded from the
default `pytest tests/` run. Invoke explicitly:

    pytest -m eval

Cost: ~$0.05-0.10 per test (5 tests at ~$0.015/call grounded).
Total per full eval run: ~$0.10. Run before merging prompt changes
or whenever you suspect Gemini's behavior drifted.

Each case is a hand-built scenario chosen to exercise a specific
behavior we care about. The CRITICAL case is the "settled market"
scenario — that is the v1 failure mode the entire v2 architecture
was built to fix. If that one ever fails, do NOT ship the prompt.

Pass criteria are LENIENT by design:
  - Recommendation must match the expected category (BUY/SKIP)
  - For settled markets, edge_estimate must be near market_prob
  - Confidence cannot be "high" when we deliberately gave thin context
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest

from tennis_edge.agent.decisions import EvAnalysis
from tennis_edge.agent.llm import (
    BudgetTracker,
    GeminiGroundedProvider,
    PricingRates,
    PromptContext,
)


# All eval tests share this marker so pytest -m eval picks them up.
pytestmark = pytest.mark.eval


@dataclass(frozen=True)
class _EvalCase:
    """One eval scenario.

    expected_recs: tuple of acceptable recommendation values. Pass if
                   actual is in this set.
    max_confidence: highest confidence level allowed. None = any OK.
    edge_close_to_market: if True, edge_estimate must be within 0.20
                          of market_prob (used for settled markets).
    """

    name: str
    ctx: PromptContext
    expected_recs: tuple[str, ...]
    max_confidence: str | None = None
    edge_close_to_market: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Cases. Picked to cover the failure modes the prompt must handle.
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> PromptContext:
    base = dict(
        ticker="EVAL-TEST",
        player_yes="?",
        player_no="?",
        tournament="?",
        surface="Hard",
        round_name="R32",
        best_of=3,
        model_pre_match=0.5,
        market_yes_cents=50,
        yes_form_last10="?",
        no_form_last10="?",
        h2h_summary="?",
        yes_days_since_last_match=None,
        no_days_since_last_match=None,
    )
    base.update(overrides)
    return PromptContext(**base)


CASES: list[_EvalCase] = [
    # ------------------------------------------------------------------
    # CASE 1: settled-market regression. THIS IS THE v1 FAILURE MODE.
    # If Gemini recommends BUY here, the entire v2 thesis is wrong.
    # ------------------------------------------------------------------
    _EvalCase(
        name="settled_market_no_fade",
        ctx=_ctx(
            ticker="EVAL-SETTLED",
            player_yes="Jannik Sinner",
            player_no="Cameron Norrie",
            tournament="Mutua Madrid Open 2026",
            surface="Clay",
            round_name="R16",
            best_of=3,
            model_pre_match=0.78,
            market_yes_cents=98,  # priced as if YES is essentially won
            yes_form_last10="9-1",
            no_form_last10="5-5",
            h2h_summary="Sinner leads 4-0",
        ),
        expected_recs=("SKIP", "BUY_YES"),
        edge_close_to_market=True,
        notes=(
            "Market at 98c implies the match is decided or over. "
            "Gemini must NOT recommend BUY_NO at any meaningful size, "
            "and any BUY_YES recommendation must price edge near "
            "market (no fade against settled outcome)."
        ),
    ),

    # ------------------------------------------------------------------
    # CASE 2: thin context, Gemini search should find nothing useful.
    # Expected: low or medium confidence, lean toward SKIP.
    # ------------------------------------------------------------------
    _EvalCase(
        name="thin_context_obscure_match",
        ctx=_ctx(
            ticker="EVAL-OBSCURE",
            player_yes="John Q. Player",  # fictional name
            player_no="Anonymous Opponent",
            tournament="Hypothetical Cup",
            model_pre_match=0.5,
            market_yes_cents=50,
        ),
        expected_recs=("SKIP",),
        max_confidence="medium",
        notes=(
            "Fictional match — search will return nothing. The prompt "
            "explicitly says 'lean toward SKIP / lower confidence "
            "rather than fabricating context'. Anything else is the "
            "model hallucinating data."
        ),
    ),

    # ------------------------------------------------------------------
    # CASE 3: pre-match clear favorite, market underprices.
    # Realistic scenario the agent should handle.
    # ------------------------------------------------------------------
    _EvalCase(
        name="prematch_strong_favorite_underpriced",
        ctx=_ctx(
            ticker="EVAL-FAV",
            player_yes="Top-ranked player",
            player_no="Lower-ranked qualifier",
            tournament="ATP Tour event 2026",
            surface="Hard",
            round_name="R32",
            model_pre_match=0.85,
            market_yes_cents=60,  # implied 60% << 85% model
            yes_form_last10="8-2",
            no_form_last10="3-7",
            h2h_summary="favorite leads 3-0",
            yes_days_since_last_match=3,
            no_days_since_last_match=2,
        ),
        # Either is acceptable — Gemini may search and find context
        # that justifies trusting the model anchor (BUY_YES) or that
        # explains the market discount (SKIP). Both are reasoned
        # responses; what we forbid is BUY_NO.
        expected_recs=("BUY_YES", "SKIP"),
        notes=(
            "Acceptable to BUY_YES (trust the anchor) or SKIP (defer "
            "to live news). Forbidden: BUY_NO — there is no story "
            "consistent with the static features that says fade the "
            "favorite."
        ),
    ),

    # ------------------------------------------------------------------
    # CASE 4: extreme-low price (<10c) — the v1 'in-progress losing'
    # signature. Gemini must NOT recommend BUY_YES.
    # ------------------------------------------------------------------
    _EvalCase(
        name="extreme_low_price_likely_in_progress",
        ctx=_ctx(
            ticker="EVAL-LOW",
            player_yes="Underdog",
            player_no="Favorite",
            tournament="ATP event 2026",
            model_pre_match=0.30,
            market_yes_cents=3,  # match basically over against YES
            h2h_summary="Favorite leads 5-1",
        ),
        expected_recs=("SKIP", "BUY_NO"),
        edge_close_to_market=True,
        notes=(
            "Market at 3c means YES is essentially eliminated. The "
            "v1 bug recommended BUY_YES on similar setups (model "
            "said 30% so 'edge' looked huge). Hard fail if Gemini "
            "recommends BUY_YES."
        ),
    ),

    # ------------------------------------------------------------------
    # CASE 5: balanced match, no clear edge. Healthy SKIP behavior.
    # ------------------------------------------------------------------
    _EvalCase(
        name="balanced_no_edge",
        ctx=_ctx(
            ticker="EVAL-EVEN",
            player_yes="Player A",
            player_no="Player B",
            tournament="ATP tour 2026",
            model_pre_match=0.50,
            market_yes_cents=50,
            yes_form_last10="5-5",
            no_form_last10="5-5",
            h2h_summary="even, 3-3",
        ),
        expected_recs=("SKIP",),
        notes=(
            "Both sides priced at coin flip, no static edge. Healthy "
            "behavior is to recognize there's nothing to trade."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Fixture: shared provider across the eval run
# ---------------------------------------------------------------------------


def _load_key() -> str | None:
    key = os.environ.get("TENNIS_EDGE_GEMINI_KEY")
    if key:
        return key
    # .env fallback so devs don't have to export
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env",
    )
    if not os.path.exists(env_path):
        return None
    for line in open(env_path):
        if line.startswith("TENNIS_EDGE_GEMINI_KEY="):
            key = line.split("=", 1)[1].strip()
            os.environ["TENNIS_EDGE_GEMINI_KEY"] = key
            return key
    return None


@pytest.fixture(scope="module")
def grounded_provider(tmp_path_factory):
    key = _load_key()
    if not key:
        pytest.skip("TENNIS_EDGE_GEMINI_KEY not set; cannot run eval suite.")

    rates = PricingRates(2.50, 10.00, 10.00)
    budget_path = tmp_path_factory.mktemp("eval-budget") / "budget.json"
    bt = BudgetTracker(
        budget_path,
        # Generous cap for the eval run; 5 cases at ~$0.02 each.
        {"gemini-3.1-pro-preview-grounded": 1.00},
    )
    return GeminiGroundedProvider(
        model="gemini-3.1-pro-preview",
        rates=rates,
        budget=bt,
        api_key=key,
    )


# ---------------------------------------------------------------------------
# Eval driver — one parametrized test per case so failures localize.
# ---------------------------------------------------------------------------


CONF_RANK = {"low": 0, "medium": 1, "high": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
async def test_grounded_prompt_eval(case: _EvalCase, grounded_provider):
    """Drive the grounded provider against one hand-built scenario.

    Asserts only the LENIENT criteria documented per case. The point
    is regression detection across prompt changes, not a tight
    benchmark. A prompt edit that flips any of these is a structural
    change that needs explicit review.
    """
    result = await grounded_provider.analyze(case.ctx)
    a: EvAnalysis = result.analysis

    # Print a short summary so eval failures show what Gemini said.
    print(
        f"\n[{case.name}]\n"
        f"  rec={a.recommendation} conf={a.confidence} "
        f"edge_est={a.edge_estimate:.3f} (market={case.ctx.market_yes_cents}c)\n"
        f"  reasoning: {a.reasoning[:240]}"
    )

    # Recommendation must be in the allowed set.
    assert a.recommendation in case.expected_recs, (
        f"{case.name}: rec={a.recommendation} not in {case.expected_recs}\n"
        f"  reasoning: {a.reasoning[:200]}"
    )

    # Confidence ceiling.
    if case.max_confidence is not None:
        assert CONF_RANK[a.confidence] <= CONF_RANK[case.max_confidence], (
            f"{case.name}: conf={a.confidence} > max={case.max_confidence}"
        )

    # Edge near market price (settled or near-settled markets).
    if case.edge_close_to_market:
        market_prob = case.ctx.market_yes_cents / 100.0
        delta = abs(a.edge_estimate - market_prob)
        assert delta < 0.20, (
            f"{case.name}: edge_estimate={a.edge_estimate:.3f} "
            f"too far from market_prob={market_prob:.3f} (Δ={delta:.3f}). "
            f"Settled markets must not be faded."
        )
