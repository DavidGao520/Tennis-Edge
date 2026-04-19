"""Phase 2 Agent LLM provider + budget tracker.

                          prompt_builder
                                │
                                ▼
                      ┌──────────────────────┐
                      │   LLMProvider ABC    │
                      │                      │
                      │   .analyze(ctx) ────►│──► EvAnalysis
                      └──────────┬───────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
   │ FakeLLMProvider │  │ GeminiProvider  │  │ (future: Claude)│
   │   (tests only)  │  │ google-genai    │  │                 │
   └─────────────────┘  └────────┬────────┘  └─────────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │ BudgetTracker│── data/agent_budget.json
                         │  per-month   │    (atomic write + lock)
                         └──────────────┘

Key design decisions (Phase 2 eng review):

1. BudgetTracker enforces a hard monthly cost cap. Pre-flight cost
   estimate (input tokens counted via API + worst-case output) is
   checked BEFORE sending the request. Over cap = BudgetExceeded, no
   call made, safety watchdog trips kill switch. Running over cap is
   worse than missing a signal.

2. State persists to data/agent_budget.json with atomic write
   (os.replace). Survives daemon restart. Month rollover detected
   from a `month_key` stamp; a new month resets counters cleanly.

3. Output validation: Gemini response_schema constrains the reply to
   the EvAnalysis shape. Any JSON parse or pydantic validation failure
   is raised as LLMOutputError so the 3x-consecutive kill switch in
   agent/safety.py can count it as an LLM failure.

4. Real SDK import is lazy so tests can use FakeLLMProvider without
   google-genai installed in every environment.

5. Pricing rates are configured, not hardcoded. The rate card for
   preview models changes; we default to conservative numbers but the
   real production config reads rates from config/default.yaml.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .decisions import EvAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for LLM failures that should increment the 3x counter."""


class LLMOutputError(LLMError):
    """LLM returned something we could not parse/validate into EvAnalysis."""


class LLMCallError(LLMError):
    """LLM call itself failed (network, auth, server error)."""


class BudgetExceeded(LLMError):
    """Pre-flight estimate says this call would push us over the monthly cap."""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptContext:
    """All inputs the text-only Phase 3A prompt needs.

    Kept flat and string-friendly so the template is easy to version.
    """

    ticker: str
    player_yes: str
    player_no: str
    tournament: str
    surface: str
    round_name: str
    best_of: int
    model_pre_match: float            # P(YES wins)
    market_yes_cents: int             # Kalshi YES price
    yes_form_last10: str              # e.g. "7-3 last 10"
    no_form_last10: str
    h2h_summary: str                  # e.g. "YES leads 5-2 overall, 2-1 on clay"
    yes_days_since_last_match: int | None
    no_days_since_last_match: int | None
    extra_notes: str = ""             # anything the caller wants to append


PROMPT_TEMPLATE_V1 = """You are a quantitative analyst for a Kalshi tennis prediction market.

Market: {ticker}
Outcome: YES = "{player_yes} wins", NO = "{player_no} wins"
Context: {tournament}, {round_name}, {surface}, best-of-{best_of}

Pre-match model anchor: P({player_yes} wins) = {model_pre_match:.3f}
Kalshi YES market price: {market_yes_cents}c (implied P = {market_prob:.3f})

Recent form:
- {player_yes}: {yes_form_last10}
- {player_no}: {no_form_last10}

Head-to-head: {h2h_summary}

Fatigue / rest:
- {player_yes}: {yes_rest}
- {player_no}: {no_rest}

{extra_notes}

Task: estimate the true probability that YES wins, then produce a
recommendation. Respond strictly in the JSON shape requested by the
schema. edge_estimate is your probability, not the gap vs market.
"""


def build_prompt(ctx: PromptContext, template: str = PROMPT_TEMPLATE_V1) -> str:
    """Render a PromptContext into the plain-text prompt body."""
    return template.format(
        ticker=ctx.ticker,
        player_yes=ctx.player_yes,
        player_no=ctx.player_no,
        tournament=ctx.tournament,
        round_name=ctx.round_name,
        surface=ctx.surface,
        best_of=ctx.best_of,
        model_pre_match=ctx.model_pre_match,
        market_yes_cents=ctx.market_yes_cents,
        market_prob=ctx.market_yes_cents / 100.0,
        yes_form_last10=ctx.yes_form_last10,
        no_form_last10=ctx.no_form_last10,
        h2h_summary=ctx.h2h_summary,
        yes_rest=(
            f"{ctx.yes_days_since_last_match}d since last match"
            if ctx.yes_days_since_last_match is not None else "unknown"
        ),
        no_rest=(
            f"{ctx.no_days_since_last_match}d since last match"
            if ctx.no_days_since_last_match is not None else "unknown"
        ),
        extra_notes=ctx.extra_notes or "",
    )


# JSON schema for Gemini response_schema (structured output). Mirrors
# EvAnalysis but in the subset of JSON schema that the SDK accepts.
GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edge_estimate": {"type": "number"},
        "recommendation": {
            "type": "string",
            "enum": ["BUY_YES", "BUY_NO", "SKIP"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "reasoning": {"type": "string"},
        "key_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["edge_estimate", "recommendation", "confidence", "reasoning"],
}


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricingRates:
    """Dollar cost per 1M tokens. Read from config.

    Gemini 3.x bills thinking tokens separately from output tokens. If
    the provider does not separate the two, set thinking_per_1m_usd
    equal to output_per_1m_usd.
    """

    input_per_1m_usd: float
    output_per_1m_usd: float
    thinking_per_1m_usd: float

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
    ) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_1m_usd
            + output_tokens / 1_000_000 * self.output_per_1m_usd
            + thinking_tokens / 1_000_000 * self.thinking_per_1m_usd
        )


@dataclass
class BudgetState:
    """One month's running totals for one provider."""

    total_cost_usd: float = 0.0
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


def _current_month_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


class BudgetTracker:
    """Per-provider, per-month hard cost cap.

    The call path is:

      est_cost = rates.cost(est_input, max_output, max_thinking)
      tracker.reserve(provider, est_cost)   → BudgetExceeded if over cap
      actual = llm.call(...)
      tracker.record(provider, actual.cost)

    We use reserve-then-record so a call that never reaches record
    (network error during call) does not silently consume budget.
    reserve is optimistic — the actual call cost might differ from the
    estimate. If estimate overshoots, record clamps to real. If
    estimate undershoots, the next reserve will see higher totals.

    Persists to data/agent_budget.json on every mutation with atomic
    write (tempfile + os.replace). Under-the-GIL asyncio.Lock guards
    against concurrent reserves from the queue worker.
    """

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        monthly_cap_usd: dict[str, float],
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.monthly_cap_usd = dict(monthly_cap_usd)
        self._lock = asyncio.Lock()
        self._month_key = _current_month_key()
        self._state: dict[str, BudgetState] = {}
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            blob = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("budget state unreadable, starting fresh: %s", e)
            return

        month_key = blob.get("month_key")
        if month_key != self._month_key:
            # Rolled over. Start fresh — last month's totals are archived
            # only in the previous file content (which we are about to
            # overwrite). For Phase 3A audit needs, that is fine.
            logger.info("budget: month rollover %s → %s", month_key, self._month_key)
            return

        for provider, s in blob.get("providers", {}).items():
            self._state[provider] = BudgetState(**s)

    def _persist_unlocked(self) -> None:
        blob = {
            "month_key": self._month_key,
            "providers": {k: v.__dict__ for k, v in self._state.items()},
        }
        # Atomic write: temp file in same dir, then os.replace.
        dir_ = self.state_path.parent
        fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".budget.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        except Exception:
            # Clean up the tempfile if the replace failed.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- API ----

    async def reserve(self, provider: str, estimated_cost_usd: float) -> None:
        """Raise BudgetExceeded if this call would push us over the cap.

        Caller must have already computed estimated_cost using
        PricingRates.cost(input_tokens, max_output, max_thinking).
        Conservative estimation is the point — real cost is expected
        to be lower.
        """
        async with self._lock:
            self._maybe_roll_month_unlocked()
            cap = self.monthly_cap_usd.get(provider, float("inf"))
            state = self._state.setdefault(provider, BudgetState())
            projected = state.total_cost_usd + estimated_cost_usd
            if projected > cap:
                raise BudgetExceeded(
                    f"{provider}: projected ${projected:.4f} > cap ${cap:.2f} "
                    f"(current ${state.total_cost_usd:.4f} + est ${estimated_cost_usd:.4f})"
                )

    async def record(
        self,
        provider: str,
        cost_usd: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        """Record the actual cost and token counts from a completed call."""
        async with self._lock:
            self._maybe_roll_month_unlocked()
            state = self._state.setdefault(provider, BudgetState())
            state.total_cost_usd += cost_usd
            state.call_count += 1
            state.input_tokens += input_tokens
            state.output_tokens += output_tokens
            state.thinking_tokens += thinking_tokens
            self._persist_unlocked()

    def snapshot(self, provider: str) -> BudgetState:
        """Read-only point-in-time view. No lock needed for caller reads."""
        return self._state.get(provider, BudgetState())

    def remaining_usd(self, provider: str) -> float:
        cap = self.monthly_cap_usd.get(provider, float("inf"))
        return max(0.0, cap - self.snapshot(provider).total_cost_usd)

    def _maybe_roll_month_unlocked(self) -> None:
        current = _current_month_key()
        if current != self._month_key:
            logger.info("budget: month rolled %s → %s", self._month_key, current)
            self._month_key = current
            self._state = {}
            # Wipe the file too. Losing last month's in-memory total is
            # fine; it is already on disk in the old file, which we
            # overwrite here. For Phase 3A audit needs this is acceptable.
            self._persist_unlocked()


# ---------------------------------------------------------------------------
# Provider ABC + impls
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """What a provider returns. analysis is already validated."""

    analysis: EvAnalysis
    raw_output: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float
    provider: str  # e.g. "gemini-3.1-pro-preview"


class LLMProvider(ABC):
    """Abstract provider. Implementations handle:
      1. Pre-flight input-token count
      2. Budget reserve
      3. Actual call
      4. Output parse/validate
      5. Budget record
    """

    name: str

    @abstractmethod
    async def analyze(self, ctx: PromptContext) -> LLMResult: ...


class FakeLLMProvider(LLMProvider):
    """Deterministic fake used by tests.

    Returns a fixed EvAnalysis (or raises a configured error). Tracks
    how many times analyze() was called so tests can assert cooldowns
    and queue behavior.
    """

    def __init__(
        self,
        name: str = "fake",
        analysis: EvAnalysis | None = None,
        raise_exc: Exception | None = None,
        cost_usd: float = 0.0001,
        input_tokens: int = 100,
        output_tokens: int = 50,
        thinking_tokens: int = 0,
        budget: BudgetTracker | None = None,
    ):
        self.name = name
        self._analysis = analysis or EvAnalysis(
            edge_estimate=0.55,
            recommendation="BUY_YES",
            confidence="medium",
            reasoning="fake reasoning for test",
            key_factors=["fake factor 1"],
        )
        self._raise = raise_exc
        self._cost = cost_usd
        self._in = input_tokens
        self._out = output_tokens
        self._thk = thinking_tokens
        self._budget = budget
        self.call_count = 0

    async def analyze(self, ctx: PromptContext) -> LLMResult:
        self.call_count += 1
        if self._budget is not None:
            await self._budget.reserve(self.name, self._cost)
        if self._raise is not None:
            raise self._raise
        if self._budget is not None:
            await self._budget.record(
                self.name, self._cost,
                input_tokens=self._in,
                output_tokens=self._out,
                thinking_tokens=self._thk,
            )
        return LLMResult(
            analysis=self._analysis,
            raw_output=self._analysis.model_dump_json(),
            input_tokens=self._in,
            output_tokens=self._out,
            thinking_tokens=self._thk,
            cost_usd=self._cost,
            provider=self.name,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini 3.x provider via google-genai SDK.

    The SDK import is deferred to __init__ so that modules importing
    this file do not require google-genai to be installed. Tests that
    use FakeLLMProvider never trigger the import.

    Configuration:
      model           — e.g. "gemini-3.1-pro-preview"
      api_key         — from env var (default TENNIS_EDGE_GEMINI_KEY)
      rates           — PricingRates (read from config/default.yaml)
      budget          — BudgetTracker instance
      max_output      — worst-case output tokens for pre-flight estimate
      max_thinking    — worst-case thinking tokens for pre-flight estimate
    """

    def __init__(
        self,
        model: str,
        rates: PricingRates,
        budget: BudgetTracker,
        api_key: str | None = None,
        # Gemini 3.x thinking models burn >1K reasoning tokens on short
        # prompts. Real smoke test used ~700 thinking + ~230 output for
        # one decision; 8192/8192 gives 10x safety margin before truncation.
        # Truncated JSON surfaces as LLMOutputError which kills the run.
        max_output_tokens: int = 8192,
        max_thinking_tokens: int = 8192,
        request_timeout_s: float = 60.0,
    ):
        # Lazy SDK import — raises ImportError here instead of at module
        # load. Tests that never instantiate GeminiProvider do not need
        # the SDK installed.
        from google import genai  # noqa: F401 — intentional
        from google.genai import types as _types  # noqa: F401

        self._genai = genai
        self._types = _types

        self.name = model
        self.model = model
        self.rates = rates
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.max_thinking_tokens = max_thinking_tokens
        self.request_timeout_s = request_timeout_s

        key = api_key or os.environ.get("TENNIS_EDGE_GEMINI_KEY")
        if not key:
            raise RuntimeError(
                "TENNIS_EDGE_GEMINI_KEY not set and no api_key passed"
            )
        # google-genai also honors GEMINI_API_KEY; we set both to be safe.
        os.environ.setdefault("GEMINI_API_KEY", key)
        self._client = genai.Client(api_key=key)

    async def analyze(self, ctx: PromptContext) -> LLMResult:
        prompt = build_prompt(ctx)

        # Pre-flight token count for the input. The SDK's count_tokens
        # is synchronous; wrap in to_thread so the event loop stays free.
        try:
            tok_resp = await asyncio.to_thread(
                self._client.models.count_tokens,
                model=self.model,
                contents=prompt,
            )
            est_input = int(tok_resp.total_tokens or 0)
        except Exception as e:
            raise LLMCallError(f"count_tokens failed: {e}") from e

        est_cost = self.rates.cost(
            est_input, self.max_output_tokens, self.max_thinking_tokens
        )
        await self.budget.reserve(self.name, est_cost)

        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
            max_output_tokens=self.max_output_tokens,
        )

        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.request_timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise LLMCallError(f"generate_content timeout after {self.request_timeout_s}s") from e
        except Exception as e:
            raise LLMCallError(f"generate_content failed: {e}") from e

        raw = resp.text or ""
        try:
            analysis = EvAnalysis.model_validate_json(raw)
        except (ValidationError, ValueError) as e:
            raise LLMOutputError(f"bad LLM JSON: {e}: {raw[:300]}") from e

        usage = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        thk_tok = int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        actual_cost = self.rates.cost(in_tok, out_tok, thk_tok)

        await self.budget.record(
            self.name, actual_cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=thk_tok,
        )

        return LLMResult(
            analysis=analysis,
            raw_output=raw,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=thk_tok,
            cost_usd=actual_cost,
            provider=self.name,
        )
