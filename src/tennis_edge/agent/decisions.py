"""Phase 2 Agent decision log.

Two append-only JSONL files:

  data/agent_decisions.jsonl   — one line per (model → LLM → maybe order)
                                 cycle. Primary key: decision_id.
  data/agent_settlements.jsonl — one line per settled market, written by
                                 the settlement poller after Kalshi
                                 resolves. FK: decision_id.

Why two files instead of rewriting one: append-only is safe under crash
(O_APPEND is atomic per small write on POSIX), the settlement poller
can run on its own schedule, and `replay()` does the join at read time.
Rewriting a single JSONL to backfill outcome is how you lose a row.

Why JSONL instead of SQLite: Phase 3A expects ~20 decisions/day over
6 weeks ≈ 840 rows. A flat file at that size is cheaper than a schema
migration and stays greppable and crash-safe. Promote to SQLite only
when we need indexed queries — we do not today.

Durability: every append flushes and fsyncs. An agent that loses
decisions to power loss is an agent you cannot audit.

                   ┌────────────────────┐
   loop.py ───────►│ append_decision()  │──► decisions.jsonl (fsync)
                   └────────────────────┘
                   ┌────────────────────┐
   poller ────────►│ append_settlement()│──► settlements.jsonl (fsync)
                   └────────────────────┘
                   ┌────────────────────┐
   replay() ──────►│ join on decision_id│──► iter[(Decision, Settlement?)]
                   └────────────────────┘
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EvAnalysis(BaseModel):
    """Structured, validated LLM output.

    The LLM provider (agent/llm.py) is expected to return JSON that
    validates against this schema. A parse failure is counted as an LLM
    failure by the 3x-consecutive kill switch.
    """

    edge_estimate: float = Field(
        ge=0.0, le=1.0,
        description="LLM's point estimate of P(YES wins) as a probability.",
    )
    recommendation: Literal["BUY_YES", "BUY_NO", "SKIP"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(max_length=2000)
    key_factors: list[str] = Field(default_factory=list, max_length=5)

    # Optional: the single biggest risk to this call. Free-text; used
    # for post-mortem analytics (group losing decisions by risk class
    # to see whether injuries, fatigue, or altitude bites us most).
    # Optional for backward compatibility with v2.0 logged decisions.
    key_risk: str | None = Field(default=None, max_length=500)


class AgentDecision(BaseModel):
    """One full decision record. Append-only.

    decision_id is the join key with SettlementRecord. Generate a UUID4
    in the loop and retain it across any retry so idempotent order
    placement remains traceable.
    """

    # identity
    ts: datetime
    run_id: str
    decision_id: str
    ticker: str

    # prices at decision time
    model_pre_match: float = Field(
        ge=0.0, le=1.0,
        description="Glicko-2 model pre-match anchor probability for YES.",
    )
    market_yes_cents: int = Field(ge=0, le=100)
    edge_at_decision: float = Field(
        description="model_pre_match - market_yes_cents/100 at decision time.",
    )

    # LLM
    llm_provider: str
    llm_prompt_hash: str
    llm_raw_output: str
    analysis: EvAnalysis

    # 3A = []; 3B/3C may populate after screenshot pipeline lands
    screenshot_paths: list[str] = Field(default_factory=list)

    # execution
    mode: Literal["shadow", "human_in_loop", "auto"]
    executed: bool = False
    order_id: str | None = None
    edge_at_execution: float | None = None
    reject_reason: str | None = None


class SettlementRecord(BaseModel):
    """Settlement event tied back to a decision.

    Written asynchronously by the settlement poller. A decision may
    never get a settlement (agent decided not to trade, market voided,
    etc.). `replay()` returns None for the settlement in that case.
    """

    ts: datetime
    decision_id: str
    ticker: str
    outcome: Literal["won", "lost", "void"]
    realized_pnl: float
    settled_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def prompt_hash(template: str, inputs: dict) -> str:
    """Stable 16-char sha256 hex over (template + sorted inputs).

    Used to group A/B-tested decisions by prompt version. sort_keys so
    dict ordering doesn't break grouping.
    """
    blob = template + "\x1e" + json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class DecisionLog:
    """Append-only JSONL log for agent decisions and settlements."""

    def __init__(
        self,
        decisions_path: str | os.PathLike[str],
        settlements_path: str | os.PathLike[str],
    ):
        self.decisions_path = Path(decisions_path)
        self.settlements_path = Path(settlements_path)
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        self.settlements_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writers ----

    def append_decision(self, decision: AgentDecision) -> None:
        """Write one decision line and fsync before returning."""
        self._append(self.decisions_path, decision.model_dump_json())

    def append_settlement(self, settlement: SettlementRecord) -> None:
        """Write one settlement line and fsync before returning."""
        self._append(self.settlements_path, settlement.model_dump_json())

    @staticmethod
    def _append(path: Path, line: str) -> None:
        # Open in append mode. O_APPEND guarantees atomic position seek
        # before each write on POSIX, so multi-process writes could not
        # interleave even if we ever have them (we do not today).
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ---- readers ----

    def iter_decisions(self) -> Iterator[AgentDecision]:
        """Yield decisions in insertion order. Skips malformed lines.

        A corrupt JSONL line is logged and skipped — a single bad line
        must not halt a replay of the whole log.
        """
        yield from self._iter(self.decisions_path, AgentDecision)

    def iter_settlements(self) -> Iterator[SettlementRecord]:
        """Yield settlements in insertion order. Skips malformed lines."""
        yield from self._iter(self.settlements_path, SettlementRecord)

    @staticmethod
    def _iter(path: Path, model: type[BaseModel]):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue  # tolerate blank lines from hand-editing
                try:
                    yield model.model_validate_json(line)
                except ValidationError as e:
                    logger.warning(
                        "%s line %d invalid, skipping: %s", path.name, lineno, e
                    )

    def replay(self) -> Iterator[tuple[AgentDecision, SettlementRecord | None]]:
        """Yield each decision paired with its settlement (or None).

        Loads all settlements into memory first since a settlement can
        arrive out of order relative to decisions. At Phase 3A volumes
        (~840 decisions over 6 weeks) this is trivially small.
        """
        settlements: dict[str, SettlementRecord] = {}
        for s in self.iter_settlements():
            # If a market gets "re-settled" somehow (should never happen)
            # keep the most recent record.
            settlements[s.decision_id] = s
        for d in self.iter_decisions():
            yield d, settlements.get(d.decision_id)

    # ---- utility ----

    def count_decisions(self) -> int:
        """Cheap line count. Used by /agent status without full parse."""
        if not self.decisions_path.exists():
            return 0
        with self.decisions_path.open("rb") as f:
            return sum(1 for line in f if line.strip())
