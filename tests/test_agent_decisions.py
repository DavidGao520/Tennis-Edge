"""Tests for the agent decision log (JSONL append-only).

Covers:
  - EvAnalysis / AgentDecision / SettlementRecord schema validation
  - Crash-safe append (round-trip, multi-record ordering)
  - Replay: decision + settlement join by decision_id
  - Malformed-line tolerance
  - Missing-file tolerance
  - prompt_hash determinism
  - count_decisions cheap line-count
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tennis_edge.agent.decisions import (
    AgentDecision,
    DecisionLog,
    EvAnalysis,
    SettlementRecord,
    prompt_hash,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _analysis(**kwargs) -> EvAnalysis:
    base = dict(
        edge_estimate=0.62,
        recommendation="BUY_YES",
        confidence="medium",
        reasoning="Holmgren has a strong clay H2H advantage and market overreacted to set 1.",
        key_factors=["H2H 22-15 on clay", "Market reacted to set-1 score only"],
    )
    base.update(kwargs)
    return EvAnalysis(**base)


def _decision(**kwargs) -> AgentDecision:
    base = dict(
        ts=datetime(2026, 4, 18, 14, 30, 0, tzinfo=timezone.utc),
        run_id="run-abc",
        decision_id="dec-0001",
        ticker="KXATPMATCH-26MAY02HOLMBRO-HOL",
        model_pre_match=0.53,
        market_yes_cents=15,
        edge_at_decision=0.38,
        llm_provider="gemini-2.5-pro",
        llm_prompt_hash="a" * 16,
        llm_raw_output='{"edge_estimate": 0.62, "recommendation": "BUY_YES"}',
        analysis=_analysis(),
        mode="shadow",
    )
    base.update(kwargs)
    return AgentDecision(**base)


def _settlement(**kwargs) -> SettlementRecord:
    base = dict(
        ts=datetime(2026, 4, 18, 17, 45, 0, tzinfo=timezone.utc),
        decision_id="dec-0001",
        ticker="KXATPMATCH-26MAY02HOLMBRO-HOL",
        outcome="won",
        realized_pnl=42.50,
        settled_at=datetime(2026, 4, 18, 17, 40, 0, tzinfo=timezone.utc),
    )
    base.update(kwargs)
    return SettlementRecord(**base)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_ev_analysis_rejects_out_of_range_edge():
    with pytest.raises(ValidationError):
        _analysis(edge_estimate=1.5)
    with pytest.raises(ValidationError):
        _analysis(edge_estimate=-0.1)


def test_ev_analysis_rejects_unknown_recommendation():
    with pytest.raises(ValidationError):
        _analysis(recommendation="MAYBE_BUY")


def test_ev_analysis_rejects_unknown_confidence():
    with pytest.raises(ValidationError):
        _analysis(confidence="very-high")


def test_ev_analysis_caps_key_factors_at_five():
    with pytest.raises(ValidationError):
        _analysis(key_factors=["a", "b", "c", "d", "e", "f"])


def test_ev_analysis_caps_reasoning_length():
    with pytest.raises(ValidationError):
        _analysis(reasoning="x" * 2001)


def test_decision_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        _decision(mode="yolo")


def test_decision_rejects_out_of_range_market_cents():
    with pytest.raises(ValidationError):
        _decision(market_yes_cents=101)
    with pytest.raises(ValidationError):
        _decision(market_yes_cents=-1)


def test_decision_rejects_out_of_range_model_prob():
    with pytest.raises(ValidationError):
        _decision(model_pre_match=1.2)


def test_settlement_rejects_unknown_outcome():
    with pytest.raises(ValidationError):
        _settlement(outcome="push")


# ---------------------------------------------------------------------------
# Round-trip + ordering
# ---------------------------------------------------------------------------


def test_append_and_read_single_decision(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    d = _decision()
    log.append_decision(d)

    loaded = list(log.iter_decisions())
    assert len(loaded) == 1
    assert loaded[0] == d


def test_append_preserves_order(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    ids = [f"dec-{i:04d}" for i in range(5)]
    for did in ids:
        log.append_decision(_decision(decision_id=did))

    loaded_ids = [d.decision_id for d in log.iter_decisions()]
    assert loaded_ids == ids


def test_count_decisions_matches_writes(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    assert log.count_decisions() == 0

    for i in range(3):
        log.append_decision(_decision(decision_id=f"dec-{i}"))

    assert log.count_decisions() == 3


def test_settlement_round_trip(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    s = _settlement()
    log.append_settlement(s)

    loaded = list(log.iter_settlements())
    assert len(loaded) == 1
    assert loaded[0] == s


# ---------------------------------------------------------------------------
# Replay / join
# ---------------------------------------------------------------------------


def test_replay_joins_decision_with_settlement(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    d = _decision(decision_id="dec-X")
    s = _settlement(decision_id="dec-X", outcome="won", realized_pnl=25.0)
    log.append_decision(d)
    log.append_settlement(s)

    pairs = list(log.replay())
    assert len(pairs) == 1
    dec, settle = pairs[0]
    assert dec.decision_id == "dec-X"
    assert settle is not None
    assert settle.outcome == "won"
    assert settle.realized_pnl == 25.0


def test_replay_yields_none_for_unsettled_decision(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="still-open"))

    pairs = list(log.replay())
    assert len(pairs) == 1
    dec, settle = pairs[0]
    assert dec.decision_id == "still-open"
    assert settle is None


def test_replay_handles_settlement_arriving_out_of_order(tmp_path):
    """Settlement for dec-2 may land before settlement for dec-1 and
    even before dec-2 itself. Replay should still pair correctly."""
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    # Settlement first, then two decisions in their own order.
    log.append_settlement(_settlement(decision_id="dec-2", outcome="lost", realized_pnl=-20.0))
    log.append_decision(_decision(decision_id="dec-1"))
    log.append_decision(_decision(decision_id="dec-2"))

    by_id = {d.decision_id: (d, s) for d, s in log.replay()}
    assert by_id["dec-1"][1] is None
    assert by_id["dec-2"][1] is not None
    assert by_id["dec-2"][1].outcome == "lost"


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------


def test_iter_tolerates_blank_lines(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="a"))
    # Simulate an operator hand-editing the file.
    with log.decisions_path.open("a") as f:
        f.write("\n\n   \n")
    log.append_decision(_decision(decision_id="b"))

    ids = [d.decision_id for d in log.iter_decisions()]
    assert ids == ["a", "b"]


def test_iter_skips_malformed_line_and_continues(tmp_path, caplog):
    log = DecisionLog(tmp_path / "d.jsonl", tmp_path / "s.jsonl")
    log.append_decision(_decision(decision_id="good-1"))
    with log.decisions_path.open("a") as f:
        f.write('{"not": "a valid AgentDecision"}\n')
    log.append_decision(_decision(decision_id="good-2"))

    with caplog.at_level("WARNING"):
        ids = [d.decision_id for d in log.iter_decisions()]

    assert ids == ["good-1", "good-2"]
    assert any("invalid" in rec.message for rec in caplog.records)


def test_iter_missing_file_is_empty(tmp_path):
    log = DecisionLog(tmp_path / "does_not_exist.jsonl", tmp_path / "s.jsonl")
    assert list(log.iter_decisions()) == []
    assert log.count_decisions() == 0


def test_parent_dir_is_created(tmp_path):
    # Paths under nested non-existent directories should still work.
    log = DecisionLog(
        tmp_path / "nested" / "dir" / "d.jsonl",
        tmp_path / "nested" / "dir" / "s.jsonl",
    )
    log.append_decision(_decision())
    assert log.decisions_path.exists()


# ---------------------------------------------------------------------------
# prompt_hash
# ---------------------------------------------------------------------------


def test_prompt_hash_is_deterministic():
    h1 = prompt_hash("tmpl", {"a": 1, "b": 2})
    h2 = prompt_hash("tmpl", {"a": 1, "b": 2})
    assert h1 == h2


def test_prompt_hash_stable_under_key_reorder():
    h1 = prompt_hash("tmpl", {"a": 1, "b": 2})
    h2 = prompt_hash("tmpl", {"b": 2, "a": 1})
    assert h1 == h2


def test_prompt_hash_changes_with_template():
    h1 = prompt_hash("tmpl-v1", {"a": 1})
    h2 = prompt_hash("tmpl-v2", {"a": 1})
    assert h1 != h2


def test_prompt_hash_changes_with_inputs():
    h1 = prompt_hash("tmpl", {"a": 1})
    h2 = prompt_hash("tmpl", {"a": 2})
    assert h1 != h2


def test_prompt_hash_is_sixteen_chars():
    h = prompt_hash("tmpl", {"a": 1})
    assert len(h) == 16
    # sha256 hex is lowercase — sanity check we are using hex, not b64.
    assert all(c in "0123456789abcdef" for c in h)
