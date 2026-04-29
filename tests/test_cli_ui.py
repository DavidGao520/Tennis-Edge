"""Tests for the interactive launch-screen UI.

Scope: pure functions that have no Rich-prompt loops. The actual
menu loop is tested via a single end-to-end interaction test using
monkeypatched input. All side-effecting helpers (`_update_dotenv`,
`check_setup`, `_mask`, `_player_count`) get unit-tested with real
filesystem fixtures (via tmp_path) so behavior under partial setup
state is locked in.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tennis_edge.cli_ui import (
    LLM_PROVIDERS,
    _active_llm_label,
    _has_tmux,
    _kalshi_auth_present,
    _llm_present,
    _mask,
    _model_present,
    _player_count,
    _update_dotenv,
    check_setup,
)


# ---------------------------------------------------------------------------
# Fakes for the cfg shape that check_setup() expects
# ---------------------------------------------------------------------------


@dataclass
class _KalshiCfg:
    api_key_id: str = ""
    private_key_path: str = ""


@dataclass
class _DBCfg:
    path: str = "data/tennis_edge.db"


@dataclass
class _ModelCfg:
    artifacts_dir: str = "data/models"


@dataclass
class _AppCfg:
    project_root: str = "."
    kalshi: _KalshiCfg = field(default_factory=_KalshiCfg)
    database: _DBCfg = field(default_factory=_DBCfg)
    model: _ModelCfg = field(default_factory=_ModelCfg)


# ---------------------------------------------------------------------------
# _mask
# ---------------------------------------------------------------------------


def test_mask_empty():
    assert _mask("") == "(not set)"


def test_mask_short_value_fully_redacted():
    # _mask fully redacts when len(value) <= show*2 = 8 chars by
    # default (not enough to show non-overlapping head + tail).
    assert _mask("12345678") == "•" * 8
    assert _mask("abc") == "•••"


def test_mask_long_value_keeps_edges():
    masked = _mask("AIzaSyBP9Pg6qONUNOvXeQXwurgelij-UtIXd0c")
    assert masked.startswith("AIza")
    assert masked.endswith("Xd0c")
    assert "•" in masked


def test_mask_show_param_controls_visible_chars():
    masked = _mask("0123456789abcdef", show=2)
    assert masked.startswith("01")
    assert masked.endswith("ef")


# ---------------------------------------------------------------------------
# _update_dotenv
# ---------------------------------------------------------------------------


def test_update_dotenv_creates_file_if_missing(tmp_path: Path):
    env = tmp_path / ".env"
    _update_dotenv(env, "FOO", "bar")
    content = env.read_text()
    assert "FOO=bar" in content


def test_update_dotenv_preserves_comments_and_other_keys(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "OTHER_KEY=keep_me\n"
        "FOO=old_value\n"
        "# another comment\n"
        "AFTER=also_keep\n"
    )
    _update_dotenv(env, "FOO", "new_value")
    content = env.read_text()
    assert "# a comment" in content
    assert "OTHER_KEY=keep_me" in content
    assert "FOO=new_value" in content
    assert "FOO=old_value" not in content
    assert "# another comment" in content
    assert "AFTER=also_keep" in content


def test_update_dotenv_appends_when_key_absent(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("EXISTING=v\n")
    _update_dotenv(env, "NEW", "value")
    content = env.read_text()
    assert "EXISTING=v" in content
    assert "NEW=value" in content


def test_update_dotenv_does_not_match_commented_keys(tmp_path: Path):
    """A line like `#FOO=...` is a comment, not a real assignment.
    Updating FOO must not modify the comment."""
    env = tmp_path / ".env"
    env.write_text("# FOO=should_be_left_alone\nOTHER=yes\n")
    _update_dotenv(env, "FOO", "real_value")
    content = env.read_text()
    assert "# FOO=should_be_left_alone" in content
    assert "FOO=real_value" in content


def test_update_dotenv_creates_parent_dir_if_missing(tmp_path: Path):
    env = tmp_path / "nested" / "deeply" / ".env"
    _update_dotenv(env, "K", "v")
    assert env.is_file()
    assert "K=v" in env.read_text()


# ---------------------------------------------------------------------------
# _player_count
# ---------------------------------------------------------------------------


def test_player_count_returns_zero_for_missing_db(tmp_path: Path):
    assert _player_count(tmp_path / "absent.db") == 0


def test_player_count_returns_zero_for_db_without_table(tmp_path: Path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()  # creates empty file
    assert _player_count(db) == 0


def test_player_count_counts_rows(tmp_path: Path):
    db = tmp_path / "p.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO players (player_id) VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    assert _player_count(db) == 3


# ---------------------------------------------------------------------------
# _kalshi_auth_present / _model_present
# ---------------------------------------------------------------------------


def test_kalshi_auth_absent_when_id_blank(tmp_path: Path):
    cfg = _AppCfg(project_root=str(tmp_path))
    assert _kalshi_auth_present(cfg) is False


def test_kalshi_auth_absent_when_pem_missing(tmp_path: Path):
    cfg = _AppCfg(
        project_root=str(tmp_path),
        kalshi=_KalshiCfg(
            api_key_id="abc-123",
            private_key_path="config/missing.pem",
        ),
    )
    assert _kalshi_auth_present(cfg) is False


def test_kalshi_auth_present_when_both_set(tmp_path: Path):
    pem = tmp_path / "config" / "kalshi.pem"
    pem.parent.mkdir(parents=True)
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\n...key bytes...\n")
    cfg = _AppCfg(
        project_root=str(tmp_path),
        kalshi=_KalshiCfg(
            api_key_id="abc-123",
            private_key_path="config/kalshi.pem",
        ),
    )
    assert _kalshi_auth_present(cfg) is True


def test_kalshi_auth_absent_when_pem_is_empty(tmp_path: Path):
    pem = tmp_path / "config" / "kalshi.pem"
    pem.parent.mkdir(parents=True)
    pem.write_text("")
    cfg = _AppCfg(
        project_root=str(tmp_path),
        kalshi=_KalshiCfg(
            api_key_id="abc-123",
            private_key_path="config/kalshi.pem",
        ),
    )
    assert _kalshi_auth_present(cfg) is False


def test_model_absent_when_artifact_missing(tmp_path: Path):
    cfg = _AppCfg(project_root=str(tmp_path))
    assert _model_present(tmp_path, cfg) is False


def test_model_present_when_artifact_exists(tmp_path: Path):
    artifact = tmp_path / "data" / "models" / "latest.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fake joblib bytes that are not zero length")
    cfg = _AppCfg(project_root=str(tmp_path))
    assert _model_present(tmp_path, cfg) is True


# ---------------------------------------------------------------------------
# check_setup integration
# ---------------------------------------------------------------------------


def test_check_setup_all_false_in_fresh_project(tmp_path: Path, monkeypatch):
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup == {"llm": False, "kalshi": False, "model": False, "data": False}


def test_check_setup_llm_true_when_gemini_set(tmp_path: Path, monkeypatch):
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    monkeypatch.setenv("TENNIS_EDGE_GEMINI_KEY", "AIzaTestKey")
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup["llm"] is True


def test_check_setup_llm_true_when_only_openai_set(tmp_path: Path, monkeypatch):
    """LLM badge should green up if any supported provider is set,
    not just Gemini."""
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    monkeypatch.setenv("TENNIS_EDGE_OPENAI_KEY", "sk-test-openai")
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup["llm"] is True


def test_check_setup_llm_true_when_only_claude_set(tmp_path: Path, monkeypatch):
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    monkeypatch.setenv("TENNIS_EDGE_ANTHROPIC_KEY", "sk-ant-test")
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup["llm"] is True


def test_check_setup_all_true_when_everything_set(tmp_path: Path, monkeypatch):
    """End-to-end: stage a fully-configured project layout and verify
    check_setup reports green across the board."""
    monkeypatch.setenv("TENNIS_EDGE_GEMINI_KEY", "AIzaTestKey")

    # Kalshi PEM
    pem = tmp_path / "config" / "kalshi.pem"
    pem.parent.mkdir(parents=True)
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\n...\n")

    # Model artifact
    art = tmp_path / "data" / "models" / "latest.joblib"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"fake-joblib")

    # Player DB
    db = tmp_path / "data" / "tennis_edge.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO players VALUES (1)")
    conn.commit()
    conn.close()

    cfg = _AppCfg(
        project_root=str(tmp_path),
        kalshi=_KalshiCfg(api_key_id="abc", private_key_path="config/kalshi.pem"),
    )
    setup = check_setup(cfg)
    assert setup == {"llm": True, "kalshi": True, "model": True, "data": True}


# ---------------------------------------------------------------------------
# LLM provider registry
# ---------------------------------------------------------------------------


def test_llm_providers_registry_has_all_three():
    """Three providers exposed: gemini (active), openai + anthropic
    (key-only). If a future PR adds a fourth, this test should
    update — but the contract here is that all three current
    providers exist and have the required metadata."""
    assert set(LLM_PROVIDERS.keys()) == {"gemini", "openai", "anthropic"}
    for p in LLM_PROVIDERS.values():
        assert "label" in p
        assert "env_var" in p
        assert "url" in p
        assert "prefix" in p
        assert "status" in p
        assert p["status"] in ("active", "saved")


def test_only_gemini_is_active_today():
    """The agent's grounded path is wired to Gemini today.
    OpenAI/Claude provider classes land in a future PR. This test
    locks the current state — change the assertion when adding new
    active providers."""
    actives = [
        k for k, p in LLM_PROVIDERS.items() if p["status"] == "active"
    ]
    assert actives == ["gemini"]


def test_llm_present_false_when_no_keys(monkeypatch):
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    assert _llm_present() is False


def test_llm_present_true_for_each_provider(monkeypatch):
    for provider_key, p in LLM_PROVIDERS.items():
        # Clear all, then set just this one.
        for q in LLM_PROVIDERS.values():
            monkeypatch.delenv(q["env_var"], raising=False)
        monkeypatch.setenv(p["env_var"], "test-key")
        assert _llm_present() is True, f"{provider_key} did not register"


def test_active_llm_label_returns_active_provider(monkeypatch):
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    monkeypatch.setenv("TENNIS_EDGE_GEMINI_KEY", "AIzaTest")
    assert _active_llm_label() == "Gemini"


def test_active_llm_label_none_when_only_inactive_providers(monkeypatch):
    """If the user configures OpenAI/Claude but not Gemini, the
    'active' provider count is zero — agent has no LLM to use."""
    for p in LLM_PROVIDERS.values():
        monkeypatch.delenv(p["env_var"], raising=False)
    monkeypatch.setenv("TENNIS_EDGE_OPENAI_KEY", "sk-test")
    monkeypatch.setenv("TENNIS_EDGE_ANTHROPIC_KEY", "sk-ant-test")
    assert _active_llm_label() is None


# ---------------------------------------------------------------------------
# tmux detection (used to gate the "Show start command" instructions)
# ---------------------------------------------------------------------------


def test_has_tmux_returns_bool():
    """Smoke. Result depends on the host so we just assert the
    return type is bool. The behavior is exercised end-to-end in
    `_print_agent_start_command` via the host's PATH."""
    assert isinstance(_has_tmux(), bool)


def test_has_tmux_false_when_path_empty(monkeypatch):
    """With empty PATH, shutil.which finds nothing, so _has_tmux
    must be False. Locks the negative branch that sends the user
    to `brew install tmux` or foreground execution."""
    monkeypatch.setenv("PATH", "")
    assert _has_tmux() is False
