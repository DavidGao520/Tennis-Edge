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
    _kalshi_auth_present,
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
    monkeypatch.delenv("TENNIS_EDGE_GEMINI_KEY", raising=False)
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup == {"gemini": False, "kalshi": False, "model": False, "data": False}


def test_check_setup_gemini_true_when_env_set(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TENNIS_EDGE_GEMINI_KEY", "AIzaTestKey")
    cfg = _AppCfg(project_root=str(tmp_path))
    setup = check_setup(cfg)
    assert setup["gemini"] is True
    # Other flags still false because the rest of the project is empty.
    assert setup["kalshi"] is False
    assert setup["model"] is False
    assert setup["data"] is False


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
    assert setup == {"gemini": True, "kalshi": True, "model": True, "data": True}
