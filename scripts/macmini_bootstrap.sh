#!/usr/bin/env bash
# Mac mini bootstrap: pull phase-2, install extras, import Phase 1
# data (without touching market_ticks), run tests, smoke the agent.
#
# Prereqs (one-time, from your MacBook):
#   scp /tmp/phase1_data.db       macmini:~/phase1_data.db
#   scp data/models/latest.joblib macmini:~/latest.joblib
#
# Usage (on Mac mini, from the tennis-edge repo root):
#   bash scripts/macmini_bootstrap.sh
#
# Idempotent: safe to re-run. Each step checks before acting.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Tennis-Edge Mac mini bootstrap ==="
echo "project root: $PROJECT_ROOT"
echo

# ---- 1. Git: make sure we are on phase-2 and pulled ----
echo "[1/6] git fetch + checkout phase-2"
git fetch origin
git checkout phase-2
git pull origin phase-2
echo "    HEAD: $(git log --oneline -1)"
echo

# ---- 2. Venv + install agent extras ----
echo "[2/6] install agent extras (google-genai)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        echo "    activated .venv"
    else
        echo "    WARN: no .venv detected. Continuing with system python."
    fi
fi
pip install -e '.[agent]' --quiet
echo "    ok"

# Python 3.14 added a security check that skips .pth files with the
# macOS UF_HIDDEN flag set. On this project's path (which contains a
# space) Finder and some backup tools set that flag, breaking the
# editable install silently. Clear the flag on every .pth in the
# venv site-packages so `import tennis_edge` works.
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [[ -n "$SITE_PKG" && -d "$SITE_PKG" ]]; then
    chflags nohidden "$SITE_PKG"/*.pth 2>/dev/null || true
fi
echo

# ---- 3. Model artifact ----
echo "[3/6] model artifact"
mkdir -p data/models
if [[ -f data/models/latest.joblib ]]; then
    echo "    data/models/latest.joblib already present — skipping"
elif [[ -f ~/latest.joblib ]]; then
    mv ~/latest.joblib data/models/latest.joblib
    echo "    moved ~/latest.joblib → data/models/latest.joblib"
else
    echo "    ERROR: no model found."
    echo "      From MacBook: scp data/models/latest.joblib macmini:~/latest.joblib"
    exit 1
fi
echo

# ---- 4. .env for Gemini key ----
echo "[4/6] .env / Gemini key"
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "    created .env from template — EDIT IT before starting the agent:"
    echo "      nano .env    # fill TENNIS_EDGE_GEMINI_KEY=..."
else
    echo "    .env already present — skipping (check TENNIS_EDGE_GEMINI_KEY is set)"
fi
echo

# ---- 5. Import Phase 1 data (without touching market_ticks) ----
echo "[5/6] import Phase 1 tables (players/matches/rankings/glicko2_ratings)"
if [[ ! -f ~/phase1_data.db ]]; then
    echo "    ERROR: ~/phase1_data.db not found."
    echo "      From MacBook:"
    echo "        python -c \"import sqlite3,os;s,d=sqlite3.connect('data/tennis_edge.db'),sqlite3.connect('/tmp/phase1_data.db');s.backup(d);s.close();d.execute('DROP TABLE IF EXISTS market_ticks');d.execute('VACUUM');d.commit();d.close()\""
    echo "        scp /tmp/phase1_data.db macmini:~/phase1_data.db"
    exit 1
fi
python scripts/import_phase1_data.py ~/phase1_data.db
echo

# ---- 6. Tests ----
echo "[6/6] pytest"
pytest tests/ -q
echo

# ---- Next steps ----
cat <<EOF

=== bootstrap done ===

Verify Gemini key is set:
  grep ^TENNIS_EDGE_GEMINI_KEY .env

Smoke test (high threshold so it barely burns API):
  tmux new -s agent-test
  tennis-edge agent start --min-edge 0.15
  # watch logs 3-5 min, Ctrl-C, then in another shell:
  tennis-edge agent status

When ready for real run (default --min-edge 0.08):
  tmux kill-session -t agent-test
  tmux new -s agent
  tennis-edge agent start
  # Ctrl-B D to detach. tmux attach -t agent to come back.

Do not touch the tick-logger tmux session — agent only reads market_ticks.

Optionally clean up the transferred file (Phase 1 data is now in the DB):
  rm ~/phase1_data.db
EOF
