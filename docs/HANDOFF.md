# Tennis-Edge Handoff — April 2026

A self-contained snapshot of where the project is and what's next.
**For the human**: keep this current as the ground truth between
sessions. **For an AI assistant resuming work**: read this first,
then `docs/PROGRESS.md` for full chronology, then dive in.

---

## Current state (one-paragraph version)

Phase 3A v2.1 agent is production-ready and merged to `main`.
Single-process: one `tennis-edge agent start` spins up
MonitorBridge + AgentLoop + SettlementPoller + SafetyMonitor.
LLM is Gemini 3.1 Pro Preview with Google Search grounding,
9-item research checklist, system_instruction persona, key_risk
field. 258 unit tests + 5 real-Gemini eval cases (the v1 failure
mode is locked as a regression test). Interactive launch screen
with multi-LLM-provider onboarding (Gemini wired, OpenAI/Claude
keys saved for future PR). Kalshi auth wired, $500.44 balance,
0 positions. **Live `--executor live` path has never been
exercised against real Kalshi yet** — that's the next thing
to do.

---

## Branches and merges

```
main    = production line, 258 tests passing
agent   = David's lane — Phase 3B/3C agent work, multi-provider LLM, ...
cli     = teammate's lane — UI iteration, onboarding polish, ...
```

`agent` and `cli` branched off `main` after Phase 3A v2.1 was
merged. Each lane lives on its own branch and merges back to
`main` via `--no-ff` PRs. The old `phase-2` and `cli-ui` names
were retired because they referred to phase-1-of-product-plan
language that no longer matches.

When in doubt, branch off `main`. Long-running `agent` / `cli`
branches exist so the two contributors stay in their lanes
without trampling each other.

---

## What's done (in priority order of importance)

### 1. Phase 3A v2.1 agent (DONE, merged)

The grounded research agent that fixes the v1 failure mode (bot
trading into settled markets because it had no live awareness).

- `agent/loop.py` — signal subscriber + decision gates + executor
- `agent/llm.py` — `GeminiGroundedProvider` with Google Search,
  `system_instruction` persona, 9-item research checklist,
  pydantic-validated EvAnalysis output
- `agent/monitor_bridge.py` — embedded EVScanner that polls
  Kalshi REST every 15s and emits MonitorSignal via async callback
- `agent/decisions.py` — JSONL append-only log with `key_risk`
- `agent/settlement.py` — counterfactual P&L poller (also writes
  to `RiskManager.daily_pnl` so the kill switch actually trips)
- `agent/safety.py` — 7 trip reasons including
  `ORDER_CONSECUTIVE_FAILURES`, file-flag IPC for pause/flatten
- `strategy/risk.py` — concurrency-safe `check_and_reserve`

### 2. Required eval suite (DONE, locked)

`tests/test_agent_gemini_eval.py` runs 5 hand-built scenarios
against real Gemini. **The `extreme_low_price` case is the v1
failure regression test** — if it ever fails, the prompt change
that broke it does not ship. Run with `pytest -m eval -v -s`,
costs ~$0.10.

### 3. Interactive CLI (DONE, merged)

`tennis-edge` (no subcommand) drops into a 5-option menu.
First-run wizard auto-triggers when API keys / model / data are
missing. Settings → LLM API key opens a provider picker
(Gemini / OpenAI / Claude). Power-user CLI completely backward
compatible.

### 4. Operational fixes (DONE, merged)

- macOS Python 3.14 ignores `.pth` files with `UF_HIDDEN` flag —
  fixed via `sitecustomize.py` (project lives under iCloud Desktop
  which auto-flags hidden)
- `.env` auto-load in `cli.py main()` — TENNIS_EDGE__* vars work
  without manual `export`
- `client_order_id` idempotency on `OrderRequest` — retry-safe
  against Kalshi network flakes
- `scripts/macmini_bootstrap.sh` — one-shot setup for any new
  machine

---

## What's NOT done (next-priority queue)

### A. **First live run** ⚠️ unblocked, urgent for product validation

The agent has never placed a real Kalshi order. We have all the
plumbing (KalshiClient + auth + place_order + idempotency key)
but never run end-to-end on real money. The first live run is
the highest-value next step.

```bash
# Pre-flight (in main)
tennis-edge   # menu → [4] Settings → [3] Validate → confirm balance
brew install tmux  # MacBook doesn't have it

# Start live with $1/order, $20 total cap, $10 daily loss kill
tmux new -s agent
# Then paste the live-mode command from menu [2] Agent → [2]
```

Watch for:
1. Does Kalshi accept our OrderRequest body shape on the first
   real submission?
2. Does the post-LLM edge re-check fire correctly on real prices?
3. Does the order_id flow through into the AgentDecision JSONL?
4. After settlement (~hours later), does the SettlementPoller
   correctly compute realized P&L and update RiskManager?

**Risk if first live run hits an issue**: a malformed POST body
fails with 400; we catch the exception, `risk.release()` runs,
no real money lost. After 3 consecutive order failures the
ORDER_CONSECUTIVE_FAILURES kill switch trips. Recovery is "fix
the bug, push, restart".

### B. Multi-provider LLM (UI shipped, agent wiring pending)

The Settings UI lets users configure OpenAI and Claude keys, but
neither provider is wired into the agent's grounded path yet.
Each provider needs:

1. New class `OpenAIProvider(LLMProvider)` using the OpenAI
   Responses API + `tools` for grounding
2. New class `AnthropicProvider(LLMProvider)` using Claude's
   `web_search` tool
3. CLI flag `--llm-provider gemini|openai|anthropic` on
   `agent start`
4. Per-provider PricingRates constants
5. Eval suite cases per provider
6. Decision log records `llm_provider` field for A/B comparison

Estimated 6-8 hours per provider, done properly.

### C. Better Kelly calibration

Current confidence multipliers `(low=0.0, medium=0.5, high=1.0)`
are guesses, no empirical basis. Plan: after 50 settled trades,
group counterfactual win-rate by `confidence` bucket and
recalibrate. If high-conf wins 60% but medium-conf wins 65%,
the multipliers are inverted from reality.

### D. Position management loop (Phase 3D)

Once positions are open, currently we just wait for settlement.
A real trader cuts losses on adverse moves, takes profits on
favorable runs. Out of v2 scope but a natural next phase.

### E. Real backtest (Anthony's lane)

Mac mini's tick logger has been collecting `market_ticks` since
April 17. By demo day there's ~5 weeks of data. Anthony's
workstream is to rewrite `backtest/engine.py` to consume real
ticks instead of synthetic odds. Not directly David's lane but
worth tracking.

### F. Arbitrage feasibility (Billy's lane)

Week 1 gate study to determine whether Kalshi tennis arb has
≥5 actionable signals/day. If not, Billy pivots to backtest
support. **Untouched as of this writing.**

---

## The minute-zero command for every new shell

```bash
cd "/Users/gaoyuan/Desktop/Trading Bot/tennis-edge"
source .venv/bin/activate
git pull
tennis-edge       # interactive menu
```

Or run subcommands directly: `tennis-edge agent status`,
`tennis-edge agent start --executor paper ...`, etc.

---

## Things that bite

1. **macOS iCloud Desktop sync re-flags `.pth` files as hidden**.
   Solution shipped: `sitecustomize.py` in venv site-packages
   (auto-installed by `scripts/macmini_bootstrap.sh`). If
   `import tennis_edge` ever breaks, run the bootstrap script.

2. **Real API keys must NEVER appear in tests**. Use synthetic
   strings like `"AIza0000000000000000000000000000000fake"`. We
   leaked one Gemini key (April 28); Google's auto-detection
   disabled it within 30 minutes. Postmortem in PROGRESS.md.

3. **macOS doesn't ship tmux**. The "Show start command" UI
   detects this via `shutil.which("tmux")` and falls back to
   instructions for `brew install tmux` or foreground execution.

4. **Kalshi tennis markets have low / no liquidity at certain
   times of day**. Saturday afternoon US time saw all 86 active
   markets with zero bids/asks (Sunday matches not yet trading).
   The agent silently skips markets with no mid_price. This is
   correct behavior; just be aware "no decisions in the last
   hour" doesn't mean broken.

5. **Gemini grounded calls take 18-30 seconds and use ~1000+
   thinking tokens**. Set `max_output_tokens=8192` and
   `max_thinking_tokens=8192` (which we do). Truncated JSON =
   `LLMOutputError` = decision lost.

---

## How to resume work in a new chat

Tell the assistant:

> "I'm continuing Tennis-Edge work. Read `docs/HANDOFF.md` and
> `docs/PROGRESS.md` for current state. I'm working on the
> Agent track. My teammate is on the CLI track. Next priority
> is [X]."

Where [X] is one of:
- "First live agent run" (Section A above)
- "Add OpenAI/Claude provider classes" (Section B)
- "Backtest the agent's counterfactual P&L logic" (Section C)
- "Whatever else seems most urgent"

---

## Session-level commit conventions

```bash
GIT_AUTHOR_NAME=DavidGao520 \
GIT_AUTHOR_EMAIL=yuangao021804@gmail.com \
GIT_COMMITTER_NAME=DavidGao520 \
GIT_COMMITTER_EMAIL=yuangao021804@gmail.com \
git commit -m "..."
```

This keeps contributor list clean (no Co-Authored-By Claude).
The user's `.gitconfig` may not have these set, so always
override per-commit.

---

## Last verified state (replace this when you commit)

- Date: April 28-29, 2026
- Branches: `main` / `agent` / `cli` (renamed from `phase-2` / `cli-ui`)
- main HEAD: latest merge, includes HANDOFF.md
- Test suite: **258 passed, 5 deselected (eval)**
- Latest eval run: 5/5 passed (April 19, ~$0.10)
- Latest real Gemini smoke: passed (55 models accessible)
- Kalshi balance: $500.44, 0 open positions
- Active LLM provider: Gemini (rotated key after April 28 leak;
  rotated again after pasting in chat — current key valid)
