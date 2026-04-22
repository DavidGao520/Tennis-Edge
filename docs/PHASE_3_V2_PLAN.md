# Phase 3 v2 Plan — Monitor → Agent → Gemini Grounded → Auto-Execute

**Date**: April 19, 2026 (evening)
**Status**: locked via chat discussion; ready to code
**Supersedes**: v1 shadow-mode pipeline (Lane 1F `agent/loop.py` as-built)

---

## Why v2

v1 (shipped today) has a structural flaw that first-run surfaced:

1. **Local Glicko-2 model returns constant ~0.27 for unrated Challenger players** — the `edge_at_decision` gate becomes meaningless when the underlying probability is a default.
2. **Agent is blind to live scores** — exact same Phase 1 failure mode (Holmgren 53% / 15% market).
3. Context builder returns empty form/H2H for players outside Sackmann's dataset.

v1's architecture asks the LLM to reason over static pre-match features we compute locally. That only works when our features are rich — false for Challenger. v2 flips the contract: **Gemini becomes the research engine**, uses Google Search grounding to look up live state itself, and we stop pretending our local model is the truth.

User's manual $10K/month workflow is already Gemini-with-live-context. v2 makes the automated pipeline match that workflow.

---

## Architecture

```
  CLI launch screen
  ┌─────────────────────────────────────────────┐
  │  [1] Arbitrage   (Billy — not in scope)     │
  │  [2] Monitor     (standalone TUI, shipped)  │
  │  [3] Agent       (new daemon, this plan)    │
  └─────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                  tennis-edge agent start                    │
  │                                                             │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Monitor scanner (embedded, no TUI)  │                    │
  │  │  - Scan Kalshi every 2s             │                    │
  │  │  - Compute prematch EV per market   │                    │
  │  │  - Emit signal if EV threshold hit  │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 │ pub/sub callback                          │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Signal filter                       │                    │
  │  │  - Series whitelist (ATP/WTA Main)  │                    │
  │  │  - Price band 10c-90c               │                    │
  │  │  - Per-ticker cooldown              │                    │
  │  │  - Queue cap                        │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Gemini GROUNDED worker              │                    │
  │  │  - google_search tool enabled       │                    │
  │  │  - System prompt: "look up live     │                    │
  │  │    state, give real P(YES)"         │                    │
  │  │  - Returns EvAnalysis + confidence  │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Execution gate                      │                    │
  │  │  - real_edge ≥ 0.10?                │                    │
  │  │  - confidence ≠ low?                │                    │
  │  │  - post-LLM edge re-check (hard)?   │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Kelly sizer                         │                    │
  │  │  size = Kelly(P_real, P_market)     │                    │
  │  │       × kelly_fraction (0.25)       │                    │
  │  │       × confidence_mult (0.5/1.0)   │                    │
  │  │       capped at $50/market          │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ RiskManager.check_and_reserve       │                    │
  │  │  hard: $50/market, $500 total,      │                    │
  │  │        -$200 daily loss → KILL      │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ Executor (mode-switched)            │                    │
  │  │   paper (first 20): simulated fill  │                    │
  │  │   auto  (after):    real Kalshi ord │                    │
  │  │   manual confirm prompt: always on  │                    │
  │  │    30s timeout → default N          │                    │
  │  └──────────────┬──────────────────────┘                    │
  │                 ▼                                           │
  │  ┌─────────────────────────────────────┐                    │
  │  │ DecisionLog.append + SafetyMonitor  │                    │
  │  │ SettlementPoller (15min) for P&L    │                    │
  │  └─────────────────────────────────────┘                    │
  └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                  Kalshi REST (paper or live)
```

---

## Locked decisions (from 2026-04-19 chat)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Monitor → Agent = same process, pub/sub callback** | Simplest; Agent owns its own scanner instance with TUI disabled. No IPC, no DB queue table. |
| 2 | **Gemini Google Search grounding replaces screenshots** | ESPN / Flashscore / tennisabstract all live on the web. Gemini's search is more reliable than our headless Chromium would be. Screenshots deferred indefinitely. |
| 3 | **Hard stops: $50/market, $500 total exposure, -$200 daily loss** | Matches original Phase 3C plan. RiskManager already enforces these; wire them in for real this time. |
| 4 | **First 20 decisions = paper + manual confirm, then auto** | Paper (`PaperTradingEngine`) runs Gemini + Kelly + everything real except the actual `KalshiClient.place_order`. After 20, if results look sane, flip to live. |
| 5 | **Week 1 whitelist: `KXATPMATCH` + `KXWTAMATCH` only (no Challengers)** | Gemini's Google Search coverage of Challenger matches is untested. Validate the grounded pipeline on markets where ESPN/Flashscore actually have data. Challenger added Week 2. |

---

## Kelly sizing math (precise)

For BUY_YES at market price `P_m` (cents / 100) with Gemini-claimed true probability `P_g`:

```
raw_edge     = P_g - P_m
kelly_pct    = raw_edge / (1 - P_m)            # fractional Kelly formula
kelly_size   = kelly_pct
                × kelly_fraction (0.25)         # fractional Kelly to survive variance
                × confidence_mult               # 1.0 high / 0.5 medium / 0.0 low
                × bankroll
kelly_size   = min(kelly_size, $50)             # per-market hard cap
```

Symmetric for BUY_NO. `kelly_size ≤ 0` → skip.

**Worked example**: Holmgren at 15c, Gemini says 55% high confidence, bankroll $5999.
- raw_edge = 0.55 - 0.15 = 0.40
- kelly_pct = 0.40 / 0.85 = 0.47 (47% Kelly — that's why we fraction)
- kelly_size = 0.47 × 0.25 × 1.0 × 5999 = $704
- capped at $50
- Result: BUY_YES $50 (333 contracts at 15c)

---

## Execution gate — precedence

Every candidate traverses this gauntlet in order. First failure exits the path.

1. **Monitor signal gate**: prematch EV ≥ 0.15 AND ticker is on Main-tour whitelist.
2. **Price band**: 10c ≤ market_yes_cents ≤ 90c. Outside = match likely in late decided state.
3. **Cooldown**: per-ticker cooldown not active.
4. **Queue cap**: signal queue under `queue_max`.
5. **Budget**: `BudgetTracker.reserve(estimate)` succeeds. Fail → kill switch.
6. **Gemini call succeeds**: returns valid `EvAnalysis`. Fail → count toward 3x kill.
7. **Gemini edge gate**: `abs(gemini_edge)` ≥ 0.10 AND `confidence` ∈ {medium, high}.
8. **Post-LLM edge re-check (HARD, not soft)**: re-read latest tick; if `abs(edge_live)` < 0.08, REJECT. v1 logged these; v2 REJECTS (too risky to fill on stale price with real money).
9. **RiskManager.check_and_reserve**: passes $50/market + $500 total + kill switch check.
10. **Manual confirm prompt**: 30s timeout, N default. `YES` → execute.
11. **Executor.place_order** (paper or live based on mode flag).

Any failure → `AgentDecision` logged with `reject_reason`, `executed=False`. Success → `executed=True`, `order_id` populated.

---

## What we keep from v1 (no change)

| File | Role in v2 |
|---|---|
| `agent/decisions.py` | Unchanged. JSONL log + `AgentDecision` + `SettlementRecord`. |
| `agent/settlement.py` | Unchanged. `SettlementPoller` + `counterfactual_pnl`. Now both paper P&L AND real P&L flow through same join. |
| `agent/safety.py` | Unchanged module. `SafetyMonitor` + 5 kill switches + file-flag IPC. Wire in `DAILY_LOSS_LIMIT` against real `RiskManager.state.daily_pnl` (v1 stubbed with `_NullRisk`). |
| `strategy/risk.py` | Unchanged. `check_and_reserve` from Lane B. |
| `strategy/sizing.py` | Unchanged. `PositionSizer` reused for Kelly math. |
| `strategy/kelly.py` | Unchanged. |
| `exchange/paper.py` | Unchanged. `PaperTradingEngine` already implements `ExchangeClient` — swap in as executor for first 20 trades. |
| `exchange/client.py` | Unchanged. `KalshiClient.place_order` used by live executor. |
| `scanner.py` | Unchanged. `EVScanner.analyze_market_pair` reused inside Monitor bridge. |

## What v2 extends (small diffs)

| File | Change |
|---|---|
| `agent/llm.py` | Add `GeminiGroundedProvider` (subclass of `GeminiProvider` with `tools=[GoogleSearch()]`). Add `PROMPT_TEMPLATE_GROUNDED_V1` — research-engine style. |
| `agent/safety.py` | No code change, but daemon now passes real `RiskManager` instead of `_NullRisk`, real `KalshiWebSocket` instead of `_DummyWS`. |
| `cli.py` | New flags on `agent start`: `--executor paper|live`, `--whitelist atp-wta-main`, `--require-confirm`. Default `--executor paper --require-confirm`. |

## What v2 rewrites

| File | Current (v1) | New (v2) |
|---|---|---|
| `agent/loop.py` | Tails `market_ticks` DB, computes edge locally, enqueues. | Subscribes to `MonitorBridge` signals, filters, enqueues. No DB tail. Executor call added after decision gate. |
| `agent/runtime.py` | `model_prob_fn` + `context_builder` for v1 prompt. | Keep `MarketCache` + `parse_market_title`. `context_builder` gets a thinner shape — just identity info; Gemini fills in live context itself. Drop `model_prob_fn` sync cache (Monitor owns the model call). |

## What v2 adds

| File | Purpose |
|---|---|
| `agent/monitor_bridge.py` | Runs `EVScanner` in a periodic loop (same as Monitor TUI but no Rich output). Emits `MonitorSignal` objects via async callback to AgentLoop. Owns Kalshi REST polling loop. |
| `agent/executor.py` | Unified `Executor` with two concrete impls: `PaperExecutor` (wraps `PaperTradingEngine`) and `LiveExecutor` (wraps `KalshiClient`). Shared `ConfirmGate` that prompts user via TUI + 30s timeout. Both emit `ExecutionResult`. |
| `tests/test_agent_monitor_bridge.py` | Fake scanner + fake Kalshi, assert signals emitted correctly with whitelist + threshold. |
| `tests/test_agent_executor.py` | Paper round-trip, confirm-timeout defaults-N, manual-confirm approves. |
| `tests/test_agent_grounded_llm.py` | Fake Gemini grounded response, verify tool_config wiring (no real network). |

---

## Data flow for one signal, end-to-end

```
T=0.0s   Monitor scan tick:
         EVScanner.analyze_market_pair("KXATPMATCH-26MAY02HOLMBRO-HOL")
           → Opportunity(edge=0.38, model_prob=0.53, market=15c, ...)
         bridge.emit(signal) — 0ms

T=0.0s   AgentLoop receives signal:
         filter: ATP Main ✓, price 15c in band ✓, no cooldown ✓
         queue.put(candidate) — start cooldown 300s

T=0.1s   Worker dequeues:
         context_builder → {ticker, player_yes, player_no, tournament, surface, round}
         (no local form/H2H anymore — Gemini fetches)

T=0.1s   GeminiGroundedProvider.analyze:
           count_tokens → est_cost $0.03
           budget.reserve("gemini-3.1-pro-preview", 0.03)
           generate_content with google_search tool enabled
           system prompt: "Look up live state for this match. Return real P(YES)."

T=15.8s  Gemini returns:
         {edge_estimate: 0.58, recommendation: BUY_YES, confidence: high,
          reasoning: "Broady up 6-3 set 1 but ankle taping, Holmgren 22-15 H2H on clay..."}
         budget.record(actual_cost $0.04)
         safety.record_llm_success()

T=15.8s  Gate:
         gemini_edge = 0.58 - 0.15 = 0.43 ≥ 0.10 ✓
         confidence "high" ≠ low ✓
         re-read tick: market now 16c → edge_live = 0.42 ≥ 0.08 ✓ (hard check)

T=15.9s  Kelly sizing:
         kelly = 0.43/0.84 × 0.25 × 1.0 × 5999 = $768 → capped $50
         BetDecision(ticker, "yes", kelly=0.008, bet_amount=$50, num_contracts=333)

T=16.0s  risk.check_and_reserve(decision) → (True, "OK")
         exposure now $50

T=16.0s  ConfirmGate.prompt():
         ┌─ AGENT RECOMMENDATION ──────────────────────────┐
         │ KXATPMATCH-26MAY02HOLMBRO-HOL                    │
         │ Holmgren vs Broady, R32, Clay                    │
         │ Market: 15c  Gemini: 58% high  Edge: +43%        │
         │ Suggested: BUY YES $50 (333 contracts)           │
         │                                                  │
         │ Gemini reasoning:                                │
         │ Broady up 6-3 set 1 but ankle taping...          │
         │                                                  │
         │ [Y] Execute   [N] Reject   auto-N in 30s         │
         └──────────────────────────────────────────────────┘

T=18s    User presses Y.

T=18.1s  PaperExecutor.place_order(order_request)
         → fills 333 contracts at 15c
         → paper balance -$50
         → returns OrderResponse(order_id="paper-abc123")

T=18.1s  decisions.append(AgentDecision(
             ticker, model_pre_match=0.53, market_yes_cents=15,
             edge_at_decision=0.38, edge_at_execution=0.42,
             analysis=ev_analysis, llm_provider="gemini-3.1-pro-preview-grounded",
             mode="auto", executed=True, order_id="paper-abc123",
         ))

T=+3hr   SettlementPoller: Kalshi shows market resolved YES
         → append SettlementRecord(decision_id, outcome=won, realized_pnl=+$283.05)
```

---

## Coding order (tomorrow)

Each step: code + tests + commit + push. Same author env as always.

1. **GeminiGroundedProvider** (extend `agent/llm.py`)
   - New class with `tools=[types.Tool(google_search=types.GoogleSearch())]` config
   - New `PROMPT_TEMPLATE_GROUNDED_V1` template
   - Tests: fake the Gemini call with a stubbed response; verify tool config wire-up, verify prompt includes required fields
   - Real smoke test: one actual grounded call to Gemini, confirm returns valid EvAnalysis
   - Est: 1.5h code + test, plus $0.05 smoke test

2. **MonitorBridge** (new `agent/monitor_bridge.py`)
   - Runs `EVScanner` against `KalshiClient.get_markets` + orderbook in a 2-5s polling loop
   - Emits `MonitorSignal(ticker, prematch_edge, market_yes_cents, ...)` via async callback
   - Filter flags: `series_whitelist: list[str]`, `min_prematch_ev: float`, `price_band: (int, int)`
   - Tests: fake scanner + fake callback, assert whitelist filtering, EV threshold, price band
   - Est: 1.5h

3. **AgentLoop rewrite** (`agent/loop.py`)
   - Drop `MarketTickReader` and `_maybe_enqueue` (DB-tail logic)
   - Add `async def on_signal(sig: MonitorSignal)` as the pub/sub endpoint
   - Same per-ticker cooldown, queue cap, freshness gate
   - Worker path unchanged in shape, but calls `executor.place_order` after decision gate
   - Post-LLM edge re-check becomes HARD (reject, not soft log)
   - Est: 2h

4. **Executor** (new `agent/executor.py`)
   - `ConfirmGate` with 30s timeout, default N
   - `PaperExecutor(paper_engine)` / `LiveExecutor(kalshi_client)` both implement common interface
   - Mode flag from config picks one
   - Tests: paper round-trip, confirm timeout defaults N, manual Y succeeds
   - Est: 1.5h

5. **Wire in CLI** (`cli.py`)
   - New flags `--executor paper|live --require-confirm/--no-confirm --whitelist atp-wta-main`
   - Replace `_DummyWS` / `_NullRisk` with real instances (RiskManager from config, WS still not owned by agent)
   - Default: `paper + require-confirm + atp-wta-main` so starting without flags is the safe path
   - Est: 1h

6. **First real paper run**
   - Set `gemini-budget $10` for the smoke session
   - Let it run for 1-2 hours, pause, read all decisions, adjust thresholds, rerun
   - Est: monitoring time, not dev time

Total dev: **~7.5 hours**. Fits in one focused day.

---

## Test strategy

- **Unit**: every new module (`grounded_llm`, `monitor_bridge`, `executor`) has its own test file, fakes for external deps (no real Kalshi, no real Gemini network).
- **Integration test**: one test wires `FakeScanner → MonitorBridge → AgentLoop → FakeLLM → PaperExecutor → DecisionLog` and asserts a full successful decision flow.
- **Real smoke**: one manual command `tennis-edge agent start --executor paper --gemini-budget 5 --require-confirm` for ~30 minutes to verify live behavior. Not in pytest.
- **Regression**: the 158 v1 tests stay green; anything that breaks is a bug in the migration, fix it.

Target: **~200 tests total passing** after v2 lands.

---

## Explicit NON-goals (keep scope disciplined)

- ❌ Challenger markets in Week 1
- ❌ Screenshot pipeline (deferred indefinitely given grounding works)
- ❌ A/B test v1 (ungrounded) vs v2 (grounded) — v1 known limited, not worth the 2x cost
- ❌ Agent-owned WebSocket connection (tick-logger still owns it; agent reads DB for post-LLM re-check only)
- ❌ Monitor TUI integration — Agent runs its own embedded scanner without the Rich display; existing `tennis-edge monitor` command is unchanged for interactive use
- ❌ Multi-provider LLM A/B (Gemini vs Claude) — Week 2 or 3 if budget and value allow
- ❌ WTA-specific rating model — Phase 1 deferred item, not relevant here
- ❌ Python 3.14 editable-install fix — cosmetic, use `PYTHONPATH=src` in the meantime

---

## Safety stack summary (what protects real money)

At launch, with `--executor live`:

1. **$50 per-market hard cap** (RiskManager + Kelly cap)
2. **$500 total exposure cap** (RiskManager)
3. **-$200 daily loss → agent kill** (RiskManager + SafetyMonitor)
4. **Confidence = low → no trade** (execution gate)
5. **Gemini edge < 0.10 → no trade** (execution gate)
6. **Post-LLM price re-check < 0.08 → hard reject** (execution gate)
7. **30s manual confirm prompt** (ConfirmGate; N default)
8. **3 consecutive LLM failures → kill** (SafetyMonitor)
9. **Budget cap hit → kill** (SafetyMonitor)
10. **Per-ticker 5-minute cooldown** (loop)
11. **`data/agent_control/flatten` file → close positions + exit** (SafetyMonitor)
12. **Per-run idempotency on orders** (client_order_id UUID4 — add to OrderRequest this lane)

12 distinct layers between "Gemini says X" and "money moves". Any one triggering stops the trade.

---

## Open follow-ups (after v2 ships, before Week 2)

1. **Measure Gemini grounding cost empirically**. Current estimate $0.03-0.05/call; need real data to tune budget.
2. **Decide on Challenger coverage**. Paper-run a small number of Challenger signals alongside Main; compare Gemini's ability to find live info.
3. **ConfirmGate UX**: is 30s TUI prompt the right surface, or should it be a macOS notification / push? (Push = don't need to be in front of terminal.)
4. **Auto-switch paper → live threshold**: hand-pick 20 now, but long-term rule should be deterministic (e.g. "if paper P&L positive over 20 trades").
5. **`client_order_id` on `OrderRequest`** — add UUID4 during executor implementation so live retries are idempotent.

---

## Known risks to watch during first paper run

1. **Gemini grounding may not find Challenger info** — already mitigated by Main-only whitelist Week 1.
2. **Grounded calls much slower than ungrounded** (15-45s vs 5-15s) — cooldown + queue cap handle this but a burst on a volatile market could fall behind.
3. **Gemini hallucinated news** — "I read that X is injured" when no source. Mitigation: Gemini response should include sources (Gemini grounding returns citations); log them; post-hoc audit.
4. **Monitor scanner cost** — REST polling every 2-5s across all ATP/WTA Main markets. Rate-limited at 7/s. Should be fine for Main only (~30 markets); Challenger would stress this.
5. **Paper engine's price assumption** — fills at exactly market_yes_cents. Real Kalshi fills could slip 1-3 cents. Paper P&L will be **optimistic** vs live. Flag this when comparing.

---

## Review + eng review trigger

After v2 code lands (~tomorrow night), recommend re-running `/plan-eng-review` on this exact document to catch anything missed. The plan is detailed enough that the review can focus on **gaps** rather than re-deriving architecture from scratch.
