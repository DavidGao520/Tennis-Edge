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

## Locked decisions (from 2026-04-19 chat + eng review)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Monitor → Agent = same process, pub/sub callback** | Simplest; Agent owns its own scanner instance with TUI disabled. No IPC, no DB queue table. |
| 2 | **Gemini Google Search grounding replaces screenshots** | ESPN / Flashscore / tennisabstract all live on the web. Gemini's search is more reliable than our headless Chromium would be. Screenshots deferred indefinitely. |
| 3 | **Hard stops: $50/market, $500 total exposure, -$200 daily loss** | Matches original Phase 3C plan. RiskManager already enforces these; wire them in for real this time. |
| 4 | **No manual confirm gate. Paper for 50 trades OR until counterfactual P&L positive over ≥10 settled markets (whichever later), then user manually edits config to `--executor live`** | Original plan's 30s TUI prompt theatrical in detached tmux: stdin gone, every prompt auto-rejects, cohort never reaches 20. Single explicit human decision point at the paper→live boundary instead. |
| 5 | **Week 1 whitelist: `KXATPMATCH` + `KXWTAMATCH` only (no Challengers)** | Gemini's Google Search coverage of Challenger matches is untested. Validate the grounded pipeline on markets where ESPN/Flashscore actually have data. Challenger added Week 2. |
| 6 | **No new Executor ABC / PaperExecutor / LiveExecutor / ConfirmGate classes** | Existing `ExchangeClient` ABC already abstracts paper vs live (both `PaperTradingEngine` and `KalshiClient` implement it). AgentLoop takes an `ExchangeClient` directly; CLI flag picks which. -4 classes vs naive plan. |
| 7 | **Delete v1 prompt + ungrounded path immediately after first clean v2 paper run** | v1 known limited; git history preserves it. No `--llm-variant` flag, no rotting code. |
| 8 | **First action in coding session: 5-minute real grounded smoke call** | Cost/latency assumptions ($0.03-0.05 / 15-45s) are guesses. If reality is 3x higher, budget + cooldown defaults need to change before we code 4 hours against wrong numbers. |

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
| `tests/test_agent_monitor_bridge.py` | Fake scanner + fake Kalshi, assert signals emitted correctly with whitelist + threshold. |
| `tests/test_agent_grounded_llm.py` | Fake Gemini grounded response, verify tool_config wiring (no real network). |
| `tests/test_agent_gemini_eval.py` | **Required eval suite**. 5 hand-picked historical Kalshi tennis markets with known outcomes. Run grounded provider, assert recommendation matches outcome ≥60% and `edge_estimate` within 0.20 of settlement. ~$0.25/run. Run before merging any prompt template change. |

**Note**: no separate `executor.py` module. `AgentLoop.__init__(exchange: ExchangeClient, ...)` takes either a `PaperTradingEngine` or `KalshiClient` directly. `cli.py agent start --executor paper|live` picks one. The `ExchangeClient` ABC in `exchange/base.py` is the only abstraction needed.

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

**Step 0 (5 minutes, do FIRST)**: Real grounded smoke call. Don't write any v2 file yet.

```bash
# From the project root, in the venv
PYTHONPATH=src python -c "
import os
for line in open('.env'):
    if line.startswith('TENNIS_EDGE_GEMINI_KEY='):
        os.environ['GEMINI_API_KEY'] = line.split('=',1)[1].strip()
from google import genai
from google.genai import types
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    response_mime_type='application/json',
)
prompt = '''Look up the live state of the Madrid Open 2026 ATP tournament right now.
Pick any active match. Return JSON: {tournament, match, current_state, source_url}.'''
resp = client.models.generate_content(
    model='gemini-3.1-pro-preview', contents=prompt, config=config,
)
print('TEXT:', resp.text)
print('USAGE:', resp.usage_metadata)
"
```

If the per-call cost is within 2x of the $0.03-0.05 estimate, proceed to Step 1. If it's 3-5x higher, **stop** and recalibrate the budget / cooldown / min-edge before writing code.

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

4. **Executor wiring** (no new file)
   - Add `AgentLoop.__init__(exchange: ExchangeClient, ...)` parameter.
   - Worker calls `await self.exchange.place_order(OrderRequest(client_order_id=decision_id, ...))` after the decision gate passes.
   - On `place_order` exception: log, call `risk.release(decision)`, append decision with `executed=False, reject_reason="order_failed"`. Do NOT trip kill switch on first failure (network flakes happen); after 3 consecutive failures, treat like LLM failures and let the existing 3x kill mechanism cover it (or add an explicit `OrderManager` failure counter — pick during impl).
   - **No ConfirmGate.** Decision gate ends at `risk.check_and_reserve` → straight to `place_order`. Paper mode is the safety; live mode is opt-in via CLI flag.
   - Tests: paper round-trip, place_order success path, place_order exception → release + log, idempotent retry on transient error
   - Est: 1h

5. **`OrderRequest.client_order_id`** (`exchange/schemas.py` + `exchange/client.py`)
   - Add `client_order_id: str | None = None` to `OrderRequest`.
   - Thread through `KalshiClient.place_order` to the actual Kalshi header (verify exact header name in their docs first; common pattern is `Idempotency-Key` or `client_order_id` body field).
   - `AgentLoop` sets `client_order_id = decision.decision_id` so retries hit the same UUID.
   - Tests: round-trip via JSON, retry-with-same-id is a no-op for Kalshi (test against `respx` mock, not real API).
   - Est: 1h

6. **Wire in CLI** (`cli.py`)
   - New flags `--executor paper|live --whitelist atp-wta-main`
   - Replace `_NullRisk` with real `RiskManager` from config; **keep `_DummyWS`** because Agent has no WS in v2 (or remove the WS check from `safety.watchdog_loop` entirely — pick one).
   - Default: `paper + atp-wta-main` so starting without flags is the safe path. Live requires explicit `--executor live`.
   - Est: 0.5h

7. **SettlementPoller wires daily P&L into RiskManager**
   - `SettlementPoller.poll_once`, after appending each `SettlementRecord`, calls `await risk.record_settlement(ticker, realized_pnl)`.
   - Without this, `risk.state.daily_pnl` stays 0 and the `-$200/day → KILL` switch is dormant. Same hidden bug as v1.
   - Tests: integration test driving `poll_once` against a fake exchange + verifying `risk.state.daily_pnl` reflects N losing settlements; integration test that 3 losing settlements → `daily_pnl ≤ -200` → `safety.check_daily_pnl` trips KILLED.
   - Est: 0.5h

8. **Required eval suite** (`tests/test_agent_gemini_eval.py`)
   - 5 hand-picked historical Kalshi tennis markets with known outcomes (3 winners, 2 losers, ATP+WTA Main).
   - For each, run `GeminiGroundedProvider.analyze` and assert `recommendation` matches outcome ≥60% of the time, `edge_estimate` within 0.20 of settlement.
   - Costs ~$0.25/run. Mark `@pytest.mark.eval` so default `pytest tests/` skips it; explicit `pytest -m eval` runs it.
   - Required before merging any prompt template change. If pass rate drops below baseline, revert.
   - Est: 1h to set up + ~$0.25 per run

9. **First real paper run**
   - Set `gemini-budget $10` for the smoke session
   - Let it run for 1-2 hours, pause, read all decisions, adjust thresholds, rerun
   - Est: monitoring time, not dev time

Total dev: **~7 hours**. Fits in one focused day. The scope reduction (drop Executor classes + ConfirmGate) saves ~1.5h and a stack of tests.

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

---

## Critical test paths (from eng review, MUST be in same PR as code)

These are the paths that, without explicit test coverage, will silently fail in production:

| # | Path | Why it matters |
|---|---|---|
| 1 | `SettlementPoller → risk.record_settlement → safety.check_daily_pnl → KILLED` | Without this integration test, the $200/day kill switch is dormant. Bit v1 silently. |
| 2 | `place_order` succeeds Kalshi-side, client times out, retry → no double position | Validates `client_order_id` idempotency. Without it, transient network blips → double exposure. |
| 3 | Gemini grounded edge < 0.10 → reject (no order placed) | New CRITICAL hard gate vs v1's soft log. |
| 4 | Confidence == "low" → reject regardless of edge | New CRITICAL hard gate. |
| 5 | Post-LLM edge re-check < 0.08 → HARD reject (decision logged, no order placed) | Tightens v1's soft "log + reject_reason" to a hard block. Matters when LLM took 30s and market moved. |
| 6 | `place_order` raises → `risk.release(decision)` called → retry safe | Without `release`, exposure ledger drifts and eventually saturates the cap. |
| 7 | Detached-tmux startup → daemon runs without TUI input dependence | Validates no stdin reads in the hot path; covers the "no manual confirm" decision. |

Plus regression tests for: cooldown enforcement, queue cap, missing API key, scanner exception in `MonitorBridge`, malformed Gemini JSON.

## Failure modes and mitigations

| # | Scenario | Test? | Error path? | Visible? | Critical |
|---|---|---|---|---|---|
| 1 | Gemini grounded returns malformed JSON | Required (#3 above) | `LLMOutputError` caught, counts toward 3x kill | Yes (log) | YES |
| 2 | `place_order` succeeds but client times out → retry double | Required (#2 above) | `client_order_id` makes retry idempotent | Silent without test | YES |
| 3 | SettlementPoller skips `risk.record_settlement` | Required (#1 above) | New test forces wire-up | Silent | YES |
| 4 | Gemini grounded cost is $0.10/call, not $0.03 | Smoke before code | `BudgetTracker` logs cumulative cost; cap trips early | Visible | Medium |
| 5 | EVScanner raises in `MonitorBridge` tick | Required (regression) | try/except in scan loop, log + continue | Visible (log) | Medium |
| 6 | Tick-logger stops, agent keeps trading on stale prices | Already covered (Lane 1D `TICK_LOGGER_STALE` switch) | Watchdog at 60s | Visible (kill) | Low |

## Worktree parallelization

Independent enough to split. Module-level dependency:

| Step | Modules | Depends on |
|------|---------|------------|
| A. `GeminiGroundedProvider` + eval suite | `agent/llm.py`, `tests/` | Step 0 smoke result |
| B. `MonitorBridge` | `agent/monitor_bridge.py`, `scanner.py` (read), `tests/` | — |
| C. `client_order_id` plumbing | `exchange/schemas.py`, `exchange/client.py`, `tests/` | — |
| D. `AgentLoop` rewrite (drop DB tail, add signal subscribe + executor wire) | `agent/loop.py`, `tests/` | A, B, C |
| E. SettlementPoller → risk wire | `agent/settlement.py`, `strategy/risk.py` (read), `tests/` | — |
| F. CLI flags + paper/live default | `cli.py` | D |

**Lanes**:
- Lane 1: A → D → F (sequential — agent spine)
- Lane 2: B (independent until D)
- Lane 3: C (independent until D)
- Lane 4: E (fully independent)

Launch B + C + E in parallel, then A, then merge into D, then F. Keeps you unblocked even when A is mid-eval-suite.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_RESOLVED | 10 issues, scope reduced (-4 classes), 4 forks resolved, 7 critical test paths mandated |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — (no UI in v2) | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**VERDICT**: ENG REVIEW COMPLETE — 4 forks decided (drop Executor classes, no manual-confirm gate, delete v1 post-paper, smoke first). Scope reduced from 7 new classes to 3. Critical test paths enumerated. Ready to implement starting with Step 0 smoke.
