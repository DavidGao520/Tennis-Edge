# Tennis-Edge Progress Log

## What is Tennis-Edge?

A quantitative tennis prediction market trading bot for [Kalshi](https://kalshi.com). It combines player rating models, real-time market data via WebSocket, and Kelly criterion position sizing to identify and execute trading opportunities in ATP/WTA tennis markets.

**Owner:** David Gao ([@DavidGao520](https://github.com/DavidGao520)) — CS student at UW, active Kalshi tennis trader with ~$8,400 net profit across 275 trades.

---

## Current Architecture

```
Sackmann (2000-2024) + TennisMyLife (2025-2026) → 78,762 matches
        ↓
    Glicko-2 Rating System → 2,775 players rated
        ↓
    36-Feature Logistic Regression Model
        ↓
    Pre-match Win Probability (anchor)
        ↓
Kalshi WebSocket (real-time bid/ask) → Market Implied Probability
        ↓
    Edge = Pre-match − Market
        ↓
    Kelly Criterion → Position Sizing
        ↓
    Live Terminal Dashboard (monitor command)
```

---

## Timeline & Key Decisions

### Day 1: Foundation

**Data Pipeline**
- Ingested Jeff Sackmann's `tennis_atp` GitHub data: 74,906 matches (2000-2024), 65,989 players, 1.34M rankings
- SQLite database with WAL mode, proper indexes
- CSV download with local caching to avoid re-downloading

**Glicko-2 Rating System**
- Implemented full Glicko-2 algorithm per Glickman's specification
- Verified against known test cases from the paper
- 30-day rating periods, processes match history chronologically
- Had to add overflow protection for extreme rating differences (new/unrated players)
- 309 rating periods computed, 2,775 unique players rated

**Feature Engineering — 36 features in 6 categories:**
- Glicko-2 (6): mu, phi for both players + diffs
- Surface (6): one-hot encoding + surface-specific win rates
- Fatigue (10): days since last match, matches in 7/14/30 day windows
- H2H (3): head-to-head wins and win rate
- Form (8): win rate over last 5/10/20 matches
- Tournament (3): level encoding, round depth, best-of-5 flag
- **Critical design:** player ordering canonicalized by ranking (p1 = higher-ranked) to prevent label leakage

**Decision: Python over TypeScript/Rust** — the edge comes from model quality and data analysis, not execution speed. Python has the best ecosystem for statistical modeling.

### Day 1-2: Model & Strategy

**Initial Model (v1)**
- Logistic Regression with StandardScaler pipeline
- Simple train/test split: 2005-2019 / 2020-2026
- No validation set, no hyperparameter tuning, C=1.0 hardcoded
- Feature importance: raw coefficient absolute values
- Result: 65.5% accuracy, Brier 0.215

**Improved Training Pipeline (v2)**
- Proper 3-way temporal split: Train (2005-2020) / Val (2021-2023) / Test (2024-2026)
- Walk-forward CV with 5 folds, per-fold hyperparameter tuning
- Grid search over C values [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
- **Best C = 0.01** (not the default 1.0 — more regularization is better)
- Permutation importance instead of raw coefficients
- Result: 64.8% accuracy on test, Brier 0.217, ECE 0.023

**Key finding:** Walk-forward CV showed accuracy drops from 70% (older data) to 65.6% (recent data). Tennis has become harder to predict — more competitive, more upsets.

**Strategy Layer**
- Full & fractional Kelly criterion (default 25% Kelly)
- Position sizing with constraints: max 5% bankroll per bet, min 3% edge threshold
- Risk management: per-market position limits, total exposure cap, daily loss kill-switch

### Day 2: Data Gap & TennisMyLife

**Problem:** Sackmann's repo only has data through 2024. No 2025-2026 match files.

**Solution:** Integrated [TennisMyLife](https://stats.tennismylife.org) as second data source.
- Free CSVs with daily updates, covers 2025-2026
- Same column schema as Sackmann but different player IDs (alphanumeric vs numeric)
- Built name-based player matching with fuzzy fallback
- Verified: 86.7% exact match overlap with Sackmann 2024 data on same matches
- **Result: 78,762 total matches through April 2026**

**Decision: TennisMyLife over API-Tennis/Sofascore** — free, CSV format (no API key needed), same schema as Sackmann, daily updates.

### Day 2: Kalshi Integration

**API Client**
- Async REST client using httpx with retry logic and rate limiting
- RSA-PSS request signing (SHA256) — matched user's existing export script's signing method (PSS.DIGEST_LENGTH, not MAX_LENGTH)
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`
- Paper trading engine implementing same ExchangeClient ABC

**Trading History Analysis**
- Imported user's Kalshi export (867 fills, 324 orders, 275 settlements)
- **Critical P&L bug fix:** initial calculation showed -$46K (wrong). Correct calculation:
  - When `action=sell, side=no`: you're selling a YES position, revenue = `yes_price` per contract (not `no_price`)
  - Must compute P&L from fill-level cash flows + settlement payouts, not from settlement fields directly
  - **Correct result: +$8,397 net P&L** (matches user's bank records within 3%)

**User's actual trading performance:**
| Category | P&L | Trades | Win% |
|----------|-----|--------|------|
| ATP Challenger | +$4,891 | 68 | 49% |
| ATP Main | +$3,846 | 49 | 51% |
| WTA Main | +$1,711 | 29 | 55% |
| NCAA | +$1,076 | 20 | 45% |
| NBA | +$947 | 13 | 77% |
| WTA Challenger | -$1,365 | 24 | 50% |
| Crypto | -$2,417 | 37 | 46% |

**Model vs User's Bets (33 matched ATP Main trades):**
- Model agreed with 61% of bets
- When model agrees: 55% win rate, +$3,459 P&L
- When model disagrees: 54% win rate, +$1,212 P&L
- **Conclusion:** User is profitable regardless of model agreement — user's edge comes from situational reads (motivation, matchup style, live game state) that the model can't capture. Model is useful as an anchor, not as the sole decision-maker.

**First live bet placed via API:** Roncadelli YES @ 63c (1 contract) — executed successfully.

### Day 2-3: In-Play Model

**Built mathematical in-play win probability model:**
- Hierarchical recursion: Points → Games → Sets → Match
- Input: each player's serve point win probability (ATP avg ~64%)
- Serve probs estimated from Glicko-2 rating gap
- Verified sanity: equal players 0-0 = 50%, up 1-0 sets = 78%, down 0-1 = 28%

**Live Score Sources:**
- ESPN API (free, no key): works for ATP/WTA main tour but not Challengers
- Sofascore API: blocked (403 — requires browser session)

**Decision to remove Live column from monitor:** Market price itself IS the best live probability — it already incorporates all live score information from thousands of traders. The valuable signal is how far the market has drifted from our pre-match anchor, not another model's live estimate.

### Day 3: WebSocket Real-Time Monitor

**Kalshi WebSocket Integration**
- URL: `wss://api.elections.kalshi.com/trade-api/ws/v2`
- Same RSA-PSS auth as REST, signing path = `/trade-api/ws/v2`
- Subscribed channels: `ticker` (all markets), `fill` (own executions)
- Auto-reconnect with exponential backoff

**Bug fix:** WebSocket ticker fields are `yes_bid_dollars` (string like `"0.5200"`), not `yes_bid` (int). Had to parse dollar strings to cents.

**Live Monitor Dashboard (`tennis-edge monitor`)**
- Real-time terminal dashboard using Rich library
- Columns: Signal, Player, Bid, Ask, Market, Pre-match, Edge, EV/$, Side, Kelly$, Vol
- Edge = Pre-match (Glicko-2 anchor) − Market (Kalshi live price)
- STRONG/MODERATE/WEAK signal classification by edge magnitude
- Alert system for new STRONG signals
- Tested: 500+ updates received in 20 seconds, 35+ active tennis markets tracked simultaneously

**Bug fix:** Model probability direction was inverted for some markets — `serve_prob_from_glicko(player, opponent)` already returns P(player wins), no additional flip needed.

---

## Current State (as of April 14, 2026)

### What Works
- ✅ 78,762 matches ingested (2000-2026)
- ✅ Glicko-2 ratings for 2,775 players
- ✅ 36-feature logistic regression with proper train/val/test pipeline
- ✅ Kalshi API: market scanning, order placement, position management
- ✅ WebSocket real-time price streaming (ticker + fill channels)
- ✅ Live terminal dashboard with EV signals
- ✅ Trading history import with correct P&L analysis
- ✅ Paper trading engine
- ✅ 20 tests passing
- ✅ CLI with 9 commands: `ingest`, `ratings`, `train`, `monitor`, `live`, `opportunities`, `history`, `player`, `scan`

### Known Limitations
1. **Model ceiling:** Logistic regression at 64.8% accuracy — XGBoost/LightGBM should capture non-linear feature interactions
2. **Pre-match only:** Monitor shows pre-match anchor probability, not live in-play probability. Market price is used as the live signal.
3. **Challenger coverage:** ESPN doesn't cover ATP/WTA Challenger events for live scores. Need paid API (api-tennis.com) for point-level challenger data.
4. **No automated trading loop:** Monitor shows signals but doesn't auto-execute. User makes final decision.
5. **WTA model:** Currently only trained on ATP data. WTA needs separate Glicko-2 ratings and model.

### File Structure (56 files, ~7,300 lines)

```
src/tennis_edge/
├── cli.py              # 9 CLI commands
├── config.py           # YAML config + env var overrides
├── realtime.py         # WebSocket monitor dashboard
├── scanner.py          # Pre-match EV scanner (REST)
├── scanner_live.py     # In-play EV scanner (live scores)
├── data/               # SQLite DB, ingestion (Sackmann + TennisMyLife), trading history
├── ratings/            # Glicko-2 algorithm + tracker
├── features/           # 36 features across 6 categories
├── model/              # Logistic regression, training pipeline, in-play model, calibration
├── strategy/           # Kelly criterion, position sizing, risk management
├── exchange/           # Kalshi REST client, WebSocket client, paper trading, auth
└── utils/              # Logging, time helpers
```

---

## Next Steps

1. **XGBoost/LightGBM model** — capture non-linear feature interactions (surface × handedness × fatigue)
2. **Auto-trading loop** — when edge > threshold for N seconds, auto-place order
3. **WTA-specific model** — separate ratings and training for WTA tour
4. **Serve stats features** — ace rate, first serve %, break point conversion from match stats
5. **Challenger live scores** — integrate paid API for point-level data on challenger matches
6. **Position tracking dashboard** — show open positions, unrealized P&L alongside monitor
7. **Backtest against actual Kalshi historical odds** — currently backtesting uses synthetic odds, need real historical market prices

---

# Phase 2: Three-Option Trader Assistant (started April 17, 2026)

## What Changed The Plan

A live demo on April 17 surfaced the real ceiling of phase 1: monitor showed Holmgren 53% pre-match while Kalshi market priced him at 15% live (he had lost first set 6-3 and was down 1-0). Bot has no live score awareness and over-weights player rating difference. **Bot is not trustworthy enough to bet on.**

The honest truth: David's $10K/month manual edge on Kalshi tennis comes from screenshot-driven analysis (with Gemini 3.1 Pro doing structured EV reasoning over live score + market price), not from the trained model. Tennis-Edge's job in phase 2 is to **make that workflow faster and more reliable**, not to replace David's judgment.

## Phase 2 Plan: Three-Option Trader Assistant (6 weeks)

Single launch screen, three options, three workstreams (one per teammate):

```
$ tennis-edge

  Tennis-Edge — Kalshi Tennis Trading Assistant

  [1] Arbitrage    — Cross-market price discrepancy scanner (real-time auto-orders)
  [2] Monitor      — Pre-match anchor + Kalshi live prices + LLM research (manual trade)
  [3] Agent        — Bot scans → LLM analyzes → recommend or auto-execute (with rails)
```

| Owner | Workstream | Status |
|---|---|---|
| **David** | LLM Research Enrichment + Agent (Option 2 R-key + Option 3) | In progress |
| **Anthony** | CLI / UI polish + Real Backtest infrastructure | In progress |
| **Billy** | Arbitrage scanner (with feasibility gate Week 1) | Feasibility study pending |

Full plan in repo at `docs/PROGRESS.md` (this file). Detailed planning artifact in private plan file. Key design decisions:

- **Agent has 3 sequential safety phases**: shadow-trade (log only) → human-in-loop (Y/N/R prompt) → restricted auto-execute (only if shadow phase shows agent matches/beats David's manual trades)
- **Hard kill switches always on**: `Q` pauses agent, `K` flattens all agent positions
- **LLM choice**: Gemini 3.1 Pro (validated by David) + Claude Opus 4.6 (A/B test for deep dives)
- **Real backtest** replaces synthetic odds with WebSocket-collected historical Kalshi prices
- **Arbitrage path has feasibility gate**: if no actionable signals in week 1 study, Billy pivots to backtest support

## Phase 2 Progress Log

### April 19, 2026 — Phase 3 v2.1: Aggressive Research Prompt + system_instruction + key_risk

After hand-running the agent on MacBook for the demo prep, the prompt was upgraded based on feedback from David's manual workflow ("my Gemini analyst auto-researches rankings, H2H, surface form, fatigue, injuries — make our agent prompt push for the same"):

**Three changes**:

1. **Persona moved to `system_instruction`**. Previously the "you are a quant analyst" framing was embedded in the user prompt body. Now it's a separate `SYSTEM_INSTRUCTION_GROUNDED_V1` constant passed via `GenerateContentConfig.system_instruction`. Cleaner architecturally — per-request user content can focus on market context, persona doesn't get overwritten by static context. Google's API also caches system instructions more aggressively.

2. **Aggressive research checklist** (9 items, in order):
   1. Live match state (most decisive — settled markets must not be faded)
   2. Current ATP/WTA rankings + trajectory
   3. Career H2H including surface and last-12-month breakdown
   4. Recent form (last 5-10 matches, opponent quality)
   5. Surface specialty / tournament fit
   6. Injury / fitness reports (last 7 days)
   7. Fatigue load (matches in last 7 days, late finishes, time zones)
   8. Environmental (altitude, court speed, weather)
   9. Off-court alpha (equipment, coaching, motivation)

   Previous prompt had 4 generic items; new one matches what an actual hedge-fund analyst's checklist looks like.

3. **`EvAnalysis.key_risk: str | None`** — new optional field for the single biggest risk to the call. Lets post-mortem analytics group losing decisions by risk class ("which kind of risk is biting us most — injuries, fatigue, altitude, or live-state uncertainty?"). Backward compatible with v2.0 logged decisions.

**Honest pushback on the analyst's other suggestions**:
- Live performance metrics (1st serve %, rally length, unforced errors) — **rejected**, we don't have them. ESPN free covers Main only; Sofascore blocked; paid feeds out of budget. The whole reason v2 uses grounded search is precisely because we don't have streaming data.
- Decimal odds format ("1.65x") — **rejected**, Kalshi trades binary 0-100¢ contracts.
- Stop-loss triggers / hedge recommendations — **rejected**, position management is Phase 3D scope, not v2.
- "High-frequency EV correction" — **rejected**, every grounded call is ~$0.015. 1-min cooldown vs 5-min cooldown = 14× cost increase for noise-level price changes.

**Eval re-run after upgrade**: 5/5 pass. Reasoning quality measurably better — `settled_market_no_fade` now cites specific sources ("beIN Sports, Olympics.com, Tennis Now"), `prematch_strong_favorite` reasons more transparently about static-vs-live tradeoff. **Critically: the v1 failure-mode regression test (`extreme_low_price`) still SKIPs correctly with edge_est=0.03 matching market. The v2 thesis holds.**

Cost per call ~$0.015-0.025 (slight increase from more aggressive search; still well under budget).

Default suite: **227 passed** (225 prior + 2 new system_instruction tests).
Eval suite: **5/5 passed @ ~$0.10** (slightly higher than v2.0 due to more search hops).

### April 19, 2026 — Phase 3 v2 Refactor: Single-Process Agent (No Tick-Logger Dependency)

**Why**: when prepping a MacBook demo for teammates, the original v2 architecture's split between `tick-logger` (writes `market_ticks` SQLite table) and `agent` (reads it for post-LLM edge re-check) made for awkward UX — two processes to start, ~60s wait for fresh ticks before the agent could trade. For both demo and operational simplicity, the agent should be a single self-contained process.

**Architectural decision change**: this overrides the original eng-review Q1 fork ("Tick source = DB tail from market_ticks"). New design:

- **AgentLoop's post-LLM price re-check now pulls from a callable** (`PriceSource = Callable[[str], tuple[int, float] | None]`), returning `(yes_cents, age_seconds)` for the most recent observation.
- **Production wiring**: `price_source = bridge.latest_price` — `MonitorBridge` already polls Kalshi REST every 15s for opportunity scanning, so it has fresh per-ticker prices already; just expose them.
- **Tick-logger stays useful**: still runs on Mac mini, still feeds `market_ticks` for Anthony's backtest engine. Decoupled from agent runtime.

**Code changes**:
- `agent/monitor_bridge.py`: new `_last_prices: dict[str, (cents, monotonic_ts)]` cache populated on every `_analyze_one` call (even for tickers the signal filter rejects — AgentLoop may need them later). New method `latest_price(ticker) -> (cents, age_s) | None`.
- `agent/loop.py`: replaced `tick_db_path: str` constructor param with `price_source: PriceSource`. Deleted `_TickReader` class and its `_TickRow` dataclass. `_post_llm_edge_check` now calls `self.price_source(ticker)` and rejects if age > 75s (5x the bridge's poll_interval, generous safety ceiling).
- `agent/safety.py`: `watchdog_loop` parameter `db_path` is now optional. When None, the `TICK_LOGGER_STALE` check is skipped — appropriate when the agent is self-contained.
- `cli.py`: bridge constructed first with `on_signal=lambda sig: agent_loop.on_signal(sig)` (late-bound closure), then `AgentLoop(price_source=bridge.latest_price, ...)`. `safety.watchdog_loop` called with `db_path=None`.

**Tradeoffs (locked in)**:

| | Before (DB tail) | After (bridge cache) |
|---|---|---|
| Re-check price freshness | ~2s | up to 15s |
| Processes per agent deployment | 2 | 1 |
| Operational overhead | tick-logger must be running | none |
| Backtest data collection | shared with agent | independent on Mac mini |

For a system with 5-minute per-ticker cooldown and 30-second LLM thinks, the 13-second freshness delta is irrelevant. The operational simplification is large.

**Test updates**:
- New `FakePriceSource` class in `test_agent_loop.py` (mimics the `MonitorBridge.latest_price` contract). Tests inject prices via `prices.set(ticker, cents, age_s)` instead of writing SQLite rows.
- 5 old `_TickReader` tests deleted. Replaced with 4 `FakePriceSource` tests verifying the contract.
- 4 new `MonitorBridge.latest_price` tests: returns None for unseen ticker, caches after scan, caches even filter-rejected tickers, age grows with time.
- All 25 AgentLoop behavior tests still pass with the new wiring.

Full suite: **225 passed** (222 prior + 4 new + 4 new − 5 deleted = +3 net), 5 deselected (eval).

**Net**: a bare `tennis-edge agent start` now spins up MonitorBridge + AgentLoop + SettlementPoller + SafetyMonitor in a single process. No prerequisites beyond model artifact + `.env`. The agent is the agent.

### April 19, 2026 — Phase 3 v2 Step 8: Gemini Eval Suite (Required Regression Gate)

**Built**: `tests/test_agent_gemini_eval.py` — required gate before any future change to `PROMPT_TEMPLATE_GROUNDED_V1` lands.

**5 hand-picked scenarios**, each targeting a specific failure mode:

| Case | Setup | Pass criteria | Tests for |
|---|---|---|---|
| `settled_market_no_fade` | Sinner-Norrie at 98c (already won) | rec ∈ {SKIP, BUY_YES}; edge near 0.98 | **The v1 bug, regression** |
| `thin_context_obscure_match` | Fictional names | rec=SKIP, conf ≤ medium | "If search returns nothing, lean SKIP" prompt rule |
| `prematch_strong_favorite_underpriced` | Model 85%, market 60% | rec ∈ {BUY_YES, SKIP} (NOT BUY_NO) | Anchor consistency |
| `extreme_low_price_likely_in_progress` | Model 30%, market 3c | rec ∈ {SKIP, BUY_NO}; edge near 0.03 | **The other v1 bug** |
| `balanced_no_edge` | All static features even | rec=SKIP | Healthy "no trade" behavior |

**Mechanics**:
- All cases marked `pytest.mark.eval`
- `pyproject.toml` registers the marker + `addopts = "-m 'not eval'"` so default `pytest tests/` skips them
- Run explicitly with `pytest -m eval`
- Cost ~$0.10 per full eval run, ~78 seconds wall time
- Pass criteria are LENIENT by design — the goal is regression detection across prompt edits, not a tight benchmark

**Real eval result against Gemini 3.1 Pro Preview**: **5/5 passed.** Most important:

The `extreme_low_price_likely_in_progress` case (market 3c, model 30%) — exactly the situation that caused v1 to recommend BUY_YES on settled markets — Gemini v2 returns:
- `rec=SKIP, confidence=low, edge_est=0.030` (matching market)
- Reasoning: "The massive drop in the underdog's probability from the pre-match model (30%) to the live market price (3%) strongly implies the underdog has lost a critical game/set."

This is the structural fix locked in as a test. Any future prompt change that re-introduces the v1 behavior fails this test before merging.

Default suite: **222 passed, 5 deselected** (eval cases). Eval suite: **5 passed @ $0.10 total**.

**Phase 3A v2 is now feature-complete.** All 8 plan steps shipped. Remaining: Step 9 = first real paper run, which is an operational step (run on Mac mini, observe, tune).

### April 19, 2026 — Phase 3 v2 Steps 4 + 6: AgentLoop Rewrite + CLI Wiring

Two large pieces shipped together (the steps are tightly coupled — testing the new loop end-to-end requires the CLI to wire the bridge).

**Step 4: AgentLoop rewrite (`src/tennis_edge/agent/loop.py`)**

v1's DB-tail reader is gone. v2 is a pure subscriber:
- `on_signal(MonitorSignal)` is the public pub/sub endpoint MonitorBridge calls
- Pre-LLM gates: safety running + cooldown + queue cap (same as v1)
- Worker: prompt_builder (returns None = silent skip, NOT an LLM failure) → grounded Gemini → multi-stage hard gate
- Hard gates: `confidence_low` / `grounded_edge_below_min` / `rec_skip` / `edge_stale` (post-LLM re-check is now a HARD reject, not v1's soft log) / `no_recent_tick` (refuse to fill at unknown price) / `kelly_zero` / `risk:*`
- On gate pass: Kelly-size with confidence multiplier (1.0 high / 0.5 medium / 0.0 low), `risk.check_and_reserve`, then `exchange.place_order(OrderRequest(client_order_id=decision_id, ...))`
- On `place_order` exception: `risk.release()` to unwind, log decision with `reject_reason="order_failed"`. **3 consecutive order failures → `safety.kill(ORDER_CONSECUTIVE_FAILURES)`** (new dedicated TripReason, distinct from LLM failures so post-mortem analytics can tell them apart).

`AgentLoop` constructor now takes a `prompt_builder` async callable + an `exchange: ExchangeClient`. The `ExchangeClient` ABC abstracts paper vs live cleanly — no `Executor` class needed (the eng-review scope reduction in action). The CLI picks `PaperTradingEngine` or `KalshiClient` and passes it in.

`SafetyMonitor` got a public `kill(reason, detail)` method so callers (Phase 3 v2: AgentLoop's order-failure path) can flip state to KILLED with a specific TripReason without piggybacking on the LLM rail. New `TripReason.ORDER_CONSECUTIVE_FAILURES` enum value.

**Test rewrite** (`tests/test_agent_loop.py`, 26 cases — replaces v1's 24):
- `_TickReader`: latest_for_ticker, none on missing, price_cents preference
- `on_signal`: enqueues running, drops paused/killed, cooldown blocks, queue cap drops overflow
- Worker happy path: places paper order, logs `executed=True` + real `order_id`
- Hard rejects: confidence_low, grounded_edge_below_min, rec_skip, edge_stale (with stale tick), no_recent_tick
- Risk gate: tight per-market cap rejects second trade with `reject_reason` starting "risk:"
- Executor failures: `place_order` raises → `risk.release` → no double-charge → 3 in a row → `ORDER_CONSECUTIVE_FAILURES` kill
- LLM failures: counted via `safety.record_llm_failure`, 3 trip kill, success resets counter
- Freshness gate, paused-mid-handle skip, prompt_builder=None silent skip
- Decision log shape: run_id, prompt_hash, raw_output, edge_at_execution all populated

**Step 6: CLI wiring (`src/tennis_edge/cli.py agent start`)**

Rewrote `agent_start` to wire the v2 stack:
- New flags: `--executor paper|live`, `--whitelist atp-wta-main|all-tennis`, `--min-prematch-ev` (was `--min-edge`), `--bankroll`, `--max-position`, `--max-total-exposure`, `--daily-loss-limit`, `--mode shadow|auto`
- Defaults are the safe path: `paper + atp-wta-main + shadow` so a bare `tennis-edge agent start` cannot place real orders.
- Spawns four coroutines via `asyncio.gather`:
  1. `MonitorBridge.run` — scans Kalshi, emits signals to `agent_loop.on_signal`
  2. `AgentLoop.run` — drains signal queue, runs grounded Gemini, places orders
  3. `SettlementPoller.run` — 15min counterfactual P&L backfill, **wired with `risk=risk` so settlements update `daily_pnl`**
  4. `SafetyMonitor.watchdog_loop` — 30s health check, real `RiskManager` (no more `_NullRisk` placeholder)
- `_DummyWS` retained — agent doesn't own a Kalshi WS in v2 either (tick-logger has it). The two WS kill switches stay opted-out by reporting fresh timestamps; `TICK_LOGGER_STALE` is the actual health signal.

CLI smoke-tested:
- `tennis-edge agent --help` lists subcommands
- `tennis-edge agent start --help` shows all 11 flags with defaults
- `tennis-edge agent status` displays correctly with empty JSONL

Full suite: **222 passed** (220 prior + 26 new agent_loop − 24 old = +2 net).

**Next**: Step 8 = required eval suite (5 historical Kalshi markets, `pytest -m eval`). Then Step 9 = first real paper run.

### April 19, 2026 — Phase 3 v2 Steps 3 + 7: Idempotency + RiskManager Wire

Two short, independent lanes shipped together:

**Step 3: `OrderRequest.client_order_id` plumbing**
- Added `client_order_id: str | None = None` to `OrderRequest`
- `KalshiClient.place_order` already calls `model_dump(exclude_none=True)` so the field auto-flows to the wire when set
- Phase 3A `AgentLoop` will set `client_order_id = decision.decision_id` (UUID4) so a network-timeout retry hitting Kalshi twice is treated as the same order — single fill, not double position
- Paper engine accepts the field but ignores it (no network = no idempotency need)

Tests (`tests/test_exchange_idempotency.py`, 8 cases):
- Schema default is None; accepts a string; round-trips JSON
- `model_dump(exclude_none=True)` includes when set, excludes when None
- KalshiClient sends in body when set (verified with respx mock)
- KalshiClient omits from body when None (avoids null-validation issue)
- Paper engine accepts without crashing

**Step 7: SettlementPoller → RiskManager wire (closes the v1 dormant kill switch)**
- `SettlementPoller(..., risk: RiskManager | None = None)` — optional injection
- After every settlement append, calls `await self.risk.record_settlement(ticker, pnl)` if `risk` is wired
- Without this, `RiskManager.state.daily_pnl` stayed at 0 forever and the SafetyMonitor's `DAILY_LOSS_LIMIT` kill switch was dormant — the same bug as v1
- A buggy `record_settlement` is logged but does NOT halt the poller (settlement record is the source of truth, daily_pnl is derived)

Tests (`tests/test_agent_settlement.py`, +4 cases, 25 total):
- `record_settlement` called with correct pnl
- Backwards compatible: works without `risk` param (analytics replay)
- **End-to-end regression**: 3 losing settlements at -$50 each (counterfactual) → `daily_pnl = -$150` → `SafetyMonitor.check_daily_pnl` trips `DAILY_LOSS_LIMIT`. Without the v2 wire, this test fails because daily_pnl never moves. This is the v1 bug, captured.
- A crashing risk manager doesn't take down the poller; settlement still appends

Full suite: **220 passed** (208 prior + 8 idempotency + 4 settlement regression).

**Next**: Step 4 = `AgentLoop` rewrite. Depends on Steps 1+2 (both shipped). Drops DB-tail reader, adds `on_signal` endpoint subscribed to MonitorBridge, calls `exchange.place_order(OrderRequest(client_order_id=decision.decision_id, ...))` after the decision gate. ~2h.

### April 19, 2026 — Phase 3 v2 Step 2: MonitorBridge

**Built**: `src/tennis_edge/agent/monitor_bridge.py` — embedded scanner-runner that pushes filtered signals to `AgentLoop` via async callback. v2 replacement for v1's DB-tail reader.

**Architecture**:
- Periodic poll (~15s) over series whitelist via `KalshiClient.get_markets`
- Per market: fetch orderbook → run `EVScanner.analyze_market_pair` → produce `Opportunity`
- Three filters in order: series whitelist (category), price band (10-90c default), min prematch EV (0.15 default)
- Pass-through emits `MonitorSignal` via `await on_signal(sig)` to consumer

**`MonitorSignal` fields**: ticker, player_yes, player_no, category, market_yes_cents, model_prob, market_prob, prematch_ev, recommended_side, detected_at. Carries everything `AgentLoop.on_signal` will need without requiring it to recompute or look up.

**Whitelists exposed as constants**:
- `WHITELIST_ATP_WTA_MAIN` = `("KXATPMATCH", "KXWTAMATCH")` — Phase 3A v2 Week 1 default
- `WHITELIST_ALL_TENNIS` = adds Challengers, for Week 2+

**Robustness**:
- `get_markets` failure on one series logs and continues to the next
- Orderbook failure on one ticker → scanner runs with `ob=None` (falls back to last_price / yes_bid)
- Consumer (`on_signal`) exception is logged but does NOT block the bridge from emitting subsequent signals — the bridge has to be more reliable than its consumer
- `run()` loop survives a `scan_once` crash (logs + continues to next interval), exits promptly on `request_stop()`

**Tests**: `tests/test_agent_monitor_bridge.py` — 21 cases.
- `_category_passes` helper: 7 cases mapping Opportunity.category to series whitelist
- `scan_once` happy path: signal emitted with correct fields
- 4 filter rejection paths: below min EV, outside price band, mid_price=None, Challenger under Main whitelist
- 1 positive: Challenger passes under full whitelist
- Robustness: empty markets, None opportunity, get_markets exception isolated per series, orderbook exception falls through to scanner, crashing consumer doesn't kill bridge
- `run()` loop: exits promptly on stop, survives scan exception
- `stats` counts scans and signals correctly

Full suite: **208 passed** (187 prior + 21 new).

**Next**: Step 3 = `OrderRequest.client_order_id` plumbing (independent lane, takes ~1h). Then Step 4 = `AgentLoop` rewrite (depends on both Step 1 and Step 2).

### April 19, 2026 — Phase 3 v2 Step 0 + Step 1: Grounded Smoke + GeminiGroundedProvider

**Step 0 grounded smoke** (5 min, before any code):
- One real `gemini-3.1-pro-preview` call with `tools=[GoogleSearch()]` enabled
- 18.2s latency, **$0.0153 per call**
- Returned valid JSON; correctly cited a real ATP tour URL with Madrid Open data
- **Cost is BELOW the $0.03-0.05 estimate**, monthly $50 cap = ~3260 calls. Comfortable.
- Decision: proceed without retuning defaults

**Step 1 GeminiGroundedProvider** (extended `agent/llm.py`):
- New `PROMPT_TEMPLATE_GROUNDED_V1` — explicitly instructs the model to use Google Search to verify live state, with hard rule "if the match is settled or in progress, your edge_estimate should match the market price (no fade)". This is the v1 fix.
- Refactored `GeminiProvider.analyze` to use `_get_template()` and `_build_config()` overridable hooks. No behavior change for ungrounded path.
- New `GeminiGroundedProvider(GeminiProvider)` overrides those two hooks: returns the grounded template and a config with `tools=[GoogleSearch()]`. Drops `response_schema` (not always compatible with tool calling per Google docs); relies on prompt-level shape spec + pydantic validation, which has been proven via real smoke.
- Provider name suffixed with `-grounded` so `BudgetTracker` keeps grounded spend in its own bucket.
- Default `request_timeout_s` bumped from 60s to 90s (grounded calls are slower; smoke = 18s).

**End-to-end real smoke through new class**: Sinner vs Norrie at 85c. Gemini searched, found the match had just finished 6-2 7-5, **correctly recommended SKIP** with high confidence and 0.85 edge_estimate (matching settled price, no edge to extract). Cost $0.0165, 18.7s.

This is the structural fix for v1: agent now refuses to trade against settled outcomes because grounding tells it the truth.

**Tests** (`tests/test_agent_llm.py` +8 cases):
- `_get_template()` returns grounded template (regression check that v1 path stays on V1)
- `_build_config()` includes `GoogleSearch` tool
- `_build_config()` drops `response_schema` (incompatibility avoidance)
- Provider name suffixed with `-grounded`
- Default timeout ≥ 90s
- Inherits API key check from base
- Ungrounded provider config unchanged (regression)
- Grounded prompt contains "Google Search" directive + JSON schema spec inline + settled-outcome rule

Full suite: **187 passed** (179 prior + 8 new).

**Next**: Lane B = `MonitorBridge` (parallelizable; runs scanner, emits signals via async callback).

### April 19, 2026 — Phase 3 v2 Plan Eng-Reviewed + Tightened

Ran `/plan-eng-review` on `docs/PHASE_3_V2_PLAN.md`. 10 issues found, 4 architectural forks resolved:

1. **Dropped `Executor` ABC + `PaperExecutor` + `LiveExecutor` + `ConfirmGate` classes.** They duplicated the existing `ExchangeClient` ABC. `AgentLoop` takes an `ExchangeClient` directly; CLI flag picks paper or live. -4 classes, plan back under complexity threshold.
2. **Killed manual-confirm gate.** 30s TUI prompt was theatrical inside detached tmux (stdin gone → every prompt auto-rejects → cohort never reaches 20 → never flips to live). Replaced with single explicit decision point: "paper for 50 trades or until counterfactual P&L positive over 10+ settled, then user manually edits config to `--executor live`."
3. **Delete v1 prompt + ungrounded path immediately after first clean v2 paper run.** No `--llm-variant` flag, no rotting code; git history preserves v1.
4. **First action in coding session: 5-min real grounded smoke call.** Cost/latency assumptions ($0.03-0.05/call, 15-45s) are guesses; if reality is 3x higher, budget + cooldown + min-edge defaults all need to change before writing 4hr of code.

**7 critical test paths** added to plan that MUST land with the code (each one corresponds to a silent-failure mode):
- SettlementPoller → `risk.record_settlement` → daily P&L kill switch (the v1 dormant-switch bug)
- `client_order_id` idempotency on `place_order` retry
- Gemini grounded edge < 0.10 → hard reject
- Confidence == "low" → hard reject
- Post-LLM edge re-check < 0.08 → HARD reject (v1 was soft)
- `place_order` exception → `risk.release` called → retry safe
- Detached-tmux startup runs without stdin

**Required eval suite**: 5 historical Kalshi tennis markets, run grounded provider, assert recommendation ≥60% match outcome + `edge_estimate` within 0.20. ~$0.25/run, marked `@pytest.mark.eval`. Required before merging any prompt template change.

**Coding order**: 9 steps (added smoke as Step 0; explicit `client_order_id` step; explicit SettlementPoller→risk wire step). Total ~7h dev. Lanes B/C/E parallelizable.

Plan committed to `docs/PHASE_3_V2_PLAN.md`. Code starts tomorrow with the smoke call.

### April 19, 2026 — Phase 3 v2 Plan Locked (Monitor → Grounded Gemini → Auto-Execute)

After reading the 22 first-run decisions, the structural problem was clear: local Glicko-2 returns a constant 0.27 for unrated Challenger players, so the `edge` gate fires on noise, and v1's context builder passes empty form/H2H data to the LLM. The LLM reasoning is technically correct given the inputs — the inputs are broken.

**Direction change**: v2 flips the contract. Monitor detects a prematch EV signal → hands off to Agent → Gemini with **Google Search grounding** looks up live state itself → computes real P(YES) → Kelly sizing → Kalshi order (paper for first 20 decisions, live after).

This matches the user's actual $10K/month manual workflow (Gemini + live context + Kelly-sized manual trade).

**5 decisions locked via chat** (captured in detail in `docs/PHASE_3_V2_PLAN.md`):
1. Monitor → Agent = same-process pub/sub callback
2. Gemini Google Search grounding replaces screenshots (indefinitely)
3. Hard stops: $50/market, $500 total, -$200 daily → kill
4. First 20 trades paper + 30s manual confirm; auto after
5. Week 1 whitelist: ATP/WTA Main only (Challenger Week 2+)

**v1 code mostly survives**: `decisions.py`, `llm.py` (extended), `safety.py`, `settlement.py`, `risk.py`, `sizing.py`, `paper.py`, `scanner.py` all reused. The rewrites are `loop.py` (drop DB-tail, subscribe to Monitor) and `runtime.py` (drop local `model_prob_fn` for Gemini path). New files: `grounded_llm` (in `llm.py`), `monitor_bridge.py`, `executor.py`.

**Estimated ~7.5 hours** of dev to reach first paper run. Test strategy: unit coverage per module with fakes, one end-to-end integration test, one real-network smoke against Gemini grounded. Target ~200 tests post-v2.

**12 distinct safety layers** between "Gemini says X" and "money moves" — enumerated in the plan doc. Paper executor default on any run without explicit `--executor live`.

Plan committed to `docs/PHASE_3_V2_PLAN.md`. Re-run `/plan-eng-review` on it tomorrow before coding to catch gaps.

### April 19, 2026 — Phase 3A First Real Run (Mac Mini Smoke Test)

**Bootstrapped Mac mini** with Phase 1 data via `scripts/macmini_bootstrap.sh`:
- `git pull origin phase-2` (all 10 Phase 2 commits).
- `pip install -e '.[agent]'` installed `google-genai` 1.73.
- 114 MB `phase1_data.db` `scp`'d from MacBook, imported via `scripts/import_phase1_data.py`. Final counts on Mac mini: players 66,010 / matches 78,762 / rankings 1,340,288 / glicko2_ratings 46,898 / **market_ticks 613,342 preserved** (tick-logger's tmux session ran uninterrupted the whole time).
- `data/models/latest.joblib` (3.8 KB logistic model) placed.
- `.env` populated with `TENNIS_EDGE_GEMINI_KEY`.
- `pytest` 179 passed on Mac mini (same as MacBook).

**First live agent run** (`tennis-edge agent start --min-edge 0.15`, ~3 minutes):
- ✅ End-to-end pipeline works. No crashes, no kill-switch trips.
- ✅ 22 decisions written to `data/agent_decisions.jsonl`.
- ✅ 13 real Gemini 3.1 Pro Preview calls (the other 9 candidates were filtered pre-LLM: stale queue, budget reserve, or edge collapsed before dequeue).
- ✅ Post-LLM edge re-check firing: one example logged `edge=-0.369 → -0.429 rec=BUY_NO` on a WTA challenger, showing the re-validation path is exercised.
- ✅ BudgetTracker persisted state correctly across the run.

**Cost finding** (the surprise of the day):
- 13 calls / 3 min = **$0.1391 total ≈ $0.0107 per call** — matches the single-call smoke test from Lane 1E exactly, so pricing is predictable, not a bug.
- Extrapolated: **$2.80/hr, $67/day, ~$2,000/month at `--min-edge 0.15`.**
- The configured monthly cap of $50 would trip `BUDGET_EXCEEDED` in about 18 hours at this rate.
- Root cause: `--min-edge 0.15` combined with illiquid Challenger markets (wide bid/ask spreads drive |edge| above threshold easily) produced a candidate every ~14 seconds. The planned `--min-edge 0.08` would only increase this.

**Decision**: paused before any long run. Threshold / cooldown / budget calibration deferred to tomorrow. The 22 existing decisions are kept for analysis (will read them tomorrow morning to judge LLM reasoning quality before touching knobs).

**Likely Phase 3A operating config** (to be verified tomorrow):
- `--min-edge 0.12` or `0.15` (wait for empirical candidate-rate data)
- `--cooldown 1800` (30 min per ticker, up from 5 min)
- `--gemini-budget 100-200` (top up API balance)

**Also shipped today**:
- `scripts/import_phase1_data.py` — idempotent SQLite DELETE+INSERT for Phase 1 tables, never touches `market_ticks`.
- `scripts/macmini_bootstrap.sh` — one-shot Mac mini setup wrapping git pull + pip install + model placement + .env init + data import + pytest.

**Operational learning**: `tennis-edge` CLI is only on PATH inside the venv. Either `source .venv/bin/activate` before using or `ln -s ~/Tennis-Edge/.venv/bin/tennis-edge /usr/local/bin/tennis-edge` for a global shim. The agent start command must be run inside the venv either way (for the import path).

### April 19, 2026 — Agent Lane H: CLI + Runtime Wiring (Phase 3A Ready to Run)

**Built**: `src/tennis_edge/agent/runtime.py` (real `model_prob_fn` + `context_builder` wired to existing Glicko-2 stack) + CLI command group `tennis-edge agent`.

**With this lane shipped, Phase 3A is runnable end-to-end.** `tmux new -s agent && tennis-edge agent start` on the Mac mini and shadow-trade decisions start accumulating.

**`agent/runtime.py`**:
- `AgentRuntime` bundles DB, `RatingTracker`, `FeatureBuilder`, `LogisticPredictor`, and `MarketCache`. Exposes closures that match the loop's protocols.
- `model_prob_fn(ticker)` — sync, cache-only. Cache miss → fires an async prewarm via `loop.create_task` so the next reader poll (2s later) hits. Returns None on miss, unknown player, or model failure.
- `context_builder(ticker, model_prob, market_yes_cents)` — sync, queries DB for last-10 form, H2H summary, rest days, surface inference.
- `MarketCache` — 5 min TTL, in-flight coalescing via `asyncio.Task` map so concurrent `get()` calls share one REST hit.
- `parse_market_title()` handles canonical Kalshi shape "Will X win the X vs Y: Round Of 32 match?". Tests cover unparseable and the YES-is-second-side case.
- Surface inference: "clay"/"madrid"/"roland" → Clay, "wimbledon"/"grass" → Grass, else Hard.
- Round inference: "final" (not "semifinal") → F, "semifinal" → SF, "quarterfinal" → QF, "16"/"32"/"64"/"128" → R16/R32/etc. Best-of defaults to 3 (Grand Slam override requires tournament metadata we don't carry today).

**CLI subcommands under `tennis-edge agent`**:
- `start [--mode shadow] [--min-edge 0.08] [--cooldown 300] [--gemini-budget 50] [--model gemini-3.1-pro-preview]` — foreground daemon. Reads `TENNIS_EDGE_GEMINI_KEY` from env or `.env`. Runs AgentLoop + SettlementPoller + SafetyMonitor watchdog concurrently via `asyncio.gather`. SIGINT/SIGTERM → graceful shutdown. On USER_FLATTEN trip, clears the flag post-exit so next start doesn't immediately re-trip.
- `pause` — `touch data/agent_control/pause`. Reversible.
- `resume` — remove pause flag.
- `flatten` — `touch data/agent_control/flatten`. Terminal trip. In 3A with no executor wired, daemon simply exits.
- `status` — reads decisions/settlements JSONL + budget JSON. Shows decision count, settlement count, unresolved, pause/flatten flags, per-provider budget, and counterfactual shadow P&L (wins/losses/void + total) when settlements exist. Never writes — safe to run concurrently with the daemon.

**Foreground-in-tmux pattern** matches the tick-logger workflow already proven on the Mac mini. No daemonization, no PID files, no systemd unit. `tmux new -s agent` + `Ctrl-B D` to detach + `tmux attach -t agent` to re-attach.

**Phase 3A safety simplifications** (documented in the `_DummyWS` / `_NullRisk` docstrings inside cli.py):
- Agent does not own a live WS in shadow mode — it reads `market_ticks` via the tick-logger's DB. Both `seconds_since_last_*` helpers return `0.0` so the WS kill switches never trip; the tick-logger-stale switch is the real health signal.
- No executor in 3A means no `RiskManager` calls, so `daily_pnl` is forced to 0 so the DAILY_LOSS_LIMIT switch stays dormant. Both switches come alive in Phase 3B when the executor lands.

**Live-match predicate**: `is_live_match()` queries `SELECT MAX(received_at) FROM market_ticks WHERE received_at > now-300`. Any tick in the last 5 minutes = a match is live somewhere. That's the signal the WS_STALE watchdog would need if the agent ever opens its own WS.

**Tests**: `tests/test_agent_runtime.py` — 21 cases. Covers title parsing (5), surface inference (4), round inference (6), tournament inference (2), MarketCache caches + coalesces + handles network errors + TTL expiry (4). Full model-prediction path is not re-tested here — already covered by the existing scanner path through `FeatureBuilder` and `LogisticPredictor`.

**Smoke-tested CLI**:
```
$ tennis-edge agent pause
Paused — flag at data/agent_control/pause
$ tennis-edge agent status    # shows flag: present
$ tennis-edge agent resume
Resumed
$ tennis-edge agent flatten
Flatten signaled — flag at data/agent_control/flatten
```

Full suite: 179 passed (158 prior + 21 new).

**Plan context**: Lane H per Phase 2 eng review — final lane of the original agent plan. With this landing, **the Phase 3A shadow-trade pipeline is production-ready**. Anthony's lanes (monitor redesign, real backtest engine) and Billy's arbitrage feasibility gate are the remaining Phase 2 work, plus Phase 3B/3C when ready.

### April 19, 2026 — Agent Lane G: Settlement Poller (Counterfactual P&L)

**Built**: `src/tennis_edge/agent/settlement.py` — closes the loop on Phase 3A analytics by backfilling `agent_settlements.jsonl` with counterfactual P&L per decision.

**The question this answers**: "If the agent had executed its recommendations, what would the P&L have been?" That's the single metric that decides whether Phase 3A promotes to 3B.

**Approach**:
1. Read all decisions + all existing settlements from the JSONL files
2. For each decision without a matching settlement, look up the Kalshi market
3. If `status in {"settled", "finalized"}`, compute counterfactual P&L at fixed notional and append a `SettlementRecord`
4. Idempotent: re-running never double-counts

**Counterfactual P&L math** (`counterfactual_pnl(rec, market_yes_cents, result, notional=$50)`):
- `BUY_YES` + YES wins: bought at P cents, $50 notional → `int($50 * 100 / P)` contracts, each pays $1, profit = `contracts * (100-P)/100`
- `BUY_YES` + NO wins: full stake lost, `-$50`
- `BUY_NO` + NO wins: symmetric (buys NO at `100-P` cents)
- `BUY_NO` + YES wins: `-$50`
- `SKIP` + any outcome: outcome recorded, `pnl=0` (no counterfactual trade). Still settled so replay analytics can ask "what fraction of SKIPs would have been profitable?"
- Void market (status settled, result empty): outcome=void, pnl=0 regardless of recommendation
- Degenerate fill prices (0c, 100c): outcome recorded, pnl=0 instead of dividing by zero

**Default notional $50** matches the Phase 3C position cap from the plan — lets shadow-mode numbers translate directly to expected 3C P&L without rescaling.

**Poller runtime**:
- `SettlementPoller(log, exchange, config)` with structural typing on `exchange` — accepts anything with `async get_market(ticker) -> Market`. Real `KalshiClient` in prod, `FakeExchange` in tests, no factory needed.
- `poll_once()` dedups decisions by ticker so one REST call resolves all decisions on the same market.
- `per_market_delay_s` (default 0.2s) keeps us under Kalshi's rate limit on backlogs.
- `run()` loops at `poll_interval_s` (default 900s = 15 min; matches the rough window between match end and Kalshi settlement).
- `request_stop()` short-circuits mid-loop and mid-poll.
- Exchange exceptions are logged and skipped per-ticker — one bad market doesn't halt the scan.

**Tests**: `tests/test_agent_settlement.py` — 21 cases.
- `counterfactual_pnl` math for every (rec, result) cell: BUY_YES/NO win/lose, SKIP, void, unknown result, degenerate prices, default notional, unknown recommendation warning (10)
- `poll_once`: writes for resolved market, skips non-terminal, idempotent for already-settled, handles exception per-ticker, void market, multiple decisions same ticker deduped, SKIP settles without P&L, "finalized" status also terminal, no-unresolved returns 0, stop short-circuits (10)
- `run()` loop exits on stop (1)

Full suite: 158 passed (137 prior + 21 new).

**Plan context**: Lane G per Phase 2 eng review. Independent of the main agent loop — can run in its own daemon or alongside. Phase 3A end-to-end metrics are now computable: `DecisionLog.replay()` joins decisions to settlements, analytics can compute win rate, ROI, and SKIP counterfactuals.

**Remaining: Lane H (CLI).** `tennis-edge agent start/pause/flatten/status`, real `model_prob_fn` + `context_builder` wired to the existing Glicko-2 DB. Then the daemon can actually run on the Mac mini.

### April 19, 2026 — Agent Lane 1F: Main Loop (Shadow Mode)

**Built**: `src/tennis_edge/agent/loop.py` — the agent spine. Ties every prior lane together into a single-worker, bounded-queue, DB-tailing shadow-trade loop.

**Components**:
- `MarketTickReader` — tails `market_ticks` via read-only SQLite URI. `latest_per_ticker()` returns one row per ticker since cursor (GROUP BY + MAX inside a subquery), cursor advances on each poll. `latest_for_ticker()` point-lookup for post-LLM edge re-check. `price_cents()` preference: `last_price → mid(bid,ask) → ask → bid → None`.
- `Candidate` dataclass — `decision_id` (UUID4 first 12 hex), ticker, `model_prob`, `market_yes_cents`, `edge_at_decision`, `enqueued_at` (monotonic).
- `AgentLoop.run()` — spawns `_reader_loop` + `_worker_loop`, waits on stop event OR `safety.is_killed()`, clean cancel on shutdown.

**Enqueue gating** (all must pass):
1. Cooldown: per-ticker, default 300s, started the moment we enqueue (not when LLM returns) to prevent same-market spam while LLM is working.
2. Price decodable: `price_cents(row)` not None.
3. `model_prob_fn(ticker)` returns a float (None = unknown ticker, skip).
4. `abs(edge) >= min_edge` (default 0.08 = 8pp, higher than Phase 1 scanner's 3% — only burn LLM dollars on real signals).
5. Queue not full: `queue_max` default 20, overflow drops with warning.

**Worker path** (single coroutine):
1. Safety check: `is_running()`? Skip if paused or killed.
2. Freshness gate: age > `max_candidate_age_s` (default 60s) → drop without LLM call.
3. `context_builder(ticker, model_prob, market_yes_cents)` → `PromptContext` or None. None = skip without counting as LLM failure (context unavailable ≠ LLM error).
4. `llm.analyze(ctx)` — catches `BudgetExceeded`, `LLMError`, bare `Exception`. All three route through `safety.record_llm_failure` so the 3x kill switch counts.
5. On success: `safety.record_llm_success()` resets the counter.
6. Post-LLM edge re-check: re-read latest tick, compute `edge_at_execution`. If `abs(edge_at_execution) < stale_edge_threshold`, set `reject_reason="edge_stale"`.
7. Append `AgentDecision` — in Phase 3A shadow mode, `executed=False` always. Edge-stale decisions are still logged so shadow analytics can measure LLM latency's cost in missed signals.

**Helpers for testing** (not wired into production paths): `tick_once()` single poll, `drain_once()` single candidate handoff, `cooldown_remaining(ticker)`. Enabled deterministic tests — no asyncio race flakes.

**What's NOT in this lane**:
- Executor (Phase 3B adds `human_in_loop` prompt + order placement; 3C adds restricted auto)
- Settlement poller (Lane G, separate)
- CLI wrapper (Lane H)
- Screenshot pipeline (deferred to Phase 3B per eng review Fork #3)

**Tests**: `tests/test_agent_loop.py` — 24 cases. Drive-deterministic via real SQLite `market_ticks` + controlled inserts + `tick_once()` / `drain_once()`. Covers:
- Tick reader: latest-per-ticker, cursor advance, price preference, missing DB (6)
- Enqueue gating: edge threshold, unknown ticker, no prices, cooldown block, cooldown=0 allows requeue, queue cap drop (7)
- Worker: paused skip, stale-candidate drop, context=None skip (no LLM failure), happy path logs decision, LLM failure increments counter, 3x failures trip kill switch, post-LLM edge stale → reject, post-LLM edge holds → no reject, killed-between-enqueue-and-dequeue skip, shadow mode never sets executed (10)
- `cooldown_remaining` (1)

Full suite: 137 passed (113 prior + 24 new).

**Plan context**: Lane 1F per Phase 2 eng review. Depends on all five prior lanes (B, C, 1A, 1E, 1D). With this lane shipped, the Phase 3A shadow-trade pipeline is code-complete. Remaining: Lane G (settlement poller to backfill outcomes) and Lane H (CLI `agent start/pause/flatten/status` so it can actually be run from the shell).

### April 19, 2026 — Agent Lane 1D: Safety Module (5 Kill Switches + File-Flag IPC)

**Built**: `src/tennis_edge/agent/safety.py` — owns agent run-state (`RUNNING` / `PAUSED` / `KILLED`) and evaluates every kill switch from Lane B, Lane C, Lane 1A, and Lane 1E.

**Kill switches** (any one flips `RUNNING` → `KILLED`, terminal):
1. `LLM_CONSECUTIVE_FAILURES` — 3 in a row, any `LLMError` subclass. `record_llm_success()` resets the counter.
2. `WS_RECONNECT_STARVATION` — `seconds_since_last_connect > 60` (or `None` = never connected). Reads `last_connect_ts` from Lane C.
3. `WS_STALE_WITH_LIVE_MATCH` — `seconds_since_last_message > 60` AND caller-provided `live_match_fn()` returns True. Supports sync or async predicate. Quiet nights with no matches do not trip.
4. `TICK_LOGGER_STALE` — `SELECT MAX(received_at) FROM market_ticks` via read-only SQLite URI; if > 60s old, empty, missing table, or missing DB, trip. This is the 5th kill switch added during the eng review (not in the original plan).
5. `BUDGET_EXCEEDED` — any configured provider has `remaining_usd <= 0`. Mirrors Lane 1E's pre-flight `BudgetExceeded` but at the daemon level so the agent exits instead of silently rejecting every candidate.
6. `DAILY_LOSS_LIMIT` — `risk.state.daily_pnl <= -limit` (default $200). Lane B's RiskManager trips its own internal kill; this mirrors it at the agent layer for clean shutdown.
7. `USER_FLATTEN` — flatten flag file present. Daemon responsibility: close agent-owned positions, then `clear_flatten_flag()` before exit.

**File-flag IPC** (CLI writer, daemon reader):
- `data/agent_control/pause` — exists = `PAUSED`, deleted = `RUNNING` (reversible, re-checked every watchdog tick).
- `data/agent_control/flatten` — exists = trip `USER_FLATTEN`. Flatten takes precedence over pause.
- Helpers: `touch_pause_flag`, `clear_pause_flag`, `touch_flatten_flag`, `clear_flatten_flag`. All tolerate missing files.

**Watchdog loop** (`watchdog_loop`) runs the checks in order: user flags → budget → daily P&L → WS health → tick-logger. Short-circuits when `PAUSED` so a user pausing during a known-bad condition (market close, mid-maintenance) cannot trip a reversible pause into a terminal kill. Exits cleanly once `is_killed()` flips.

**Trip idempotency**: first trip wins. `_trip_once` is lock-guarded; subsequent attempts are no-ops. `KILLED` is terminal — clearing the pause flag cannot resurrect a killed daemon.

**Dep typing**: deps (WS, budget, risk manager, DB path) come in as structural `Protocol`s so tests use minimal duck-typed fakes. Real wiring happens in Lane 1F / Lane H.

**Tests**: `tests/test_agent_safety.py` — 28 cases.
- Initial state (1)
- LLM counter: trips at threshold, success resets (2)
- WS: reconnect starvation, never connected, stale+live match trip, stale without live match does not, async predicate, fresh does not trip (6)
- Tick logger: fresh OK, stale trips, empty table, missing DB, missing table (5)
- Budget: exceeded trips, remaining OK (2)
- Daily P&L: limit trips, within limit OK, positive P&L OK (3)
- File-flag IPC: pause sets/clears, flatten trips, flatten beats pause, cleanup tolerates missing (5)
- Trip idempotency: first trip wins, KILLED terminal (2)
- `watchdog_loop`: exits when killed, skips checks while paused (2)

Full suite: 113 passed (85 prior + 28 new).

**Plan context**: Lane 1D per Phase 2 eng review. Depends on Lane B (RiskManager), Lane C (WS timestamps), Lane 1A (LLMError hierarchy), Lane 1E (BudgetTracker). Blocks Lane 1F (`agent/loop.py` — worker needs the monitor to gate candidates).

### April 19, 2026 — Agent Lane 1E: LLM Provider + Budget Tracker

**Built**: `src/tennis_edge/agent/llm.py` — provider abstraction, real Gemini implementation, persistent per-month budget enforcement, structured JSON output via response_schema.

- `LLMProvider` ABC with `async analyze(PromptContext) -> LLMResult`.
- `FakeLLMProvider` for tests. Tracks `call_count`, can be injected with a canned `EvAnalysis` or configured to raise.
- `GeminiProvider` wraps the new `google-genai` SDK (not the deprecated `google-generativeai`). Lazy import inside `__init__` so the module loads without the SDK installed.
- `BudgetTracker` — per-provider monthly USD cap, atomic state file at `data/agent_budget.json` (tempfile + `os.replace` + `fsync`), month rollover resets counters, `reserve()` raises `BudgetExceeded` pre-flight when the projected total would cross the cap, `record()` logs the actual cost after the call completes.
- `PricingRates(input/output/thinking per 1M usd)` — Gemini 3.x bills thinking tokens separately, so the tracker accounts for all three.
- `PromptContext` + `build_prompt` + `PROMPT_TEMPLATE_V1` — text-only structured prompt (Phase 3A picked text-only per eng review). Includes ticker, player names, tournament, surface, pre-match model prob, Kalshi YES cents, rolling form, H2H, rest days, free-form extra_notes.
- `GEMINI_RESPONSE_SCHEMA` — JSON schema passed via `response_schema` to force the model into the exact shape `EvAnalysis` expects. JSON parse or pydantic validation failure raises `LLMOutputError` which the 3x-consecutive kill switch in Lane 1D will count.

**Error hierarchy**: `LLMError` (base) → `LLMCallError` (network, auth, server), `LLMOutputError` (parse/validate), `BudgetExceeded` (hard cap). All three inherit from `LLMError` so the safety watchdog catches one class.

**Real smoke test against Gemini 3.1 Pro Preview**:
- Sample input: Holmgren vs Broady clay R32, pre-match 53%, market 15c, form + H2H + rest days.
- Output: `edge_estimate=0.55, recommendation=BUY_YES, confidence=high`, full reasoning + 4 key factors, all schema-valid.
- Tokens: 224 input / 231 output / 714 thinking.
- Cost: ~$0.01 per decision at preview-tier rates.
- **Action**: default `max_output_tokens` raised from 2048 to 8192. Gemini 3.x reasoning budget is large enough that the first smoke with 512 got truncated mid-JSON and raised `LLMOutputError`. 8192/8192 gives 10x safety margin.

**Plumbing**:
- `pyproject.toml` — new optional extra `[agent] = google-genai>=1.70`. Install with `pip install -e '.[agent]'`.
- `.env` — gitignored, holds `TENNIS_EDGE_GEMINI_KEY`. `.env.example` committed as a template.
- Old `google-generativeai` SDK is deprecated per Google; only the new `google-genai` namespace is used.

**Tests**: `tests/test_agent_llm.py` — 20 cases. Covers `PricingRates` math, `BudgetTracker` reserve/record/persist/rollover/corrupt-file tolerance, `FakeLLMProvider` happy + budget wiring + error paths (reserve succeeds but record skipped on call failure — no silent budget consumption), `LLMError` hierarchy, `build_prompt` substitution, and `GeminiProvider` missing-key path. Real Gemini calls are not in the unit suite — tests use the fake.

Full suite: 85 passed (65 prior + 20 new).

**Plan context**: Lane 1E per Phase 2 eng review. Depends on Lane 1A (`EvAnalysis` schema). Blocks 1F (loop needs a provider) and 1D (safety needs the error hierarchy to count failures).

### April 18, 2026 — Agent Lane 1A: Decision Log (JSONL Append-Only)

**Built**: `src/tennis_edge/agent/` package with `decisions.py` — schemas + JSONL writer. This is the foundation; everything else in Lane 1 (`llm.py`, `safety.py`, `loop.py`) imports from here.

Two schemas:
- `EvAnalysis` — parsed LLM output (edge_estimate ∈ [0,1], recommendation ∈ {BUY_YES, BUY_NO, SKIP}, confidence, reasoning ≤ 2000 chars, up to 5 key_factors). Parse failure = LLM failure for the 3x-consecutive kill switch in Lane 1D.
- `AgentDecision` — full record with decision_id (join key), model_pre_match, market_yes_cents, edge_at_decision, llm_provider + llm_prompt_hash + llm_raw_output, analysis, mode ∈ {shadow, human_in_loop, auto}, executed + order_id + edge_at_execution.

Two JSONL files, both append-only with fsync:
- `data/agent_decisions.jsonl` — one line per model→LLM cycle
- `data/agent_settlements.jsonl` — one line per settled market, written later by the settlement poller (Lane G), joined on decision_id

**Why two files instead of rewriting one**: append-only is crash-safe (O_APPEND is atomic per small write on POSIX), the poller runs on its own schedule, and `replay()` does the join at read time. Rewriting JSONL rows to backfill `outcome_at_settle` is how you lose data.

**Why JSONL instead of SQLite**: Phase 3A expects ~20 decisions/day × 6 weeks ≈ 840 rows. Flat file is cheaper than a schema migration and stays greppable. Promote only when indexed queries are actually needed.

**Durability**: every append fsyncs before return. An agent that can lose audit records to power loss is an agent you cannot trust.

**Also shipped**: `prompt_hash(template, inputs)` — stable 16-char sha256 for A/B-grouping decisions by prompt version. Survives dict key reordering (sort_keys=True in JSON encoding).

**Tests**: `tests/test_agent_decisions.py` — 25 cases across schema validation (8), round-trip + ordering (4), replay + out-of-order settlement join (3), tolerance for blank lines / malformed lines / missing files / nested parent dirs (4), and prompt_hash determinism (6).

Full suite: 65 passed (40 prior + 25 new).

**Plan context**: Lane 1A per Phase 2 eng review. Dependency for 1E (`llm.py` fills in `analysis`), 1D (`safety.py` counts parse failures), 1F (`loop.py` appends), G (poller appends settlements). Nothing in Lane 1 can land until this does — it's the data contract for the whole agent.

### April 18, 2026 — Agent Lane C: WebSocket Health Timestamps

**Built**: `KalshiWebSocket` now exposes two monotonic timestamps for the Phase 2 agent safety watchdog.

- `last_message_ts` — set on every decoded frame (ticker, orderbook, trade, fill, subscribed, error — all count). Cleared to `None` on disconnect so a stale pre-disconnect reading cannot mask an outage.
- `last_connect_ts` — set after a successful WebSocket upgrade. Preserved across disconnects so the watchdog can detect reconnect-loop starvation independently of message traffic.
- Helpers: `seconds_since_last_message()` and `seconds_since_last_connect()` return `None` or wall seconds. No locking needed — scalar read/write is atomic under the GIL and readers only compare against `time.monotonic()`.

**Why two timestamps, not one**: the original plan said "auto-pause if WebSocket disconnects > 60 seconds" but that conflates two distinct failure modes. Quiet nights have no ticker traffic by design — treating silence as an outage would trip false positives. The agent safety module will combine these two signals with an "is any match live" predicate:
- `seconds_since_last_connect > 60` → link is down, pause regardless of match state.
- `seconds_since_last_message > 60` AND live match in progress → link is up but the ticker stream went silent during play, pause.

**Tests**: `tests/test_ws_timestamps.py` — 10 cases. Drives `_handle_message` directly with synthetic frames; no real socket needed. The last case (`test_watchdog_shape_healthy_then_stale`) walks the exact state sequence the safety watchdog will see and asserts the timestamps behave the way that watchdog expects.

Full suite: 40 passed (30 prior + 10 new).

**Plan context**: Lane C of Phase 2 eng review. Independent of Lane B (risk race) and Lane 1 (agent spine). Landing before `agent/safety.py` so the safety module can build on a stable timestamp API instead of monkey-patching ws.py.

### April 18, 2026 — Agent Lane B: RiskManager Race Fix

**Built**: Concurrency-safe `RiskManager` in `strategy/risk.py`. Pre-existing TOCTOU race between `check_trade` and `record_trade` would have let two concurrent agent candidates both pass the limit check and both record, producing combined exposure above the configured caps. With Phase 2's LLM-worker queue about to exercise this path for real, the fix had to land before any agent executor code.

- New atomic API: `await mgr.check_and_reserve(decision)` validates and records under a single `asyncio.Lock`. Returns `(allowed, reason)`.
- `await mgr.release(decision)` unwinds a reservation when the downstream order fails. Idempotent — safe to call after settlement cleared the position, and safe to call twice without going negative.
- `record_settlement` and `reset_daily` are now async and locked. `summary()` stays sync (read-only snapshot).
- Deprecated split API (`check_trade` / `record_trade`) removed. Grep confirmed zero callers in the codebase, so clean delete.

**Tests**: `tests/test_risk_concurrency.py` — 10 cases. The regression test (`test_concurrent_reserve_respects_per_market_cap`) races 10 candidates through `asyncio.gather`; before the fix, all 10 would be admitted. After the fix, exactly 2 are admitted and final exposure matches the cap.

Full suite: 30 passed (20 existing + 10 new).

**Plan context**: Per Phase 2 eng review, this is Lane B — independent of the agent spine and of Lane C (WS timestamps). Landing first unblocks `agent/loop.py` without requiring agent code to also ship the race fix. Review doc decisions locked:
- Tick source: DB tail from `market_ticks` (single-writer, many-reader)
- Runtime: headless CLI daemon (`tennis-edge agent start/pause/flatten`)
- Phase 3A LLM input: text-only structured prompt (screenshots deferred to 3B)
- Decision log: JSONL append-only

### April 17, 2026 — Tick Logger Shipped (Day 1)

**Built**: `tennis-edge log-ticks` — Kalshi WebSocket tick streaming to SQLite (`market_ticks` table)

- New table `market_ticks(id, ticker, ts, yes_bid, yes_ask, last_price, volume, received_at)` with indexes on `(ticker, ts)` and `received_at`
- Async logger filters to tennis markets only (4 series prefixes: KXATPMATCH, KXATPCHALLENGERMATCH, KXWTAMATCH, KXWTACHALLENGERMATCH)
- Buffered writes: flushes every 30s OR every 500 rows whichever first (was initially 5s/100, raised to reduce I/O after deployment)
- Graceful SIGINT/SIGTERM handling with final buffer flush
- 60-second stats heartbeat for visibility on long-running deployments

**Deployment**: Running on Mac mini (not MacBook, which sleeps when David closes the lid).
- Mac mini config: `pmset sleep=0 disksleep=0 autorestart=1` so it survives power cuts and never sleeps
- Inside `tmux new -s ticks` so SSH disconnects don't kill it
- Verified producing data: 60s smoke test = 59 ticks across active markets, ~1 tick/sec at off-peak (will spike to 100s/sec during peak match hours)

**Why this is the critical path**: Every day the logger is not running is a day of real Kalshi price data Anthony's backtest engine can never recover. Phase 1 backtest used synthetic odds; phase 2 demo credibility depends entirely on this dataset accumulating cleanly.

**Branch**: All phase 2 work on `phase-2` branch. `main` stays stable. Teammates work in their own forks.

## Phase 2 — Open Workstreams

### David
- [x] Tick logger (technically Anthony's workstream but David built it Day 1 to start data accumulation)
- [ ] Agent shadow-trade logging skeleton (`agent_decisions` table + writer)
- [ ] Gemini API client + structured EV prompt v1
- [ ] LLM research R-key in monitor

### Anthony
- [ ] Monitor layout redesign (multi-panel: market list / detail / research / positions)
- [ ] Onboarding wizard (first-run flow: ingest → train → first paper trade)
- [ ] Real backtest engine consuming `market_ticks` data

### Billy
- [ ] Arbitrage feasibility study (Week 1 gate): pull Kalshi + Polymarket tennis orderbooks, check 3 arb types (cross-venue, intra-Kalshi YES/NO sum, related-market consistency). Decision criteria: ≥5 actionable signals/day with >$5 expected profit each → continue. Else pivot to backtest support.

## Phase 2 — Premises Being Tested

1. **Tennis arb opportunities exist on Kalshi at meaningful frequency.** UNVERIFIED — Billy's week 1 study confirms or pivots.
2. **LLM with screenshot + structured prompt produces useful EV analysis.** PARTIALLY VERIFIED ($10K/month with Gemini). Agent shadow-trade phase tests reproducibility without human judgment in loop.
3. **5 weeks of WebSocket-collected Kalshi prices = sufficient sample for credible backtest.** Reasonable for high-volume markets, marginal for Challengers. Acceptable.
4. **3 teammates can ship 3 distinct workstreams in 5-6 weeks while integrating cleanly.** Tight. Week 5 reserved for integration. Option 1 has built-in fallback if it fails.
5. **David's $10K/month manual edge is reproducible signal, not variance.** 275 trades, +5.9% ROI is moderate evidence. Not pure luck given consistency across categories.
