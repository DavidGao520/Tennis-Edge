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
