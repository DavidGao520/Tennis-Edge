# Tennis-Edge

A quantitative tennis prediction market trading bot for [Kalshi](https://kalshi.com). Combines Glicko-2 player ratings, machine learning match prediction, and real-time in-play EV calculation to identify trading opportunities in ATP/WTA tennis markets.

## Architecture

```
Live Scores (ESPN) ──→ In-Play Model ──→ Win Probability
                                              │
Sackmann + TennisMyLife ──→ Glicko-2 ──→ Features ──→ ML Model ──→ Pre-match Prob
                                              │
                                    Kalshi API ──→ Market Odds
                                              │
                                         EV = Model - Market
                                              │
                                    Kelly Criterion ──→ Bet Sizing
                                              │
                                    Risk Manager ──→ Execute / Pass
```

## Features

- **Glicko-2 Rating System** — 2,775 players rated across 78,762 historical matches (2000-2026)
- **36-Feature ML Model** — Logistic regression with Glicko-2 ratings, surface, fatigue, H2H, form, tournament level
- **Proper Training Pipeline** — Train/Validation/Test temporal splits, walk-forward CV, hyperparameter tuning, permutation importance
- **In-Play Win Probability** — Mathematical model computing exact win probability from any score state (sets, games, points) using tennis scoring structure
- **Real-Time EV Scanner** — Combines live scores + in-play model + Kalshi market odds to find edge
- **Kalshi API Integration** — RSA-PSS authenticated client for market scanning, order placement, position management
- **Risk Management** — Position limits, exposure caps, daily loss kill-switch
- **Paper Trading** — Full simulation engine sharing the same interface as live trading
- **Trading History Analysis** — Import and analyze Kalshi export data with correct P&L calculation

## Quick Start

```bash
# Install
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure Kalshi API
export TENNIS_EDGE__KALSHI__API_KEY_ID="your-key-id"
export TENNIS_EDGE__KALSHI__PRIVATE_KEY_PATH="path/to/private_key.pem"

# Ingest data (downloads ~80K matches from Sackmann + TennisMyLife)
tennis-edge ingest

# Compute Glicko-2 ratings
tennis-edge ratings

# Train model with validation
tennis-edge train

# Scan live in-play matches for EV
tennis-edge live

# Scan pre-match opportunities
tennis-edge opportunities

# Analyze your Kalshi trading history
tennis-edge history --export-dir path/to/kalshi_export/

# Look up a player
tennis-edge player Alcaraz
```

## Data Sources

| Source | Coverage | Data |
|--------|----------|------|
| [Sackmann tennis_atp](https://github.com/JeffSackmann/tennis_atp) | 2000-2024 | 74,906 matches, 65,989 players, 1.34M rankings |
| [TennisMyLife](https://stats.tennismylife.org) | 2025-2026 | 3,857 matches (daily updates) |
| [ESPN API](https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard) | Live | Real-time set/game scores |
| [Kalshi API](https://kalshi.com) | Live | Market odds, orderbooks, order execution |

## Model Performance

Trained on 46,790 matches (2005-2020), validated on 8,719 (2021-2023), tested on 6,932 (2024-2026).

| Metric | Value |
|--------|-------|
| Accuracy | 64.8% |
| Baseline | 64.6% |
| AUC-ROC | 0.642 |
| Brier Score | 0.217 |
| Best C (tuned) | 0.01 |

Top features by permutation importance: `mu_diff` (Glicko-2 rating gap), `p2_win_rate_last_20`, `p1_mu`, `p1_surface_wr`, `best_of_5`.

## Project Structure

```
src/tennis_edge/
├── cli.py              # Click CLI: ingest, ratings, train, live, opportunities, history, player
├── config.py           # YAML config + env var overrides
├── scanner.py          # Pre-match EV scanner
├── scanner_live.py     # In-play EV scanner (live scores + model + Kalshi)
├── data/
│   ├── db.py           # SQLite connection manager (WAL mode)
│   ├── schema.py       # Table definitions
│   ├── ingest.py       # Sackmann + TennisMyLife ETL pipeline
│   ├── history.py      # Kalshi trading history import & P&L analysis
│   └── models.py       # Domain dataclasses
├── ratings/
│   ├── glicko2.py      # Glicko-2 algorithm (Glickman spec)
│   └── tracker.py      # Rating history management
├── features/
│   ├── builder.py      # Feature vector orchestration (36 features)
│   ├── surface.py      # Surface encoding + surface-specific win rates
│   ├── fatigue.py      # Match load, rest days
│   ├── h2h.py          # Head-to-head records
│   ├── form.py         # Recent win rates (5/10/20 match windows)
│   └── tournament.py   # Tournament level + round encoding
├── model/
│   ├── predictor.py    # Logistic regression pipeline (pluggable for XGBoost)
│   ├── training.py     # Train/Val/Test splits, walk-forward CV, hyperparam tuning
│   ├── calibration.py  # Brier score, ECE, reliability diagrams
│   └── inplay.py       # In-play win probability from score state
├── strategy/
│   ├── kelly.py        # Full & fractional Kelly criterion
│   ├── sizing.py       # Position sizing with constraints
│   └── risk.py         # Kill-switch, exposure caps, position limits
├── exchange/
│   ├── base.py         # ExchangeClient ABC
│   ├── auth.py         # RSA-PSS request signing
│   ├── client.py       # Async Kalshi REST client
│   ├── paper.py        # Paper trading engine
│   ├── schemas.py      # Pydantic v2 API models
│   ├── matching.py     # Market title → player ID resolution
│   └── livescore.py    # Live score fetcher (ESPN + Sofascore)
├── backtest/
│   ├── engine.py       # Walk-forward backtesting with periodic retraining
│   ├── metrics.py      # P&L, ROI, Sharpe, drawdown
│   └── report.py       # Console + HTML reports
└── utils/
    ├── logging.py      # Structured logging with TRADE/OPPORTUNITY levels
    └── time.py         # Date helpers
```

## Documentation

- [Progress log](docs/PROGRESS.md) — timeline, current state, known limitations, and next steps.

## License

MIT

