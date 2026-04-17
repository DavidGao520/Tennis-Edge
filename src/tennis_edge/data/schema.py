"""SQLite DDL schema definitions."""

PLAYERS_DDL = """
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    hand TEXT,
    birth_date TEXT,
    country_code TEXT,
    height_cm INTEGER
);
"""

MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tourney_id TEXT NOT NULL,
    tourney_name TEXT,
    surface TEXT,
    draw_size INTEGER,
    tourney_level TEXT,
    tourney_date TEXT NOT NULL,
    match_num INTEGER,
    winner_id INTEGER NOT NULL REFERENCES players(player_id),
    loser_id INTEGER NOT NULL REFERENCES players(player_id),
    score TEXT,
    best_of INTEGER,
    round TEXT,
    minutes INTEGER,
    winner_rank INTEGER,
    loser_rank INTEGER,
    winner_seed INTEGER,
    loser_seed INTEGER,
    w_ace INTEGER, w_df INTEGER, w_svpt INTEGER,
    w_1st_in INTEGER, w_1st_won INTEGER, w_2nd_won INTEGER,
    w_sv_gms INTEGER, w_bp_saved INTEGER, w_bp_faced INTEGER,
    l_ace INTEGER, l_df INTEGER, l_svpt INTEGER,
    l_1st_in INTEGER, l_1st_won INTEGER, l_2nd_won INTEGER,
    l_sv_gms INTEGER, l_bp_saved INTEGER, l_bp_faced INTEGER,
    UNIQUE(tourney_id, match_num)
);
"""

RANKINGS_DDL = """
CREATE TABLE IF NOT EXISTS rankings (
    ranking_date TEXT NOT NULL,
    ranking INTEGER NOT NULL,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    ranking_points INTEGER,
    PRIMARY KEY (ranking_date, player_id)
);
"""

GLICKO2_RATINGS_DDL = """
CREATE TABLE IF NOT EXISTS glicko2_ratings (
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    as_of_date TEXT NOT NULL,
    mu REAL NOT NULL,
    phi REAL NOT NULL,
    sigma REAL NOT NULL,
    PRIMARY KEY (player_id, as_of_date)
);
"""

# Phase 2 — Real Backtest infrastructure.
# Logs every WebSocket ticker update for tennis markets so we can replay
# real Kalshi prices in backtest engine instead of synthetic odds.
MARKET_TICKS_DDL = """
CREATE TABLE IF NOT EXISTS market_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    ts INTEGER NOT NULL,
    yes_bid INTEGER,
    yes_ask INTEGER,
    last_price INTEGER,
    volume INTEGER,
    received_at INTEGER NOT NULL
);
"""

INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id);",
    "CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id);",
    "CREATE INDEX IF NOT EXISTS idx_rankings_player_date ON rankings(player_id, ranking_date);",
    "CREATE INDEX IF NOT EXISTS idx_glicko2_player ON glicko2_ratings(player_id, as_of_date);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_ticker_ts ON market_ticks(ticker, ts);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_received_at ON market_ticks(received_at);",
]

ALL_DDL = [PLAYERS_DDL, MATCHES_DDL, RANKINGS_DDL, GLICKO2_RATINGS_DDL, MARKET_TICKS_DDL] + INDEXES_DDL
