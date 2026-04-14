"""Shared test fixtures."""

import pytest
from tennis_edge.data.db import Database
from tennis_edge.ratings.glicko2 import Glicko2Engine, Glicko2Rating


@pytest.fixture
def in_memory_db():
    """In-memory SQLite database with schema initialized."""
    db = Database(":memory:")
    db.connect()
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def glicko2_engine():
    return Glicko2Engine(tau=0.5)


@pytest.fixture
def sample_players(in_memory_db):
    """Insert sample players into the database."""
    players = [
        (1, "Novak", "Djokovic", "R", "1987-05-22", "SRB", 188),
        (2, "Rafael", "Nadal", "L", "1986-06-03", "ESP", 185),
        (3, "Roger", "Federer", "R", "1981-08-08", "SUI", 185),
        (4, "Carlos", "Alcaraz", "R", "2003-05-05", "ESP", 183),
    ]
    in_memory_db.executemany(
        "INSERT INTO players (player_id, first_name, last_name, hand, birth_date, country_code, height_cm) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        players,
    )
    in_memory_db.commit()
    return players


@pytest.fixture
def sample_matches(in_memory_db, sample_players):
    """Insert sample matches."""
    matches = [
        ("2023-T001", "Test Open", "Hard", 32, "A", "2023-01-15", 1, 1, 2, "6-3 6-4", 3, "F", 95, 1, 2, None, None),
        ("2023-T001", "Test Open", "Hard", 32, "A", "2023-01-14", 2, 1, 3, "7-5 6-3", 3, "SF", 110, 1, 3, None, None),
        ("2023-T001", "Test Open", "Hard", 32, "A", "2023-01-14", 3, 2, 3, "6-4 6-7 6-3", 3, "SF", 145, 2, 3, None, None),
        ("2023-T002", "Clay Classic", "Clay", 32, "M", "2023-05-20", 1, 2, 1, "6-2 6-4", 3, "F", 100, 2, 1, None, None),
        ("2023-T002", "Clay Classic", "Clay", 32, "M", "2023-05-19", 2, 2, 4, "6-3 6-1", 3, "SF", 75, 2, 4, None, None),
    ]

    null_stats = tuple([None] * 18)  # 18 stat columns
    in_memory_db.executemany(
        "INSERT INTO matches (tourney_id, tourney_name, surface, draw_size, tourney_level, "
        "tourney_date, match_num, winner_id, loser_id, score, best_of, round, minutes, "
        "winner_rank, loser_rank, winner_seed, loser_seed, "
        "w_ace, w_df, w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_sv_gms, w_bp_saved, w_bp_faced, "
        "l_ace, l_df, l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_sv_gms, l_bp_saved, l_bp_faced) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?" + ",?" * 18 + ")",
        [m + null_stats for m in matches],
    )
    in_memory_db.commit()
    return matches
