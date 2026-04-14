"""Tests for feature engineering modules."""

from tennis_edge.features.surface import encode_surface
from tennis_edge.features.tournament import encode_tournament
from tennis_edge.features.h2h import compute_h2h
from tennis_edge.features.form import compute_form
from tennis_edge.features.fatigue import compute_fatigue


def test_surface_encoding():
    result = encode_surface("Hard")
    assert result["surface_hard"] == 1.0
    assert result["surface_clay"] == 0.0
    assert result["surface_grass"] == 0.0


def test_tournament_encoding():
    result = encode_tournament("G", "F", 5)
    assert result["tourney_level"] == 5.0  # Grand Slam
    assert result["round_depth"] == 7.0  # Final
    assert result["best_of_5"] == 1.0


def test_h2h_no_prior(in_memory_db, sample_players):
    result = compute_h2h(in_memory_db, 1, 4, "2020-01-01")
    assert result["h2h_win_rate_p1"] == 0.5  # No prior meetings


def test_h2h_with_history(in_memory_db, sample_matches):
    # Player 1 (Djokovic) beat Player 2 (Nadal) on 2023-01-15
    result = compute_h2h(in_memory_db, 1, 2, "2023-12-01")
    assert result["h2h_wins_p1"] >= 1


def test_form_no_matches(in_memory_db, sample_players):
    result = compute_form(in_memory_db, 1, "2020-01-01")
    assert result["win_rate_last_5"] == 0.5  # Default


def test_fatigue_no_matches(in_memory_db, sample_players):
    result = compute_fatigue(in_memory_db, 1, "2020-01-01")
    assert result["days_since_last"] == 30  # Default
    assert result["matches_last_7d"] == 0.0
