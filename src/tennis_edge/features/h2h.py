"""Head-to-head record features."""

from __future__ import annotations

from ..data.db import Database


def compute_h2h(
    db: Database, player1_id: int, player2_id: int, before_date: str
) -> dict[str, float]:
    """Compute head-to-head features between two players before a date.

    Returns:
        h2h_wins_p1: Number of wins by player 1 vs player 2
        h2h_wins_p2: Number of wins by player 2 vs player 1
        h2h_win_rate_p1: Win rate of player 1 (0.5 if no prior meetings)
    """
    row = db.query_one(
        "SELECT "
        "  SUM(CASE WHEN winner_id = ? AND loser_id = ? THEN 1 ELSE 0 END) as p1_wins, "
        "  SUM(CASE WHEN winner_id = ? AND loser_id = ? THEN 1 ELSE 0 END) as p2_wins "
        "FROM matches "
        "WHERE ((winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)) "
        "  AND tourney_date < ?",
        (
            player1_id, player2_id,
            player2_id, player1_id,
            player1_id, player2_id,
            player2_id, player1_id,
            before_date,
        ),
    )

    p1_wins = float(row["p1_wins"] or 0) if row else 0.0
    p2_wins = float(row["p2_wins"] or 0) if row else 0.0
    total = p1_wins + p2_wins

    return {
        "h2h_wins_p1": p1_wins,
        "h2h_wins_p2": p2_wins,
        "h2h_win_rate_p1": p1_wins / total if total > 0 else 0.5,
    }
