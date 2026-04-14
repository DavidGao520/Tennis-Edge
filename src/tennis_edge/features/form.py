"""Recent form features: win rates over sliding windows."""

from __future__ import annotations

from ..data.db import Database


def compute_form(db: Database, player_id: int, before_date: str) -> dict[str, float]:
    """Compute recent form features for a player.

    Returns:
        win_rate_last_5: Win rate in last 5 matches
        win_rate_last_10: Win rate in last 10 matches
        win_rate_last_20: Win rate in last 20 matches
    """
    result = {}

    for window in [5, 10, 20]:
        rows = db.query_all(
            "SELECT winner_id FROM matches "
            "WHERE (winner_id = ? OR loser_id = ?) AND tourney_date < ? "
            "ORDER BY tourney_date DESC, id DESC LIMIT ?",
            (player_id, player_id, before_date, window),
        )

        if not rows:
            result[f"win_rate_last_{window}"] = 0.5
            continue

        wins = sum(1 for r in rows if r["winner_id"] == player_id)
        result[f"win_rate_last_{window}"] = wins / len(rows)

    return result
