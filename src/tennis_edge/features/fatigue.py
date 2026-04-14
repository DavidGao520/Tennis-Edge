"""Fatigue-related features: recent match load and rest days."""

from __future__ import annotations

from ..data.db import Database


def compute_fatigue(db: Database, player_id: int, match_date: str) -> dict[str, float]:
    """Compute fatigue features for a player before a specific match date.

    Returns:
        days_since_last: Days since player's last match (capped at 90, default 30)
        matches_last_7d: Number of matches in last 7 days
        matches_last_14d: Number of matches in last 14 days
        matches_last_30d: Number of matches in last 30 days
    """
    # Days since last match
    last_match = db.query_one(
        "SELECT MAX(tourney_date) as last_date FROM matches "
        "WHERE (winner_id = ? OR loser_id = ?) AND tourney_date < ?",
        (player_id, player_id, match_date),
    )

    if last_match and last_match["last_date"]:
        from datetime import date
        ld = date.fromisoformat(last_match["last_date"])
        md = date.fromisoformat(match_date)
        days_since = (md - ld).days
    else:
        days_since = 30  # default for unknown

    days_since = min(days_since, 90)  # cap outliers

    # Recent match counts
    counts = {}
    for window, label in [(7, "7d"), (14, "14d"), (30, "30d")]:
        from datetime import date, timedelta
        md = date.fromisoformat(match_date)
        start = (md - timedelta(days=window)).isoformat()
        row = db.query_one(
            "SELECT COUNT(*) as cnt FROM matches "
            "WHERE (winner_id = ? OR loser_id = ?) "
            "  AND tourney_date >= ? AND tourney_date < ?",
            (player_id, player_id, start, match_date),
        )
        counts[f"matches_last_{label}"] = float(row["cnt"]) if row else 0.0

    return {
        "days_since_last": float(days_since),
        **counts,
    }
