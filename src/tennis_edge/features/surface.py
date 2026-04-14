"""Surface-related features."""

from __future__ import annotations

from ..data.db import Database

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]


def encode_surface(surface: str) -> dict[str, float]:
    """One-hot encode the surface type."""
    result = {}
    for s in SURFACES:
        result[f"surface_{s.lower()}"] = 1.0 if surface == s else 0.0
    return result


def surface_win_rate(
    db: Database, player_id: int, surface: str, before_date: str
) -> float:
    """Player's historical win rate on a specific surface before a date."""
    row = db.query_one(
        "SELECT "
        "  SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins, "
        "  COUNT(*) as total "
        "FROM matches "
        "WHERE (winner_id = ? OR loser_id = ?) "
        "  AND surface = ? AND tourney_date < ?",
        (player_id, player_id, player_id, surface, before_date),
    )
    if not row or not row["total"] or row["total"] == 0:
        return 0.5
    return row["wins"] / row["total"]
