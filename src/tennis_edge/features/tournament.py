"""Tournament-level and round features."""

from __future__ import annotations

# Tournament level encoding (ordinal by prestige)
LEVEL_MAP = {
    "G": 5,  # Grand Slam
    "M": 4,  # Masters 1000
    "A": 3,  # ATP 500
    "D": 2,  # ATP 250
    "F": 3,  # Tour Finals (similar to 500)
}

# Round encoding (ordinal by advancement)
ROUND_MAP = {
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
    "RR": 4,  # Round-robin (similar depth to R16)
    "BR": 5,  # Bronze medal match
    "ER": 0,  # Early round / qualifying
}


def encode_tournament(
    tourney_level: str, round_name: str, best_of: int | None
) -> dict[str, float]:
    """Encode tournament level and round as features."""
    return {
        "tourney_level": float(LEVEL_MAP.get(tourney_level, 2)),
        "round_depth": float(ROUND_MAP.get(round_name, 3)),
        "best_of_5": 1.0 if best_of == 5 else 0.0,
    }
