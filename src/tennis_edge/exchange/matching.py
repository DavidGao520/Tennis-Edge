"""Match Kalshi market titles to ATP player IDs."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from ..data.db import Database

logger = logging.getLogger(__name__)

# Patterns to extract player names from market titles
VERSUS_PATTERN = re.compile(
    r"(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\s*[-–—]\s*|\s*$)", re.IGNORECASE
)
WILL_WIN_PATTERN = re.compile(
    r"Will\s+(.+?)\s+win\s+", re.IGNORECASE
)


class PlayerMatcher:
    """Match Kalshi market titles to ATP player IDs via fuzzy name matching."""

    def __init__(self, db: Database):
        self.db = db
        self._name_cache: dict[str, int | None] = {}

    def match_market(self, title: str) -> tuple[int, int] | None:
        """Extract two player IDs from a market title.

        Returns (player1_id, player2_id) or None.
        """
        names = self._extract_names(title)
        if not names or len(names) < 2:
            return None

        p1 = self._resolve_name(names[0])
        p2 = self._resolve_name(names[1])

        if p1 is None or p2 is None:
            logger.debug("Could not resolve: %s -> %s, %s -> %s", names[0], p1, names[1], p2)
            return None

        return (p1, p2)

    def _extract_names(self, title: str) -> list[str]:
        """Extract player name strings from market title."""
        # Try "X vs Y" pattern
        m = VERSUS_PATTERN.search(title)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]

        # Try "Will X win" pattern (only gets one name)
        m = WILL_WIN_PATTERN.search(title)
        if m:
            return [m.group(1).strip()]

        return []

    def _resolve_name(self, name: str) -> int | None:
        """Resolve a player name to their database ID."""
        if name in self._name_cache:
            return self._name_cache[name]

        # Clean the name
        name_clean = name.strip().rstrip(".")
        parts = name_clean.split()

        if not parts:
            return None

        # Try exact last name match first
        last_name = parts[-1]
        rows = self.db.query_all(
            "SELECT player_id, first_name, last_name FROM players "
            "WHERE LOWER(last_name) = ?",
            (last_name.lower(),),
        )

        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]

        # If multiple matches, try first name too
        if len(rows) > 1 and len(parts) >= 2:
            first_initial = parts[0][0].lower() if parts[0] else ""
            for row in rows:
                if row["first_name"] and row["first_name"][0].lower() == first_initial:
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

            # Try full first name match
            first_name = parts[0]
            for row in rows:
                if row["first_name"] and row["first_name"].lower() == first_name.lower():
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

        # Try LIKE search as fallback
        if len(rows) == 0:
            rows = self.db.query_all(
                "SELECT player_id, first_name, last_name FROM players "
                "WHERE LOWER(last_name) LIKE ?",
                (f"%{last_name.lower()}%",),
            )
            if len(rows) == 1:
                self._name_cache[name] = rows[0]["player_id"]
                return rows[0]["player_id"]

        self._name_cache[name] = None
        return None
