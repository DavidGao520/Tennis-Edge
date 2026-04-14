"""Track and manage Glicko-2 ratings over match history."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

from ..data.db import Database
from .glicko2 import Glicko2Engine, Glicko2Rating

logger = logging.getLogger(__name__)


class RatingTracker:
    """Process match history chronologically and store rating snapshots."""

    def __init__(self, db: Database, engine: Glicko2Engine, period_days: int = 30):
        self.db = db
        self.engine = engine
        self.period_days = period_days
        self._cache: dict[int, Glicko2Rating] = {}

    def compute_all_ratings(self) -> tuple[int, int]:
        """Walk through all matches, update ratings period by period.

        Returns:
            (number_of_periods, number_of_unique_players_rated)
        """
        # Clear existing ratings
        self.db.execute("DELETE FROM glicko2_ratings")
        self.db.commit()
        self._cache.clear()

        # Get all matches ordered chronologically
        matches = self.db.query_all(
            "SELECT tourney_date, winner_id, loser_id "
            "FROM matches ORDER BY tourney_date, id"
        )

        if not matches:
            logger.warning("No matches found in database")
            return 0, 0

        # Group matches into rating periods
        periods = self._group_into_periods(matches)
        all_rated_players: set[int] = set()

        for period_end, period_matches in periods:
            # Collect results per player
            player_results: dict[int, list[tuple[int, float]]] = defaultdict(list)

            for m in period_matches:
                w_id = m["winner_id"]
                l_id = m["loser_id"]
                player_results[w_id].append((l_id, 1.0))
                player_results[l_id].append((w_id, 0.0))

            # Update ratings for all players who played
            for player_id, results in player_results.items():
                current = self._get_rating(player_id)
                opponents = [self._get_rating(opp_id) for opp_id, _ in results]
                outcomes = [outcome for _, outcome in results]

                new_rating = self.engine.rate(current, opponents, outcomes)
                self._cache[player_id] = new_rating
                all_rated_players.add(player_id)

            # Store snapshots for this period
            self._store_period_ratings(period_end, player_results.keys())

        logger.info(
            "Computed ratings across %d periods for %d players",
            len(periods), len(all_rated_players),
        )
        return len(periods), len(all_rated_players)

    def get_rating(self, player_id: int, as_of: date | None = None) -> Glicko2Rating:
        """Look up the most recent rating for a player as of a date."""
        if as_of is None:
            return self._get_rating(player_id)

        row = self.db.query_one(
            "SELECT mu, phi, sigma FROM glicko2_ratings "
            "WHERE player_id = ? AND as_of_date <= ? "
            "ORDER BY as_of_date DESC LIMIT 1",
            (player_id, as_of.isoformat()),
        )

        if row:
            return Glicko2Rating(mu=row["mu"], phi=row["phi"], sigma=row["sigma"])
        return self.engine.new_rating()

    def get_rating_history(self, player_id: int) -> list[tuple[str, Glicko2Rating]]:
        """Full rating trajectory for a player."""
        rows = self.db.query_all(
            "SELECT as_of_date, mu, phi, sigma FROM glicko2_ratings "
            "WHERE player_id = ? ORDER BY as_of_date",
            (player_id,),
        )
        return [
            (row["as_of_date"], Glicko2Rating(row["mu"], row["phi"], row["sigma"]))
            for row in rows
        ]

    # --- Internal ---

    def _get_rating(self, player_id: int) -> Glicko2Rating:
        if player_id in self._cache:
            return self._cache[player_id]
        return self.engine.new_rating()

    def _group_into_periods(
        self, matches: list
    ) -> list[tuple[date, list]]:
        """Group matches into fixed-length rating periods."""
        if not matches:
            return []

        first_date = date.fromisoformat(matches[0]["tourney_date"])
        periods: list[tuple[date, list]] = []
        current_period_end = first_date + timedelta(days=self.period_days)
        current_matches: list = []

        for m in matches:
            m_date = date.fromisoformat(m["tourney_date"])
            if m_date <= current_period_end:
                current_matches.append(m)
            else:
                if current_matches:
                    periods.append((current_period_end, current_matches))
                current_matches = [m]
                # Advance period end to cover this match
                while current_period_end < m_date:
                    current_period_end += timedelta(days=self.period_days)

        if current_matches:
            periods.append((current_period_end, current_matches))

        return periods

    def _store_period_ratings(self, period_end: date, player_ids) -> None:
        """Write rating snapshots for all players who played in this period."""
        rows = []
        for pid in player_ids:
            r = self._cache.get(pid)
            if r:
                rows.append((pid, period_end.isoformat(), r.mu, r.phi, r.sigma))

        self.db.executemany(
            "INSERT OR REPLACE INTO glicko2_ratings (player_id, as_of_date, mu, phi, sigma) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.db.commit()
