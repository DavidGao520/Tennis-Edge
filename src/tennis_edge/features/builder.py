"""Feature builder: orchestrate all feature modules into a labeled dataset."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..data.db import Database
from ..ratings.tracker import RatingTracker
from .fatigue import compute_fatigue
from .form import compute_form
from .h2h import compute_h2h
from .surface import encode_surface, surface_win_rate
from .tournament import encode_tournament

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build feature vectors for match prediction.

    Canonicalizes each match so player_1 = higher-ranked (lower rank number)
    to prevent trivial ordering bias.
    """

    def __init__(self, db: Database, tracker: RatingTracker):
        self.db = db
        self.tracker = tracker

    def build_match_features(
        self,
        winner_id: int,
        loser_id: int,
        tourney_date: str,
        surface: str,
        tourney_level: str,
        round_name: str,
        best_of: int | None,
        winner_rank: int | None,
        loser_rank: int | None,
    ) -> tuple[dict[str, float], float] | None:
        """Build feature vector for a single match.

        Returns:
            (features_dict, label) where label=1.0 if p1 wins, 0.0 if p2 wins.
            Returns None if essential data is missing.
        """
        match_date_obj = date.fromisoformat(tourney_date)

        # Canonicalize: p1 = higher-ranked (lower rank number)
        r1 = winner_rank if winner_rank else 9999
        r2 = loser_rank if loser_rank else 9999

        if r1 <= r2:
            p1_id, p2_id = winner_id, loser_id
            label = 1.0  # p1 (favorite) won
        else:
            p1_id, p2_id = loser_id, winner_id
            label = 0.0  # p1 (favorite) lost

        features: dict[str, float] = {}

        # Glicko-2 ratings
        r1_rating = self.tracker.get_rating(p1_id, match_date_obj)
        r2_rating = self.tracker.get_rating(p2_id, match_date_obj)
        features["p1_mu"] = r1_rating.mu
        features["p2_mu"] = r2_rating.mu
        features["mu_diff"] = r1_rating.mu - r2_rating.mu
        features["p1_phi"] = r1_rating.phi
        features["p2_phi"] = r2_rating.phi
        features["phi_diff"] = r1_rating.phi - r2_rating.phi

        # Surface
        features.update(encode_surface(surface))
        features["p1_surface_wr"] = surface_win_rate(self.db, p1_id, surface, tourney_date)
        features["p2_surface_wr"] = surface_win_rate(self.db, p2_id, surface, tourney_date)
        features["surface_wr_diff"] = features["p1_surface_wr"] - features["p2_surface_wr"]

        # Fatigue
        fat1 = compute_fatigue(self.db, p1_id, tourney_date)
        fat2 = compute_fatigue(self.db, p2_id, tourney_date)
        for k, v in fat1.items():
            features[f"p1_{k}"] = v
        for k, v in fat2.items():
            features[f"p2_{k}"] = v
        features["rest_diff"] = fat1["days_since_last"] - fat2["days_since_last"]

        # Head-to-head
        h2h = compute_h2h(self.db, p1_id, p2_id, tourney_date)
        features.update(h2h)

        # Form
        form1 = compute_form(self.db, p1_id, tourney_date)
        form2 = compute_form(self.db, p2_id, tourney_date)
        for k, v in form1.items():
            features[f"p1_{k}"] = v
        for k, v in form2.items():
            features[f"p2_{k}"] = v
        features["form_diff_5"] = form1["win_rate_last_5"] - form2["win_rate_last_5"]
        features["form_diff_10"] = form1["win_rate_last_10"] - form2["win_rate_last_10"]

        # Tournament
        features.update(encode_tournament(tourney_level, round_name, best_of))

        return features, label

    def build_dataset(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Build labeled dataset for model training.

        Returns DataFrame with feature columns + 'label' column.
        """
        matches = self.db.query_all(
            "SELECT winner_id, loser_id, tourney_date, surface, tourney_level, "
            "       round, best_of, winner_rank, loser_rank "
            "FROM matches "
            "WHERE tourney_date >= ? AND tourney_date <= ? "
            "ORDER BY tourney_date, id",
            (start_date.isoformat(), end_date.isoformat()),
        )

        rows = []
        skipped = 0

        for m in matches:
            result = self.build_match_features(
                winner_id=m["winner_id"],
                loser_id=m["loser_id"],
                tourney_date=m["tourney_date"],
                surface=m["surface"] or "Hard",
                tourney_level=m["tourney_level"] or "D",
                round_name=m["round"] or "R32",
                best_of=m["best_of"],
                winner_rank=m["winner_rank"],
                loser_rank=m["loser_rank"],
            )

            if result is None:
                skipped += 1
                continue

            features, label = result
            features["label"] = label
            rows.append(features)

        if skipped > 0:
            logger.info("Skipped %d matches with missing data", skipped)

        df = pd.DataFrame(rows)
        logger.info("Built dataset: %d samples, %d features", len(df), len(df.columns) - 1)
        return df
