"""Live EV scanner: scan Kalshi → match players → model predict → EV/Kelly."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .data.db import Database
from .exchange.schemas import Market, Orderbook
from .features.builder import FeatureBuilder
from .model.predictor import MatchPredictor
from .ratings.tracker import RatingTracker
from .strategy.kelly import edge, expected_value, fractional_kelly
from .strategy.sizing import PositionSizer, BetDecision

logger = logging.getLogger(__name__)

# Parse market title: "Will Carlos Alcaraz win the Alcaraz vs Virtanen: Round Of 32 match?"
TITLE_RE = re.compile(
    r"Will (.+?) win the .+? vs .+?: (.+?) match\?", re.IGNORECASE
)
VERSUS_RE = re.compile(
    r"(.+?) vs\.? (.+?)(?:\s*:\s*(.+?))?(?:\s+match\??)?$", re.IGNORECASE
)

# Parse ticker for player codes and surface clues
TICKER_RE = re.compile(
    r"KX(ATP|WTA)(CHALLENGER)?MATCH-(\d{2})([A-Z]{3})(\d{2})(.+)-([A-Z]{2,5})$"
)


@dataclass
class Opportunity:
    """A potential trading opportunity."""
    ticker: str
    player_name: str  # player this YES market is for
    opponent_name: str
    match_title: str
    round_info: str
    category: str  # ATP Main, ATP Challenger, etc.

    # Market data
    yes_bid: int | None  # cents
    yes_ask: int | None  # cents
    mid_price: float | None  # cents
    market_implied_prob: float  # from mid or ask

    # Model data
    model_prob: float  # model's estimated prob this player wins
    model_confidence: float  # abs(model_prob - 0.5)

    # EV metrics
    edge: float  # model_prob - market_prob
    ev_per_dollar: float  # expected value per dollar risked
    kelly_fraction: float  # fractional Kelly bet size
    recommended_side: str  # "yes" or "no" or "pass"

    # Player IDs for reference
    player_id: int | None = None
    opponent_id: int | None = None

    @property
    def signal_strength(self) -> str:
        ae = abs(self.edge)
        if ae >= 0.15:
            return "STRONG"
        elif ae >= 0.08:
            return "MODERATE"
        elif ae >= 0.03:
            return "WEAK"
        return "NONE"


class EVScanner:
    """Scan live Kalshi markets and calculate EV using the trained model."""

    def __init__(
        self,
        db: Database,
        tracker: RatingTracker,
        builder: FeatureBuilder,
        model: MatchPredictor,
        sizer: PositionSizer,
        kelly_frac: float = 0.25,
    ):
        self.db = db
        self.tracker = tracker
        self.builder = builder
        self.model = model
        self.sizer = sizer
        self.kelly_frac = kelly_frac
        self._name_cache: dict[str, int | None] = {}

    def analyze_market_pair(
        self,
        market_yes: Market,
        market_no: Market | None,
        orderbook_yes: Orderbook | None,
        orderbook_no: Orderbook | None,
    ) -> Opportunity | None:
        """Analyze a pair of markets (YES/NO for one player) and return opportunity."""

        # Extract player names from title
        title = market_yes.title or ""
        player_name, opponent_name, round_info = self._parse_title(title)
        if not player_name or not opponent_name:
            return None

        # Determine market price
        yes_bid = orderbook_yes.best_yes_bid if orderbook_yes else None
        yes_ask = orderbook_yes.best_yes_ask if orderbook_yes else None

        if yes_bid is not None and yes_ask is not None:
            mid = (yes_bid + yes_ask) / 2.0
            market_prob = mid / 100.0
        elif yes_ask is not None:
            market_prob = yes_ask / 100.0
            mid = float(yes_ask)
        elif yes_bid is not None:
            market_prob = yes_bid / 100.0
            mid = float(yes_bid)
        elif market_yes.last_price is not None:
            market_prob = market_yes.last_price
            mid = market_yes.last_price * 100
            yes_bid = None
            yes_ask = None
        else:
            return None  # No price data at all

        if market_prob <= 0.01 or market_prob >= 0.99:
            return None

        # Resolve players to DB IDs
        player_id = self._resolve_player(player_name)
        opponent_id = self._resolve_player(opponent_name)

        if player_id is None or opponent_id is None:
            logger.debug("Could not resolve: %s (%s) or %s (%s)",
                        player_name, player_id, opponent_name, opponent_id)
            return None

        # Get model prediction
        model_prob = self._predict_match(player_id, opponent_id)
        if model_prob is None:
            return None

        # Calculate EV metrics
        edge_val = edge(model_prob, market_prob)
        ev = expected_value(model_prob, market_prob)
        kelly = fractional_kelly(model_prob, market_prob, self.kelly_frac)

        if kelly > 0:
            side = "yes"
        elif kelly < 0:
            side = "no"
        else:
            side = "pass"

        # Categorize
        ticker = market_yes.ticker
        if "KXATPCHALLENGER" in ticker:
            category = "ATP Challenger"
        elif "KXATPMATCH" in ticker:
            category = "ATP Main"
        elif "KXWTACHALLENGER" in ticker:
            category = "WTA Challenger"
        elif "KXWTAMATCH" in ticker:
            category = "WTA Main"
        else:
            category = "Other"

        return Opportunity(
            ticker=ticker,
            player_name=player_name,
            opponent_name=opponent_name,
            match_title=title,
            round_info=round_info,
            category=category,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=mid,
            market_implied_prob=market_prob,
            model_prob=model_prob,
            model_confidence=abs(model_prob - 0.5),
            edge=edge_val,
            ev_per_dollar=ev,
            kelly_fraction=abs(kelly),
            recommended_side=side,
            player_id=player_id,
            opponent_id=opponent_id,
        )

    def _parse_title(self, title: str) -> tuple[str, str, str]:
        """Extract player name, opponent name, round from market title."""
        # "Will Carlos Alcaraz win the Alcaraz vs Virtanen: Round Of 32 match?"
        m = TITLE_RE.match(title)
        if m:
            player = m.group(1).strip()
            round_info = m.group(2).strip()
            # Extract opponent from "X vs Y" in the title
            vs_match = re.search(r"the\s+(.+?)\s+vs\s+(.+?):", title, re.IGNORECASE)
            if vs_match:
                name1 = vs_match.group(1).strip()
                name2 = vs_match.group(2).strip()
                # Player is one of them, opponent is the other
                if self._name_similar(player, name1):
                    opponent = name2
                else:
                    opponent = name1
                return player, opponent, round_info

        # Fallback: try to parse "X vs Y" from title
        vs = re.search(r"(\w[\w\s'-]+?)\s+vs\.?\s+(\w[\w\s'-]+?)(?:\s*:|$)", title)
        if vs:
            return vs.group(1).strip(), vs.group(2).strip(), ""

        return "", "", ""

    def _name_similar(self, full_name: str, short_name: str) -> bool:
        """Check if names refer to same person."""
        fn = full_name.lower().replace("-", " ").split()
        sn = short_name.lower().replace("-", " ").split()
        if not fn or not sn:
            return False
        return fn[-1] == sn[-1] or fn[0] == sn[0]

    def _resolve_player(self, name: str) -> int | None:
        """Resolve player name to DB ID."""
        if name in self._name_cache:
            return self._name_cache[name]

        parts = name.strip().split()
        if not parts:
            return None

        last = parts[-1]
        first = parts[0] if len(parts) > 1 else ""

        # Try exact last name
        rows = self.db.query_all(
            "SELECT player_id, first_name, last_name FROM players "
            "WHERE LOWER(last_name) = ?",
            (last.lower(),),
        )

        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]

        # Multiple matches: try first name
        if rows and first:
            for row in rows:
                if row["first_name"] and row["first_name"].lower() == first.lower():
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]
                if row["first_name"] and row["first_name"][0].lower() == first[0].lower():
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

        # LIKE fallback
        rows = self.db.query_all(
            "SELECT player_id, first_name, last_name FROM players "
            "WHERE LOWER(last_name) LIKE ? ORDER BY player_id LIMIT 5",
            (f"%{last.lower()}%",),
        )
        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]
        if rows and first:
            for row in rows:
                if row["first_name"] and row["first_name"].lower().startswith(first[:3].lower()):
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

        self._name_cache[name] = None
        return None

    def _predict_match(self, p1_id: int, p2_id: int) -> float | None:
        """Get model probability that p1 beats p2.

        Note: the model predicts P(higher-ranked player wins).
        We need to figure out which player is higher-ranked and adjust.
        """
        today = date.today().isoformat()

        # Get rankings to determine ordering
        r1 = self.db.query_one(
            "SELECT ranking FROM rankings WHERE player_id = ? ORDER BY ranking_date DESC LIMIT 1",
            (p1_id,),
        )
        r2 = self.db.query_one(
            "SELECT ranking FROM rankings WHERE player_id = ? ORDER BY ranking_date DESC LIMIT 1",
            (p2_id,),
        )

        rank1 = r1["ranking"] if r1 else 9999
        rank2 = r2["ranking"] if r2 else 9999

        # For the model: winner_id/loser_id don't matter for feature building
        # since it canonicalizes by rank. We just need to know the mapping.
        # Assume p1 wins (arbitrary), the model will figure out the features.
        feat_result = self.builder.build_match_features(
            winner_id=p1_id,
            loser_id=p2_id,
            tourney_date=today,
            surface="Hard",  # default, ideally from market metadata
            tourney_level="M",
            round_name="R32",
            best_of=3,
            winner_rank=rank1,
            loser_rank=rank2,
        )

        if feat_result is None:
            return None

        features, label = feat_result
        feature_df = pd.DataFrame([features])
        for col in self.model.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[self.model.feature_names].fillna(0)

        model_prob_p1_wins = float(self.model.predict_proba(feature_df)[0])

        # label=1 means p1 (higher-ranked) wins
        # If p1 (our query player) IS the higher-ranked one, model_prob = P(p1 wins)
        # If p1 is lower-ranked, model_prob = P(higher-ranked wins) = P(p2 wins), so P(p1) = 1 - model_prob
        if rank1 <= rank2:
            return model_prob_p1_wins  # p1 is favorite
        else:
            return 1.0 - model_prob_p1_wins  # p1 is underdog
