"""Real-time in-play EV scanner.

Combines:
1. Live scores (Sofascore) → current match state
2. In-play win probability model → P(win) from score state
3. Kalshi live market prices → implied probability
4. EV = model prob - market prob → trading signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .data.db import Database
from .exchange.livescore import LiveScore, match_live_to_kalshi, normalize_points
from .exchange.schemas import Market, Orderbook
from .model.inplay import InPlayModel, MatchScore, serve_prob_from_glicko
from .ratings.tracker import RatingTracker
from .strategy.kelly import edge, expected_value, fractional_kelly

logger = logging.getLogger(__name__)


@dataclass
class LiveOpportunity:
    """A live in-play trading opportunity."""
    ticker: str
    player_name: str
    opponent_name: str
    tournament: str
    round: str

    # Live score
    score_display: str  # "6-4 3-5"
    sets_p1: int
    sets_p2: int
    games_p1: int
    games_p2: int
    serving: int
    points_display: str  # "30-15"

    # Market
    market_price: float  # implied prob from Kalshi (0-1)
    yes_bid: int | None
    yes_ask: int | None

    # Model
    model_prob: float  # in-play model P(player wins)
    pre_match_prob: float  # for reference

    # EV
    edge: float
    ev_per_dollar: float
    kelly_fraction: float
    side: str  # "yes" or "no" or "pass"
    signal: str  # "STRONG", "MODERATE", "WEAK", "NONE"


class LiveEVScanner:
    """Scan live matches, compute in-play EV against Kalshi odds."""

    def __init__(
        self,
        db: Database,
        tracker: RatingTracker,
        kelly_frac: float = 0.25,
        min_edge: float = 0.05,
    ):
        self.db = db
        self.tracker = tracker
        self.kelly_frac = kelly_frac
        self.min_edge = min_edge
        self._name_cache: dict[str, int | None] = {}

    def analyze_live_match(
        self,
        live: LiveScore,
        market: Market,
        orderbook: Orderbook | None,
    ) -> LiveOpportunity | None:
        """Analyze a single live match against its Kalshi market."""

        # Extract market price
        yes_bid = orderbook.best_yes_bid if orderbook else None
        yes_ask = orderbook.best_yes_ask if orderbook else None

        if yes_bid is not None and yes_ask is not None:
            market_price = (yes_bid + yes_ask) / 200.0
        elif yes_ask is not None:
            market_price = yes_ask / 100.0
        elif market.last_price is not None:
            market_price = market.last_price
        else:
            return None

        if market_price <= 0.01 or market_price >= 0.99:
            return None

        # Determine which player this YES market is for
        title = market.title or ""
        player_name = self._extract_yes_player(title)
        if not player_name:
            return None

        # Figure out if this YES player is p1 or p2 in the live score
        p1_last = live.player1.split()[-1].lower()
        player_is_p1 = player_name.lower().split()[-1] == p1_last

        # Get Glicko-2 ratings for serve probability estimation
        p1_id = self._resolve_player(live.player1)
        p2_id = self._resolve_player(live.player2)

        if p1_id and p2_id:
            from datetime import date
            r1 = self.tracker.get_rating(p1_id, date.today())
            r2 = self.tracker.get_rating(p2_id, date.today())
            p1_serve, p2_serve = serve_prob_from_glicko(r1.mu, r2.mu)

            # Pre-match probability (for reference)
            pre_match = InPlayModel(p1_serve, p2_serve)
            pre_score = MatchScore(best_of=3)
            pre_match_prob = pre_match.win_probability(pre_score)
        else:
            p1_serve, p2_serve = 0.64, 0.64
            pre_match_prob = 0.5

        # Build current score state
        completed_sets = []
        current_set_games = (0, 0)

        for i, (g1, g2) in enumerate(live.sets):
            set_complete = (
                (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2
            ) or g1 == 7 or g2 == 7
            if set_complete:
                completed_sets.append((g1, g2))
            else:
                current_set_games = (g1, g2)

        sets_p1 = sum(1 for s in completed_sets if s[0] > s[1])
        sets_p2 = sum(1 for s in completed_sets if s[1] > s[0])

        # In-play model
        model = InPlayModel(p1_serve, p2_serve)
        score = MatchScore(
            sets_p1=sets_p1,
            sets_p2=sets_p2,
            games_p1=current_set_games[0],
            games_p2=current_set_games[1],
            points_p1=live.current_game[0],
            points_p2=live.current_game[1],
            serving=live.serving,
            best_of=3,
        )
        inplay_prob = model.win_probability(score)

        # If YES market is for p2, flip the probability
        if not player_is_p1:
            model_prob = 1.0 - inplay_prob
            pre_mp = 1.0 - pre_match_prob
        else:
            model_prob = inplay_prob
            pre_mp = pre_match_prob

        # EV calculations
        edge_val = edge(model_prob, market_price)
        ev = expected_value(model_prob, market_price)
        kelly = fractional_kelly(model_prob, market_price, self.kelly_frac)

        if kelly > 0:
            side = "yes"
        elif kelly < 0:
            side = "no"
        else:
            side = "pass"

        abs_edge = abs(edge_val)
        if abs_edge >= 0.15:
            signal = "STRONG"
        elif abs_edge >= 0.08:
            signal = "MODERATE"
        elif abs_edge >= 0.03:
            signal = "WEAK"
        else:
            signal = "NONE"

        # Score display
        score_display = " ".join(f"{s[0]}-{s[1]}" for s in live.sets)
        points_map = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}
        pt1 = points_map.get(live.current_game[0], str(live.current_game[0]))
        pt2 = points_map.get(live.current_game[1], str(live.current_game[1]))
        points_display = f"{pt1}-{pt2}" if live.status == "live" else ""

        opponent_name = live.player2 if player_is_p1 else live.player1

        return LiveOpportunity(
            ticker=market.ticker,
            player_name=player_name,
            opponent_name=opponent_name,
            tournament=live.tournament,
            round=live.round,
            score_display=score_display,
            sets_p1=sets_p1 if player_is_p1 else sets_p2,
            sets_p2=sets_p2 if player_is_p1 else sets_p1,
            games_p1=current_set_games[0] if player_is_p1 else current_set_games[1],
            games_p2=current_set_games[1] if player_is_p1 else current_set_games[0],
            serving=live.serving,
            points_display=points_display,
            market_price=market_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            model_prob=model_prob,
            pre_match_prob=pre_mp,
            edge=edge_val,
            ev_per_dollar=ev,
            kelly_fraction=abs(kelly),
            side=side,
            signal=signal,
        )

    def _extract_yes_player(self, title: str) -> str:
        """Extract player name from 'Will X win the ...' title."""
        import re
        m = re.match(r"Will (.+?) win the", title, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _resolve_player(self, name: str) -> int | None:
        if name in self._name_cache:
            return self._name_cache[name]

        parts = name.strip().split()
        if not parts:
            return None

        last = parts[-1]
        rows = self.db.query_all(
            "SELECT player_id FROM players WHERE LOWER(last_name) = ?",
            (last.lower(),),
        )
        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]

        if rows and len(parts) >= 2:
            first = parts[0]
            for row in self.db.query_all(
                "SELECT player_id, first_name FROM players WHERE LOWER(last_name) = ?",
                (last.lower(),),
            ):
                if row["first_name"] and row["first_name"].lower() == first.lower():
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

        self._name_cache[name] = None
        return None
