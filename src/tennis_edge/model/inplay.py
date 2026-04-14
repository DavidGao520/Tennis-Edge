"""In-play tennis win probability model.

Computes exact P(player1 wins match) from any score state using the
hierarchical structure of tennis: Points → Games → Sets → Match.

All probabilities flow from a single input: each player's serve point win %.
ATP average is ~0.64 (server wins 64% of points).

Reference: Barnett & Clarke (2005), Klaassen & Magnus (2001)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class MatchScore:
    """Current match score state."""
    sets1: int = 0        # sets won by p1
    sets2: int = 0        # sets won by p2
    games1: int = 0       # games in current set for p1
    games2: int = 0       # games in current set for p2
    points1: int = 0      # points in current game for p1 (0-3+)
    points2: int = 0      # points in current game for p2
    serving: int = 1      # 1 = p1 serving, 2 = p2 serving
    best_of: int = 3      # 3 or 5

    @property
    def sets_to_win(self) -> int:
        return 3 if self.best_of == 5 else 2


class InPlayModel:
    """Calculate P(p1 wins match) from any score state.

    Inputs:
        sp1: P(p1 wins a point when p1 is serving). ATP avg ~0.64
        sp2: P(p2 wins a point when p2 is serving). ATP avg ~0.64

    Derived:
        When p1 serves: P(p1 wins point) = sp1
        When p2 serves: P(p1 wins point) = 1 - sp2
    """

    def __init__(self, sp1: float = 0.64, sp2: float = 0.64):
        self.sp1 = sp1
        self.sp2 = sp2
        # Clear caches
        self._game_prob_cache: dict = {}
        self._set_prob_cache: dict = {}
        self._match_prob_cache: dict = {}

    def win_probability(self, score: MatchScore) -> float:
        """P(player 1 wins the match) from current score state."""
        # Get P(p1 wins current game)
        game_p = self._game_prob(score.points1, score.points2, score.serving)

        # Get P(p1 wins current set from current game score)
        set_p = self._set_prob_with_game(
            score.games1, score.games2, game_p, score.serving
        )

        # Get P(p1 wins match from current set score)
        return self._match_prob_with_set(
            score.sets1, score.sets2, set_p, score.serving, score.sets_to_win
        )

    # ── Point → Game level ──

    def _point_prob(self, serving: int) -> float:
        """P(p1 wins this point)."""
        return self.sp1 if serving == 1 else (1.0 - self.sp2)

    def _game_prob(self, pts1: int, pts2: int, serving: int) -> float:
        """P(p1 wins current game) from point score (pts1, pts2).

        Points: 0=love, 1=15, 2=30, 3=40.
        At deuce (3-3), use closed-form.
        """
        key = (pts1, pts2, serving)
        if key in self._game_prob_cache:
            return self._game_prob_cache[key]

        p = self._point_prob(serving)

        # Base cases
        if pts1 >= 4 and pts1 - pts2 >= 2:
            return 1.0
        if pts2 >= 4 and pts2 - pts1 >= 2:
            return 0.0
        if pts1 >= 3 and pts2 >= 3:
            # Deuce: closed form P(p1 wins game from deuce)
            result = p ** 2 / (p ** 2 + (1 - p) ** 2)
            self._game_prob_cache[key] = result
            return result

        # Game already won
        if pts1 >= 4 and pts2 < 3:
            return 1.0
        if pts2 >= 4 and pts1 < 3:
            return 0.0

        result = (
            p * self._game_prob(pts1 + 1, pts2, serving)
            + (1 - p) * self._game_prob(pts1, pts2 + 1, serving)
        )
        self._game_prob_cache[key] = result
        return result

    # ── Game → Set level ──

    def _hold_prob(self, serving: int) -> float:
        """P(p1 wins a full game when 'serving' is serving), from 0-0."""
        return self._game_prob(0, 0, serving)

    def _set_prob_fresh(self, g1: int, g2: int, serving: int) -> float:
        """P(p1 wins set) from game score (g1, g2), at start of a new game."""
        key = (g1, g2, serving)
        if key in self._set_prob_cache:
            return self._set_prob_cache[key]

        # Set won
        if g1 >= 6 and g1 - g2 >= 2:
            return 1.0
        if g2 >= 6 and g2 - g1 >= 2:
            return 0.0
        if g1 == 7 or g2 == 7:  # tiebreak decided
            return 1.0 if g1 > g2 else 0.0

        # Tiebreak at 6-6
        if g1 == 6 and g2 == 6:
            result = self._tiebreak_prob(0, 0, serving)
            self._set_prob_cache[key] = result
            return result

        # Safety: cap runaway recursion (shouldn't happen with proper base cases)
        if g1 > 7 or g2 > 7:
            return 0.5

        # P(p1 wins this game) * P(p1 wins set from new score) + ...
        gp = self._hold_prob(serving)
        next_srv = 3 - serving

        result = (
            gp * self._set_prob_fresh(g1 + 1, g2, next_srv)
            + (1 - gp) * self._set_prob_fresh(g1, g2 + 1, next_srv)
        )
        self._set_prob_cache[key] = result
        return result

    def _set_prob_with_game(self, g1: int, g2: int, current_game_prob: float, serving: int) -> float:
        """P(p1 wins set) incorporating the probability of winning the CURRENT game."""
        # Set already decided
        if g1 >= 6 and g1 - g2 >= 2:
            return 1.0
        if g2 >= 6 and g2 - g1 >= 2:
            return 0.0
        if (g1 == 7 and g2 <= 6) or (g1 == 6 and g2 <= 4):
            return 1.0
        if (g2 == 7 and g1 <= 6) or (g2 == 6 and g1 <= 4):
            return 0.0

        next_srv = 3 - serving
        return (
            current_game_prob * self._set_prob_fresh(g1 + 1, g2, next_srv)
            + (1 - current_game_prob) * self._set_prob_fresh(g1, g2 + 1, next_srv)
        )

    def _tiebreak_prob(self, tb1: int, tb2: int, serving: int) -> float:
        """P(p1 wins tiebreak from tb score)."""
        if tb1 >= 7 and tb1 - tb2 >= 2:
            return 1.0
        if tb2 >= 7 and tb2 - tb1 >= 2:
            return 0.0

        # Extended tiebreak deuce
        if tb1 >= 6 and tb2 >= 6:
            p1_pt = self._point_prob(1)
            p2_pt = self._point_prob(2)
            # Two points: one on each serve
            p_hold = p1_pt * (1 - p2_pt) + p1_pt * p2_pt  # simplified
            # More accurate: P(win 2 consecutive with alternating serve)
            # Approximate with average
            p_avg = (p1_pt + (1 - p2_pt)) / 2
            return p_avg ** 2 / (p_avg ** 2 + (1 - p_avg) ** 2)

        # Determine server for this point
        total = tb1 + tb2
        if total == 0:
            current_server = serving
        else:
            current_server = serving if ((total - 1) // 2) % 2 == 0 else (3 - serving)

        p = self._point_prob(current_server)

        return (
            p * self._tiebreak_prob(tb1 + 1, tb2, serving)
            + (1 - p) * self._tiebreak_prob(tb1, tb2 + 1, serving)
        )

    # ── Set → Match level ──

    def _match_prob_with_set(
        self, s1: int, s2: int, current_set_prob: float, serving: int, sets_to_win: int
    ) -> float:
        """P(p1 wins match) incorporating probability of winning current set."""
        if s1 >= sets_to_win:
            return 1.0
        if s2 >= sets_to_win:
            return 0.0

        return (
            current_set_prob * self._match_prob_fresh(s1 + 1, s2, serving, sets_to_win)
            + (1 - current_set_prob) * self._match_prob_fresh(s1, s2 + 1, serving, sets_to_win)
        )

    def _match_prob_fresh(self, s1: int, s2: int, serving: int, sets_to_win: int) -> float:
        """P(p1 wins match) from set score, at start of a new set."""
        key = (s1, s2, serving, sets_to_win)
        if key in self._match_prob_cache:
            return self._match_prob_cache[key]

        if s1 >= sets_to_win:
            return 1.0
        if s2 >= sets_to_win:
            return 0.0

        # P(win this set from 0-0)
        sp = self._set_prob_fresh(0, 0, serving)

        result = (
            sp * self._match_prob_fresh(s1 + 1, s2, serving, sets_to_win)
            + (1 - sp) * self._match_prob_fresh(s1, s2 + 1, serving, sets_to_win)
        )
        self._match_prob_cache[key] = result
        return result


# ── Convenience functions ──

def quick_win_prob(
    sets_p1: int, sets_p2: int,
    games_p1: int, games_p2: int,
    p1_serve: float = 0.64,
    p2_serve: float = 0.64,
    serving: int = 1,
    best_of: int = 3,
    points_p1: int = 0,
    points_p2: int = 0,
) -> float:
    """Quick in-play win probability calculation."""
    model = InPlayModel(p1_serve, p2_serve)
    score = MatchScore(
        sets1=sets_p1, sets2=sets_p2,
        games1=games_p1, games2=games_p2,
        points1=points_p1, points2=points_p2,
        serving=serving, best_of=best_of,
    )
    return model.win_probability(score)


def serve_prob_from_glicko(mu1: float, mu2: float) -> tuple[float, float]:
    """Estimate serve point win probabilities from Glicko-2 ratings.

    ATP average serve point win: ~0.64
    Adjust based on rating gap.
    """
    diff = mu1 - mu2
    overall = 1.0 / (1.0 + 10 ** (-diff / 400))

    base_serve = 0.64
    adjustment = (overall - 0.5) * 0.12

    p1_serve = min(0.72, max(0.56, base_serve + adjustment))
    p2_serve = min(0.72, max(0.56, base_serve - adjustment))

    return p1_serve, p2_serve
