"""Domain dataclasses for tennis data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Player:
    player_id: int
    first_name: str
    last_name: str
    hand: str  # 'R', 'L', 'U'
    birth_date: date | None
    country_code: str
    height_cm: int | None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(frozen=True, slots=True)
class Match:
    tourney_id: str
    tourney_name: str
    tourney_date: date
    tourney_level: str  # G=Grand Slam, M=Masters, A=ATP500, D=ATP250, F=Davis/Finals
    surface: str  # Hard, Clay, Grass, Carpet
    round: str  # F, SF, QF, R16, R32, R64, R128, RR
    best_of: int
    winner_id: int
    loser_id: int
    score: str
    minutes: int | None
    winner_rank: int | None
    loser_rank: int | None
    w_ace: int | None = None
    w_df: int | None = None
    w_svpt: int | None = None
    w_1st_in: int | None = None
    w_1st_won: int | None = None
    w_2nd_won: int | None = None
    w_sv_gms: int | None = None
    w_bp_saved: int | None = None
    w_bp_faced: int | None = None
    l_ace: int | None = None
    l_df: int | None = None
    l_svpt: int | None = None
    l_1st_in: int | None = None
    l_1st_won: int | None = None
    l_2nd_won: int | None = None
    l_sv_gms: int | None = None
    l_bp_saved: int | None = None
    l_bp_faced: int | None = None


@dataclass(frozen=True, slots=True)
class Ranking:
    ranking_date: date
    ranking: int
    player_id: int
    ranking_points: int


@dataclass(frozen=True, slots=True)
class Glicko2Rating:
    mu: float = 1500.0
    phi: float = 350.0
    sigma: float = 0.06
