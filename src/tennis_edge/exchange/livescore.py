"""Live tennis score fetcher via Flashscore/Sofascore API.

Scrapes live match scores (sets, games, points) for in-play EV calculation.
Falls back to multiple sources for reliability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

ESPN_ATP_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
ESPN_WTA_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard"

SOFASCORE_API = "https://api.sofascore.com/api/v1"
SOFASCORE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}


@dataclass
class LiveScore:
    """Live match score state."""
    player1: str
    player2: str
    sets: list[tuple[int, int]] = field(default_factory=list)  # [(p1_games, p2_games), ...]
    current_game: tuple[int, int] = (0, 0)  # points in current game
    serving: int = 1  # 1 or 2
    status: str = "live"  # "live", "finished", "not_started"
    tournament: str = ""
    round: str = ""
    surface: str = ""
    source: str = ""

    @property
    def sets_p1(self) -> int:
        return sum(1 for s in self.sets if s[0] > s[1])

    @property
    def sets_p2(self) -> int:
        return sum(1 for s in self.sets if s[1] > s[0])

    @property
    def current_set_games(self) -> tuple[int, int]:
        """Games in the current (ongoing) set."""
        if not self.sets:
            return (0, 0)
        last = self.sets[-1]
        # If last set is complete, current set is 0-0
        if (last[0] >= 6 or last[1] >= 6) and abs(last[0] - last[1]) >= 2:
            return (0, 0)
        if last[0] == 7 or last[1] == 7:  # tiebreak completed
            return (0, 0)
        return last

    @property
    def summary(self) -> str:
        sets_str = " ".join(f"{s[0]}-{s[1]}" for s in self.sets)
        return f"{self.player1} vs {self.player2}: {sets_str} ({self.status})"


async def fetch_live_scores() -> list[LiveScore]:
    """Fetch live tennis matches. Tries ESPN (free, no key) first, then Sofascore."""
    scores = await _fetch_espn_scores()
    if scores:
        logger.info("Fetched %d tennis events from ESPN", len(scores))
        return scores

    # Fallback to Sofascore
    scores = await _fetch_sofascore_scores()
    logger.info("Fetched %d tennis events from Sofascore", len(scores))
    return scores


async def _fetch_espn_scores() -> list[LiveScore]:
    """Fetch from ESPN API (free, no API key needed)."""
    scores: list[LiveScore] = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for url in [ESPN_ATP_URL, ESPN_WTA_URL]:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                for event in data.get("events", []):
                    tournament = event.get("name", "")
                    for comp in event.get("competitions", []):
                        score = _parse_espn_competition(comp, tournament)
                        if score:
                            scores.append(score)
    except Exception as e:
        logger.warning("ESPN fetch failed: %s", e)

    return scores


def _parse_espn_competition(comp: dict, tournament: str) -> LiveScore | None:
    """Parse an ESPN competition (match) into LiveScore."""
    status_info = comp.get("status", {})
    status_type = status_info.get("type", {}).get("name", "")

    if status_type == "STATUS_FINAL":
        status = "finished"
    elif status_type == "STATUS_IN_PROGRESS":
        status = "live"
    elif status_type == "STATUS_SCHEDULED":
        status = "not_started"
    else:
        status = status_type

    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return None

    c1, c2 = competitors[0], competitors[1]
    p1 = c1.get("athlete", {}).get("displayName") or c1.get("team", {}).get("name", "?")
    p2 = c2.get("athlete", {}).get("displayName") or c2.get("team", {}).get("name", "?")

    # Set scores from linescores
    ls1 = c1.get("linescores", [])
    ls2 = c2.get("linescores", [])
    sets = []
    for i in range(max(len(ls1), len(ls2))):
        g1 = int(ls1[i].get("value", 0)) if i < len(ls1) else 0
        g2 = int(ls2[i].get("value", 0)) if i < len(ls2) else 0
        sets.append((g1, g2))

    # Serving info (ESPN uses "possession" or "isServing")
    serving = 1
    if c2.get("possession", False) or c2.get("isServing", False):
        serving = 2

    # Surface
    surface = ""
    venue = comp.get("venue", {})
    if venue:
        surface_raw = venue.get("indoor", "")
        # ESPN doesn't always have surface, infer from tournament name
        tournament_lower = tournament.lower()
        if "clay" in tournament_lower or "roland" in tournament_lower or "barcelona" in tournament_lower:
            surface = "Clay"
        elif "grass" in tournament_lower or "wimbledon" in tournament_lower:
            surface = "Grass"
        else:
            surface = "Hard"

    round_info = comp.get("roundName", comp.get("status", {}).get("type", {}).get("shortDetail", ""))

    return LiveScore(
        player1=p1,
        player2=p2,
        sets=sets,
        current_game=(0, 0),  # ESPN doesn't provide point-level data
        serving=serving,
        status=status,
        tournament=tournament,
        round=round_info,
        surface=surface,
        source="espn",
    )


async def _fetch_sofascore_scores() -> list[LiveScore]:
    """Fetch from Sofascore API (may be blocked)."""
    scores: list[LiveScore] = []

    try:
        async with httpx.AsyncClient(timeout=15, headers=SOFASCORE_HEADERS) as client:
            resp = await client.get(f"{SOFASCORE_API}/sport/tennis/events/live")
            if resp.status_code != 200:
                logger.debug("Sofascore returned %d", resp.status_code)
                return scores

            data = resp.json()
            for event in data.get("events", []):
                try:
                    score = _parse_sofascore_event(event)
                    if score:
                        scores.append(score)
                except Exception as e:
                    logger.debug("Sofascore parse error: %s", e)
    except Exception as e:
        logger.debug("Sofascore failed: %s", e)

    return scores


def _parse_sofascore_event(event: dict) -> LiveScore | None:
    """Parse a Sofascore event into a LiveScore."""
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})

    p1_name = home.get("name", "")
    p2_name = away.get("name", "")

    if not p1_name or not p2_name:
        return None

    status_desc = event.get("status", {}).get("description", "")
    status_type = event.get("status", {}).get("type", "")

    if status_type == "finished":
        status = "finished"
    elif status_type == "inprogress":
        status = "live"
    elif status_type == "notstarted":
        status = "not_started"
    else:
        status = status_type

    # Parse scores
    home_score = event.get("homeScore", {})
    away_score = event.get("awayScore", {})

    sets = []
    for period in range(1, 6):
        key = f"period{period}"
        if key in home_score and key in away_score:
            p1_games = int(home_score[key])
            p2_games = int(away_score[key])
            sets.append((p1_games, p2_games))

    # Current game point score
    point_p1 = int(home_score.get("point", "0") or "0")
    point_p2 = int(away_score.get("point", "0") or "0")

    # Who is serving
    serving = 1
    home_serving = event.get("homeTeamServing")
    if home_serving is False:
        serving = 2

    # Tournament info
    tournament = event.get("tournament", {})
    tourney_name = tournament.get("name", "")
    tourney_category = tournament.get("category", {}).get("name", "")

    # Surface from ground type
    ground_type = event.get("groundType", "")
    surface_map = {"clay": "Clay", "hard": "Hard", "grass": "Grass", "hardindoor": "Hard"}
    surface = surface_map.get(ground_type.lower(), ground_type)

    # Round
    round_info = event.get("roundInfo", {})
    round_name = round_info.get("name", "")

    return LiveScore(
        player1=p1_name,
        player2=p2_name,
        sets=sets,
        current_game=(point_p1, point_p2),
        serving=serving,
        status=status,
        tournament=f"{tourney_category} - {tourney_name}" if tourney_category else tourney_name,
        round=round_name,
        surface=surface,
        source="sofascore",
    )


def match_live_to_kalshi(live: LiveScore, market_title: str) -> bool:
    """Check if a live score matches a Kalshi market by player name fuzzy match."""
    title_lower = market_title.lower()
    p1_last = live.player1.split()[-1].lower() if live.player1 else ""
    p2_last = live.player2.split()[-1].lower() if live.player2 else ""

    return p1_last in title_lower and p2_last in title_lower


def normalize_points(p1: int, p2: int) -> tuple[int, int]:
    """Convert tennis point display (0,15,30,40) to internal (0,1,2,3)."""
    point_map = {0: 0, 15: 1, 30: 2, 40: 3, 50: 4}  # 50 = AD in some APIs
    return (point_map.get(p1, p1), point_map.get(p2, p2))
