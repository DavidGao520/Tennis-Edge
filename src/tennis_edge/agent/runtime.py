"""Phase 2 Agent runtime wiring.

Bridges the agent package (generic protocols) to the live Tennis-Edge
stack (SQLite, Glicko-2, FeatureBuilder, LogisticPredictor,
KalshiClient). Kept separate from cli.py so the CLI stays thin and
so runtime wiring is testable in isolation.

Pipeline for a Kalshi market → LLM-ready context:

   ticker ─► Market (cached) ─► title parse ─► (player, opponent)
                                                    │
                                                    ▼
                                    DB.players resolve → (p1_id, p2_id)
                                                    │
                                                    ▼
                            FeatureBuilder + LogisticPredictor ─► model_prob
                                                    │
                                                    ▼
                        DB.matches rolling queries ─► form / H2H / rest
                                                    │
                                                    ▼
                                        PromptContext (text-only)

Design choices:

1. One shared Market cache (ticker → Market, 5 min TTL). The loop
   hits `model_prob_fn` and `context_builder` for the same ticker
   back-to-back; we do not want two REST calls per candidate.

2. Title-based player extraction reuses `scanner._parse_title`
   verbatim. If Kalshi changes market titles, one regex change
   fixes both scanner and agent.

3. `model_prob_fn` returns None quickly if anything is missing
   (unknown player, stale DB, network error). Loop treats None
   as "skip silently" — no LLM call, no failure counted.

4. `context_builder` returns None if player data is insufficient.
   Agent loop handles None as skip without counting as LLM failure.

5. Surface, round, best_of come from the market title when parseable;
   sensible defaults otherwise (Hard, R32, 3). The LLM prompt says
   so explicitly so Gemini can weight them appropriately.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

from ..data.db import Database
from ..exchange.client import KalshiClient
from ..exchange.schemas import Market
from ..features.builder import FeatureBuilder
from ..model.predictor import LogisticPredictor
from ..ratings.tracker import RatingTracker
from .llm import PromptContext

logger = logging.getLogger(__name__)


# Reuse scanner title regex. "Will {player} win the {p1} vs {p2}: {round} match?"
_TITLE_RE = re.compile(
    r"Will (.+?) win the .+? vs .+?: (.+?) match\?", re.IGNORECASE
)
_VS_RE = re.compile(r"the\s+(.+?)\s+vs\s+(.+?):", re.IGNORECASE)

# Default assumptions when ticker metadata is thin.
_DEFAULT_SURFACE = "Hard"
_DEFAULT_ROUND = "R32"
_DEFAULT_BEST_OF = 3


# ---------------------------------------------------------------------------
# Market cache
# ---------------------------------------------------------------------------


@dataclass
class _CachedMarket:
    market: Market
    fetched_at: float


class MarketCache:
    """Async, coalescing ticker → Market cache with TTL.

    Two consumers (model_prob_fn, context_builder) will ask for the
    same ticker back-to-back. Without coalescing, both would fire a
    REST call. The `_inflight` map ensures only one network call per
    ticker happens at a time; the second caller awaits the first.
    """

    def __init__(self, client: KalshiClient, ttl_s: float = 300.0):
        self._client = client
        self._ttl_s = ttl_s
        self._cache: dict[str, _CachedMarket] = {}
        self._inflight: dict[str, asyncio.Task[Market]] = {}

    async def get(self, ticker: str) -> Market | None:
        """Return cached market or fetch. None on exchange error."""
        now = time.monotonic()
        cached = self._cache.get(ticker)
        if cached is not None and now - cached.fetched_at < self._ttl_s:
            return cached.market

        # Coalesce concurrent requests.
        task = self._inflight.get(ticker)
        if task is None:
            task = asyncio.create_task(self._fetch(ticker))
            self._inflight[ticker] = task
        try:
            return await task
        except Exception as e:
            logger.warning("market cache: get(%s) failed: %s", ticker, e)
            return None

    async def _fetch(self, ticker: str) -> Market:
        try:
            market = await self._client.get_market(ticker)
            self._cache[ticker] = _CachedMarket(market=market, fetched_at=time.monotonic())
            return market
        finally:
            self._inflight.pop(ticker, None)


# ---------------------------------------------------------------------------
# Title parser
# ---------------------------------------------------------------------------


def parse_market_title(title: str) -> tuple[str, str, str] | None:
    """Extract (player_yes, player_no, round). Returns None if unparseable.

    YES side of a Kalshi tennis market is always one specific player
    winning. The canonical title shape is
    "Will X win the X vs Y: Round Of 32 match?". player_yes = X,
    player_no = Y, round = "Round Of 32".
    """
    if not title:
        return None
    m = _TITLE_RE.match(title)
    if not m:
        return None
    player_yes_full = m.group(1).strip()
    round_info = m.group(2).strip()

    vs = _VS_RE.search(title)
    if not vs:
        return None
    name1 = vs.group(1).strip()
    name2 = vs.group(2).strip()

    # Figure out which side is YES by last-name or first-name match.
    def _same(a: str, b: str) -> bool:
        a_parts = a.lower().replace("-", " ").split()
        b_parts = b.lower().replace("-", " ").split()
        return bool(a_parts and b_parts and (a_parts[-1] == b_parts[-1] or a_parts[0] == b_parts[0]))

    if _same(player_yes_full, name1):
        return player_yes_full, name2, round_info
    if _same(player_yes_full, name2):
        return player_yes_full, name1, round_info
    # Can't disambiguate — give up.
    return None


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class AgentRuntime:
    """Bundles the pieces the loop needs. Built once at daemon start.

    Holds: DB connection, rating tracker, feature builder, model
    predictor, market cache. Exposes `model_prob_fn` and
    `context_builder` closures that match the agent/loop.py protocols.
    """

    db: Database
    tracker: RatingTracker
    builder: FeatureBuilder
    model: LogisticPredictor
    market_cache: MarketCache
    _name_cache: dict[str, int | None] = field(default_factory=dict)

    # ---- public API: closures for the loop ----

    def model_prob_fn(self, ticker: str) -> float | None:
        """Sync ABI required by loop._maybe_enqueue. Uses cached market
        only; does NOT block on a network call.

        On cache miss: fires an async prewarm in the background so the
        next reader poll (2s later) hits the cache. Returns None this
        tick. In steady-state over a trading session, the cache TTL
        (5 min) covers every active ticker.
        """
        cached = self.market_cache._cache.get(ticker)
        if cached is None:
            self._schedule_prewarm(ticker)
            return None
        return self._predict_for_market(cached.market)

    def _schedule_prewarm(self, ticker: str) -> None:
        """Fire-and-forget market fetch. Idempotent via MarketCache's
        inflight coalescing."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not running in an async context (tests)
        loop.create_task(self.market_cache.get(ticker))

    async def prewarm_market(self, ticker: str) -> None:
        """Loop can call this to populate the cache ahead of enqueue.
        Not strictly needed — the first candidate for a ticker will
        miss model_prob_fn but the context_builder path fetches the
        market, after which subsequent candidates hit the cache."""
        await self.market_cache.get(ticker)

    def context_builder(
        self, ticker: str, model_prob: float, market_yes_cents: int,
    ) -> PromptContext | None:
        """Sync ABI required by loop._handle_candidate. Blocks on the
        cached market. Must be fast; the worker is single-threaded."""
        cached = self.market_cache._cache.get(ticker)
        if cached is None:
            # Market never got cached — skip without logging noise.
            return None
        return self._build_context(cached.market, ticker, model_prob, market_yes_cents)

    # ---- model prediction ----

    def _predict_for_market(self, market: Market) -> float | None:
        parsed = parse_market_title(market.title or "")
        if parsed is None:
            return None
        player_yes, player_no, _round = parsed

        p1 = self._resolve_player(player_yes)
        p2 = self._resolve_player(player_no)
        if p1 is None or p2 is None:
            return None

        today = date.today().isoformat()
        r1 = self.db.query_one(
            "SELECT ranking FROM rankings WHERE player_id = ? ORDER BY ranking_date DESC LIMIT 1",
            (p1,),
        )
        r2 = self.db.query_one(
            "SELECT ranking FROM rankings WHERE player_id = ? ORDER BY ranking_date DESC LIMIT 1",
            (p2,),
        )
        rank1 = r1["ranking"] if r1 else 9999
        rank2 = r2["ranking"] if r2 else 9999

        surface = _infer_surface(market)
        round_name, best_of = _infer_round_best_of(parsed[2])

        try:
            feat = self.builder.build_match_features(
                winner_id=p1, loser_id=p2,
                tourney_date=today, surface=surface,
                tourney_level="M", round_name=round_name,
                best_of=best_of, winner_rank=rank1, loser_rank=rank2,
            )
        except Exception:
            logger.exception("build_match_features failed")
            return None
        if feat is None:
            return None

        features, _label = feat
        df = pd.DataFrame([features])
        for col in self.model.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.model.feature_names].fillna(0)

        try:
            p1_wins = float(self.model.predict_proba(df)[0])
        except Exception:
            logger.exception("model.predict_proba failed")
            return None

        # Canonical order: label=1 means higher-ranked wins. If p1 is
        # higher-ranked, model_prob = P(p1 wins). Otherwise flip.
        return p1_wins if rank1 <= rank2 else 1.0 - p1_wins

    # ---- context building ----

    def _build_context(
        self,
        market: Market,
        ticker: str,
        model_prob: float,
        market_yes_cents: int,
    ) -> PromptContext | None:
        parsed = parse_market_title(market.title or "")
        if parsed is None:
            return None
        player_yes, player_no, round_info = parsed

        p1 = self._resolve_player(player_yes)
        p2 = self._resolve_player(player_no)
        if p1 is None or p2 is None:
            return None

        round_name, best_of = _infer_round_best_of(round_info)
        surface = _infer_surface(market)

        yes_form = self._form_last_n(p1, 10)
        no_form = self._form_last_n(p2, 10)
        h2h = self._h2h_summary(p1, p2, player_yes, player_no)
        yes_rest = self._days_since_last_match(p1)
        no_rest = self._days_since_last_match(p2)

        tournament = _tournament_from_market(market)

        return PromptContext(
            ticker=ticker,
            player_yes=player_yes,
            player_no=player_no,
            tournament=tournament,
            surface=surface,
            round_name=round_name,
            best_of=best_of,
            model_pre_match=model_prob,
            market_yes_cents=market_yes_cents,
            yes_form_last10=yes_form,
            no_form_last10=no_form,
            h2h_summary=h2h,
            yes_days_since_last_match=yes_rest,
            no_days_since_last_match=no_rest,
        )

    # ---- DB helpers ----

    def _resolve_player(self, name: str) -> int | None:
        """Port of EVScanner._resolve_player. Cached for the life of
        the daemon; 2.7k players, cache will converge fast."""
        cached = self._name_cache.get(name)
        if cached is not None or name in self._name_cache:
            return cached

        parts = name.strip().split()
        if not parts:
            self._name_cache[name] = None
            return None

        last = parts[-1]
        first = parts[0] if len(parts) > 1 else ""

        rows = self.db.query_all(
            "SELECT player_id, first_name, last_name FROM players "
            "WHERE LOWER(last_name) = ?",
            (last.lower(),),
        )
        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]

        if rows and first:
            for row in rows:
                fn = (row["first_name"] or "").lower()
                if fn == first.lower() or (fn and first and fn[0] == first[0].lower()):
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

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
                fn = (row["first_name"] or "").lower()
                if fn.startswith(first[:3].lower()):
                    self._name_cache[name] = row["player_id"]
                    return row["player_id"]

        self._name_cache[name] = None
        return None

    def _form_last_n(self, player_id: int, n: int) -> str:
        """Text summary of player's last N matches, e.g. '7-3 last 10'."""
        rows = self.db.query_all(
            """
            SELECT winner_id FROM matches
            WHERE winner_id = ? OR loser_id = ?
            ORDER BY tourney_date DESC LIMIT ?
            """,
            (player_id, player_id, n),
        )
        if not rows:
            return "unknown"
        wins = sum(1 for r in rows if r["winner_id"] == player_id)
        total = len(rows)
        return f"{wins}-{total - wins} last {total}"

    def _h2h_summary(
        self,
        p1_id: int, p2_id: int,
        p1_name: str, p2_name: str,
    ) -> str:
        row = self.db.query_one(
            """
            SELECT
              SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) AS p1_wins,
              SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) AS p2_wins
            FROM matches
            WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
            """,
            (p1_id, p2_id, p1_id, p2_id, p2_id, p1_id),
        )
        if not row:
            return "no prior meetings"
        p1w = row["p1_wins"] or 0
        p2w = row["p2_wins"] or 0
        if p1w == 0 and p2w == 0:
            return "no prior meetings"
        if p1w >= p2w:
            return f"{p1_name.split()[-1]} leads {p1w}-{p2w}"
        return f"{p2_name.split()[-1]} leads {p2w}-{p1w}"

    def _days_since_last_match(self, player_id: int) -> int | None:
        row = self.db.query_one(
            """
            SELECT tourney_date FROM matches
            WHERE winner_id = ? OR loser_id = ?
            ORDER BY tourney_date DESC LIMIT 1
            """,
            (player_id, player_id),
        )
        if not row or not row["tourney_date"]:
            return None
        try:
            last = datetime.strptime(row["tourney_date"], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None
        return (date.today() - last).days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_surface(market: Market) -> str:
    """Best-effort surface from market title. Defaults to Hard."""
    title = (market.title or "").lower()
    if "clay" in title or "roland" in title or "madrid" in title or "rome" in title:
        return "Clay"
    if "grass" in title or "wimbledon" in title or "queen" in title:
        return "Grass"
    return _DEFAULT_SURFACE


def _infer_round_best_of(round_info: str) -> tuple[str, int]:
    """Map a human round string to (canonical_round, best_of).

    Grand Slam main draw is best-of-5 for ATP men; everything else is
    best-of-3. Without tournament-level metadata, default to 3.
    """
    r = (round_info or "").strip().upper()
    if "FINAL" in r and "SEMI" not in r and "QUART" not in r:
        canon = "F"
    elif "SEMI" in r:
        canon = "SF"
    elif "QUART" in r:
        canon = "QF"
    elif "16" in r:
        canon = "R16"
    elif "32" in r:
        canon = "R32"
    elif "64" in r:
        canon = "R64"
    elif "128" in r:
        canon = "R128"
    else:
        canon = _DEFAULT_ROUND
    return canon, _DEFAULT_BEST_OF


def _tournament_from_market(market: Market) -> str:
    """Best-effort tournament name. Strip the "X vs Y" part from title."""
    t = market.title or ""
    # "Will X win the A vs B: Round match?" → "A vs B"
    m = re.search(r"win the (.+?) match\?", t, re.IGNORECASE)
    if m:
        return m.group(1)
    return t or market.event_ticker or "unknown"
