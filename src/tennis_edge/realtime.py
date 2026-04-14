"""Real-time trading monitor: WebSocket ticker + live scores → in-play EV → alerts.

Flow:
  1. Connect Kalshi WebSocket → real-time bid/ask prices
  2. Every 30s fetch live scores from ESPN → current set/game scores
  3. In-play model: score state + Glicko-2 serve probs → P(player wins)
  4. Edge = live_model_prob - market_prob
  5. Display dashboard with alerts
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .config import AppConfig
from .data.db import Database
from .exchange.auth import KalshiAuth
from .exchange.client import KalshiClient
from .exchange.livescore import LiveScore, fetch_live_scores
from .exchange.ws import (
    KalshiWebSocket,
    TickerUpdate,
    FillUpdate,
    TradeUpdate,
)
from .model.inplay import InPlayModel, MatchScore, serve_prob_from_glicko
from .ratings.glicko2 import Glicko2Engine
from .ratings.tracker import RatingTracker
from .strategy.kelly import edge, expected_value, fractional_kelly

logger = logging.getLogger(__name__)
console = Console()

LIVE_SCORE_INTERVAL = 30  # seconds between ESPN score fetches


@dataclass
class MarketState:
    """Tracks real-time state of a single market."""
    ticker: str
    title: str = ""
    player_name: str = ""
    opponent_name: str = ""

    # Live prices (cents)
    yes_bid: int | None = None
    yes_ask: int | None = None
    last_price: int | None = None
    volume: int = 0
    market_prob: float = 0.5

    # Model probabilities
    pre_match_prob: float | None = None   # Glicko-2 pre-match
    live_prob: float | None = None        # In-play from live score
    live_score_display: str = ""          # "6-4 3-2"

    # Serve probs (cached for in-play model)
    sp1: float = 0.64  # player serve %
    sp2: float = 0.64  # opponent serve %
    player_id: int | None = None
    opponent_id: int | None = None

    # EV (computed from live_prob if available, else pre_match)
    edge_val: float = 0.0
    ev_per_dollar: float = 0.0
    kelly_pct: float = 0.0
    signal: str = "-"
    side: str = "pass"

    # Metadata
    last_update: str = ""
    update_count: int = 0

    @property
    def effective_prob(self) -> float | None:
        """Best available model probability: live > pre-match."""
        return self.live_prob if self.live_prob is not None else self.pre_match_prob


class RealtimeMonitor:
    """Real-time monitor: WebSocket prices + live scores → EV dashboard."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.markets: dict[str, MarketState] = {}
        self.tennis_tickers: set[str] = set()
        self.total_updates = 0
        self.live_scores: list[LiveScore] = []
        self.last_score_fetch = ""
        self.alerts: list[str] = []
        self._db: Database | None = None
        self._tracker: RatingTracker | None = None
        self._name_cache: dict[str, int | None] = {}
        self.kelly_frac = config.strategy.kelly_fraction
        self.balance = 0.0

    async def run(self) -> None:
        """Main entry point."""
        key_path = Path(self.config.project_root) / self.config.kalshi.private_key_path
        auth = KalshiAuth(self.config.kalshi.api_key_id, str(key_path))

        db_path = Path(self.config.project_root) / self.config.database.path
        self._db = Database(db_path)
        self._db.connect()

        engine = Glicko2Engine(tau=self.config.ratings.tau)
        self._tracker = RatingTracker(self._db, engine, period_days=self.config.ratings.rating_period_days)

        # Fetch markets via REST
        async with KalshiClient(self.config.kalshi, auth) as client:
            self.balance = await client.get_balance()
            console.print(f"[bold green]Balance: ${self.balance:.2f}[/]")

            for series in ["KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH"]:
                try:
                    ms = await client.get_markets(series_ticker=series, status="open")
                    for m in ms:
                        self.tennis_tickers.add(m.ticker)
                        state = MarketState(
                            ticker=m.ticker,
                            title=m.title or "",
                            player_name=self._extract_player(m.title or ""),
                        )
                        self._init_model_prob(state)
                        self.markets[m.ticker] = state
                except Exception:
                    pass

            console.print(f"Tracking [bold]{len(self.tennis_tickers)}[/] tennis markets")

        # Connect WebSocket
        ws = KalshiWebSocket(
            auth=auth,
            use_demo=self.config.kalshi.use_demo,
            on_ticker=self._on_ticker,
            on_fill=self._on_fill,
            on_trade=self._on_trade,
        )
        ws.subscribe_ticker()
        ws.subscribe_fills()

        console.print("[bold]Connecting to Kalshi WebSocket...[/]\n")

        # Run WebSocket, live score fetcher, and display in parallel
        await asyncio.gather(
            ws.connect(),
            self._live_score_loop(),
            self._display_loop(),
        )

    # ── Live Score Integration ──

    async def _live_score_loop(self) -> None:
        """Periodically fetch live scores and update in-play probabilities."""
        while True:
            try:
                self.live_scores = await fetch_live_scores()
                live_matches = [s for s in self.live_scores if s.status == "live"]
                self.last_score_fetch = datetime.now().strftime("%H:%M:%S")

                # Match live scores to Kalshi markets
                for state in self.markets.values():
                    self._update_live_prob(state, live_matches)

            except Exception as e:
                logger.debug("Live score fetch error: %s", e)

            await asyncio.sleep(LIVE_SCORE_INTERVAL)

    def _update_live_prob(self, state: MarketState, live_matches: list[LiveScore]) -> None:
        """Match a market to a live score and compute in-play probability."""
        if not state.player_name:
            return

        player_last = state.player_name.split()[-1].lower()
        opponent_last = state.opponent_name.split()[-1].lower() if state.opponent_name else ""

        matched_score = None
        player_is_p1 = True

        for live in live_matches:
            p1_last = live.player1.split()[-1].lower()
            p2_last = live.player2.split()[-1].lower()

            if player_last == p1_last and (not opponent_last or opponent_last == p2_last):
                matched_score = live
                player_is_p1 = True
                break
            elif player_last == p2_last and (not opponent_last or opponent_last == p1_last):
                matched_score = live
                player_is_p1 = False
                break

        if not matched_score:
            state.live_prob = None
            state.live_score_display = ""
            return

        # Parse score into sets
        completed_sets = []
        current_set_games = (0, 0)

        for g1, g2 in matched_score.sets:
            set_complete = (
                (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2
            ) or g1 == 7 or g2 == 7
            if set_complete:
                completed_sets.append((g1, g2))
            else:
                current_set_games = (g1, g2)

        sets_p1 = sum(1 for s in completed_sets if s[0] > s[1])
        sets_p2 = sum(1 for s in completed_sets if s[1] > s[0])

        # In-play model with p1 = live.player1
        # serve probs: need to figure out which player maps to which
        if player_is_p1:
            sp1, sp2 = state.sp1, state.sp2
            score = MatchScore(
                sets1=sets_p1, sets2=sets_p2,
                games1=current_set_games[0], games2=current_set_games[1],
                serving=matched_score.serving, best_of=3,
            )
        else:
            sp1, sp2 = state.sp2, state.sp1  # swap: p1 in model = live.player1
            score = MatchScore(
                sets1=sets_p1, sets2=sets_p2,
                games1=current_set_games[0], games2=current_set_games[1],
                serving=matched_score.serving, best_of=3,
            )

        model = InPlayModel(sp1, sp2)
        p1_win_prob = model.win_probability(score)

        # Convert to YES player's probability
        if player_is_p1:
            state.live_prob = p1_win_prob
        else:
            state.live_prob = 1.0 - p1_win_prob

        # Score display
        score_str = " ".join(f"{s[0]}-{s[1]}" for s in matched_score.sets)
        state.live_score_display = score_str

    # ── WebSocket Handlers ──

    async def _on_ticker(self, update: TickerUpdate) -> None:
        """Handle real-time ticker update."""
        if update.ticker not in self.tennis_tickers:
            return

        self.total_updates += 1
        state = self.markets.get(update.ticker)
        if not state:
            return

        state.yes_bid = update.yes_bid
        state.yes_ask = update.yes_ask
        state.last_price = update.last_price
        if update.volume:
            state.volume = update.volume
        state.update_count += 1
        state.last_update = datetime.now().strftime("%H:%M:%S")

        # Market implied probability
        if state.yes_bid is not None and state.yes_ask is not None:
            state.market_prob = (state.yes_bid + state.yes_ask) / 200.0
        elif state.yes_ask is not None:
            state.market_prob = state.yes_ask / 100.0
        elif state.yes_bid is not None:
            state.market_prob = state.yes_bid / 100.0
        elif state.last_price is not None:
            state.market_prob = state.last_price / 100.0
        else:
            return

        # EV calculation: use live_prob (in-play) if available, else pre_match
        prob = state.effective_prob
        if prob is not None and 0.01 < state.market_prob < 0.99:
            state.edge_val = edge(prob, state.market_prob)
            state.ev_per_dollar = expected_value(prob, state.market_prob)
            kelly = fractional_kelly(prob, state.market_prob, self.kelly_frac)
            state.kelly_pct = abs(kelly)
            state.side = "YES" if kelly > 0 else "NO" if kelly < 0 else "pass"

            abs_edge = abs(state.edge_val)
            if abs_edge >= 0.15:
                state.signal = "STRONG"
            elif abs_edge >= 0.08:
                state.signal = "MODERATE"
            elif abs_edge >= 0.03:
                state.signal = "WEAK"
            else:
                state.signal = "-"

            # Alert on strong signals
            if abs_edge >= 0.08 and state.update_count <= 3:
                src = "LIVE" if state.live_prob is not None else "PRE"
                self.alerts.append(
                    f"[bold yellow]⚡ {state.signal}[/] {state.player_name}: "
                    f"{src}={prob*100:.0f}% mkt={state.market_prob*100:.0f}% "
                    f"edge={state.edge_val*100:+.1f}% → {state.side}"
                )
                if len(self.alerts) > 20:
                    self.alerts.pop(0)

    async def _on_fill(self, fill: FillUpdate) -> None:
        self.alerts.append(
            f"[bold green]✅ FILL[/] {fill.ticker} {fill.action} {fill.count} {fill.side} @ {fill.yes_price}c"
        )

    async def _on_trade(self, trade: TradeUpdate) -> None:
        if trade.ticker in self.tennis_tickers:
            state = self.markets.get(trade.ticker)
            if state:
                state.volume += trade.count

    # ── Display ──

    async def _display_loop(self) -> None:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                live.update(self._build_display())
                await asyncio.sleep(0.5)

    def _build_display(self) -> Table:
        now = datetime.now().strftime("%H:%M:%S")
        active = [
            s for s in self.markets.values()
            if s.yes_bid is not None or s.yes_ask is not None or s.last_price is not None
        ]
        active.sort(key=lambda s: abs(s.edge_val), reverse=True)

        n_live = sum(1 for s in active if s.live_prob is not None)
        score_info = f"Live scores: {n_live}" if n_live else f"Scores: {self.last_score_fetch or 'fetching...'}"

        table = Table(
            title=(
                f"🎾 Tennis-Edge Live Monitor  |  {len(active)} active / {len(self.tennis_tickers)} mkts  |  "
                f"{score_info}  |  Updates: {self.total_updates}  |  {now}"
            ),
            show_lines=False,
        )
        table.add_column("Signal", justify="center", width=8)
        table.add_column("Player", style="white", width=20)
        table.add_column("Score", style="cyan", width=12)
        table.add_column("Bid", justify="right", width=4)
        table.add_column("Ask", justify="right", width=4)
        table.add_column("Market", justify="right", width=7)
        table.add_column("Pre", justify="right", width=5, style="dim")
        table.add_column("Live", justify="right", width=6)
        table.add_column("Edge", justify="right", width=8)
        table.add_column("EV/$", justify="right", width=6)
        table.add_column("Side", justify="center", width=5)
        table.add_column("Kelly$", justify="right", width=7)

        for s in active[:30]:
            sig_style = {"STRONG": "bold red", "MODERATE": "yellow", "WEAK": "white"}.get(s.signal, "dim")
            edge_style = "green" if s.edge_val > 0 else "red" if s.edge_val < 0 else "dim"
            side_style = "green" if s.side == "YES" else "red" if s.side == "NO" else "dim"

            player_short = s.player_name.split()[-1] if s.player_name else s.ticker[-5:]

            # Live prob column: bold if available, dim if using pre-match
            if s.live_prob is not None:
                live_str = f"[bold]{s.live_prob*100:.0f}%[/]"
            else:
                live_str = "[dim]-[/]"

            pre_str = f"{s.pre_match_prob*100:.0f}%" if s.pre_match_prob else "-"

            table.add_row(
                f"[{sig_style}]{s.signal}[/]",
                player_short,
                s.live_score_display or "[dim]-[/]",
                f"{s.yes_bid}" if s.yes_bid else "-",
                f"{s.yes_ask}" if s.yes_ask else "-",
                f"{s.market_prob*100:.0f}%",
                pre_str,
                live_str,
                f"[{edge_style}]{s.edge_val*100:+.1f}%[/]",
                f"{s.ev_per_dollar:+.2f}" if s.effective_prob else "-",
                f"[{side_style}]{s.side}[/]",
                f"${s.kelly_pct * self.balance:.0f}" if s.effective_prob and s.kelly_pct > 0 else "-",
            )

        if self.alerts:
            table.add_section()
            for alert in self.alerts[-5:]:
                table.add_row(alert, *[""] * 11)

        return table

    # ── Init Helpers ──

    def _extract_player(self, title: str) -> str:
        m = re.match(r"Will (.+?) win the", title, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _resolve_player(self, name: str) -> int | None:
        if not name or not self._db:
            return None
        if name in self._name_cache:
            return self._name_cache[name]
        parts = name.strip().split()
        last = parts[-1] if parts else ""
        rows = self._db.query_all(
            "SELECT player_id, first_name FROM players WHERE LOWER(last_name) = ?",
            (last.lower(),)
        )
        if len(rows) == 1:
            self._name_cache[name] = rows[0]["player_id"]
            return rows[0]["player_id"]
        if rows and len(parts) >= 2:
            first = parts[0]
            for r in rows:
                if r["first_name"] and r["first_name"].lower() == first.lower():
                    self._name_cache[name] = r["player_id"]
                    return r["player_id"]
        self._name_cache[name] = None
        return None

    def _init_model_prob(self, state: MarketState) -> None:
        """Compute pre-match probability and cache serve probs for in-play use."""
        if not state.title or not self._db or not self._tracker:
            return

        player = self._extract_player(state.title)
        if not player:
            return

        vs = re.search(r"the\s+(.+?)\s+vs\s+(.+?):", state.title, re.IGNORECASE)
        if not vs:
            return

        name1, name2 = vs.group(1).strip(), vs.group(2).strip()
        if player.lower().split()[-1] == name1.lower().split()[-1]:
            opponent = name2
        else:
            opponent = name1
        state.opponent_name = opponent

        p_id = self._resolve_player(player)
        o_id = self._resolve_player(opponent)
        state.player_id = p_id
        state.opponent_id = o_id

        if not p_id or not o_id:
            return

        r1 = self._tracker.get_rating(p_id, date.today())
        r2 = self._tracker.get_rating(o_id, date.today())

        sp1, sp2 = serve_prob_from_glicko(r1.mu, r2.mu)
        state.sp1 = sp1
        state.sp2 = sp2

        # Pre-match: InPlayModel(player_serve, opponent_serve) → P(player wins)
        model = InPlayModel(sp1, sp2)
        state.pre_match_prob = model.win_probability(MatchScore(best_of=3))
