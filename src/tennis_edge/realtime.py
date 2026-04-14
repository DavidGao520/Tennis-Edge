"""Real-time trading monitor: WebSocket ticker → EV calculation → alerts.

Connects to Kalshi WebSocket, receives price updates in real-time,
runs in-play model, and alerts when edge is detected.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from .config import AppConfig
from .data.db import Database
from .exchange.auth import KalshiAuth
from .exchange.client import KalshiClient
from .exchange.ws import (
    KalshiWebSocket,
    TickerUpdate,
    OrderbookSnapshot,
    TradeUpdate,
    FillUpdate,
)
from .model.inplay import InPlayModel, MatchScore, serve_prob_from_glicko
from .ratings.glicko2 import Glicko2Engine
from .ratings.tracker import RatingTracker
from .strategy.kelly import edge, expected_value, fractional_kelly

logger = logging.getLogger(__name__)
console = Console()


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

    # Model
    model_prob: float | None = None
    pre_match_prob: float | None = None
    market_prob: float = 0.5

    # EV
    edge_val: float = 0.0
    ev_per_dollar: float = 0.0
    kelly_pct: float = 0.0
    signal: str = "NONE"
    side: str = "pass"

    # Metadata
    last_update: str = ""
    update_count: int = 0


class RealtimeMonitor:
    """Real-time monitoring system using Kalshi WebSocket.

    Flow:
      1. Connect WebSocket, subscribe to ticker + fills
      2. Fetch all tennis markets via REST
      3. On each ticker update → recalculate EV → update display
      4. Alert on STRONG signals
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.markets: dict[str, MarketState] = {}
        self.tennis_tickers: set[str] = set()
        self.total_updates = 0
        self.alerts: list[str] = []
        self._db: Database | None = None
        self._tracker: RatingTracker | None = None
        self._name_cache: dict[str, int | None] = {}
        self.kelly_frac = config.strategy.kelly_fraction
        self.min_edge = config.strategy.min_edge
        self.balance = 0.0

    async def run(self) -> None:
        """Main entry point. Connects and streams forever."""
        key_path = Path(self.config.project_root) / self.config.kalshi.private_key_path
        auth = KalshiAuth(self.config.kalshi.api_key_id, str(key_path))

        # Open DB for model predictions
        db_path = Path(self.config.project_root) / self.config.database.path
        self._db = Database(db_path)
        self._db.connect()

        engine = Glicko2Engine(tau=self.config.ratings.tau)
        self._tracker = RatingTracker(self._db, engine, period_days=self.config.ratings.rating_period_days)

        # Fetch initial market list and balance via REST
        async with KalshiClient(self.config.kalshi, auth) as client:
            self.balance = await client.get_balance()
            console.print(f"[bold green]Balance: ${self.balance:.2f}[/]")

            # Fetch all tennis markets
            for series in ["KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH"]:
                try:
                    ms = await client.get_markets(series_ticker=series, status="open")
                    for m in ms:
                        self.tennis_tickers.add(m.ticker)
                        self.markets[m.ticker] = MarketState(
                            ticker=m.ticker,
                            title=m.title or "",
                            player_name=self._extract_player(m.title or ""),
                        )
                        # Pre-compute model probability
                        self._compute_model_prob(m.ticker)
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

        # Run WebSocket and display in parallel
        await asyncio.gather(
            ws.connect(),
            self._display_loop(),
        )

    async def _on_ticker(self, update: TickerUpdate) -> None:
        """Handle real-time ticker update."""
        ticker = update.ticker

        # Only care about tennis markets
        if ticker not in self.tennis_tickers:
            return

        self.total_updates += 1

        state = self.markets.get(ticker)
        if not state:
            return

        # Update prices
        state.yes_bid = update.yes_bid
        state.yes_ask = update.yes_ask
        state.last_price = update.last_price
        if update.volume:
            state.volume = update.volume
        state.update_count += 1
        state.last_update = datetime.now().strftime("%H:%M:%S")

        # Calculate market implied probability
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

        # Calculate EV if we have model probability
        if state.model_prob is not None and 0.01 < state.market_prob < 0.99:
            state.edge_val = edge(state.model_prob, state.market_prob)
            state.ev_per_dollar = expected_value(state.model_prob, state.market_prob)
            kelly = fractional_kelly(state.model_prob, state.market_prob, self.kelly_frac)
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
                alert = (
                    f"[bold yellow]⚡ {state.signal}[/] {state.player_name}: "
                    f"model={state.model_prob*100:.0f}% market={state.market_prob*100:.0f}% "
                    f"edge={state.edge_val*100:+.1f}% → {state.side} "
                    f"(Kelly ${state.kelly_pct * self.balance:.0f})"
                )
                self.alerts.append(alert)
                if len(self.alerts) > 20:
                    self.alerts.pop(0)

    async def _on_fill(self, fill: FillUpdate) -> None:
        """Handle fill notification."""
        alert = f"[bold green]✅ FILL[/] {fill.ticker} {fill.action} {fill.count} {fill.side} @ {fill.yes_price}c"
        self.alerts.append(alert)
        logger.info("Fill: %s %s %d %s @ %d", fill.ticker, fill.action, fill.count, fill.side, fill.yes_price)

    async def _on_trade(self, trade: TradeUpdate) -> None:
        """Handle trade update (someone else traded)."""
        if trade.ticker in self.tennis_tickers:
            state = self.markets.get(trade.ticker)
            if state:
                state.volume += trade.count

    async def _display_loop(self) -> None:
        """Continuously update the terminal display."""
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                live.update(self._build_display())
                await asyncio.sleep(0.5)

    def _build_display(self) -> Table:
        """Build the live dashboard table."""
        # Header
        now = datetime.now().strftime("%H:%M:%S")

        # Main table: markets with liquidity, sorted by edge
        active = [
            s for s in self.markets.values()
            if s.yes_bid is not None or s.yes_ask is not None or s.last_price is not None
        ]
        active.sort(key=lambda s: abs(s.edge_val), reverse=True)

        table = Table(
            title=f"🎾 Tennis-Edge Live Monitor  |  {len(active)} active / {len(self.tennis_tickers)} markets  |  Updates: {self.total_updates}  |  {now}",
            show_lines=False,
        )
        table.add_column("Signal", justify="center", width=8)
        table.add_column("Player", style="white", width=22)
        table.add_column("Bid", justify="right", width=5)
        table.add_column("Ask", justify="right", width=5)
        table.add_column("Market", justify="right", width=7)
        table.add_column("Model", justify="right", width=7)
        table.add_column("Edge", justify="right", width=8)
        table.add_column("EV/$", justify="right", width=7)
        table.add_column("Side", justify="center", width=5)
        table.add_column("Kelly$", justify="right", width=8)
        table.add_column("Vol", justify="right", width=6)
        table.add_column("⏱", justify="right", width=8, style="dim")

        for s in active[:30]:
            sig_style = {"STRONG": "bold red", "MODERATE": "yellow", "WEAK": "white"}.get(s.signal, "dim")
            edge_style = "green" if s.edge_val > 0 else "red" if s.edge_val < 0 else "dim"
            side_style = "green" if s.side == "YES" else "red" if s.side == "NO" else "dim"

            player_short = s.player_name.split()[-1] if s.player_name else s.ticker[-5:]

            table.add_row(
                f"[{sig_style}]{s.signal}[/]",
                player_short,
                f"{s.yes_bid}" if s.yes_bid else "-",
                f"{s.yes_ask}" if s.yes_ask else "-",
                f"{s.market_prob*100:.0f}%",
                f"{s.model_prob*100:.0f}%" if s.model_prob else "-",
                f"[{edge_style}]{s.edge_val*100:+.1f}%[/]",
                f"{s.ev_per_dollar:+.2f}" if s.model_prob else "-",
                f"[{side_style}]{s.side}[/]",
                f"${s.kelly_pct * self.balance:.0f}" if s.model_prob and s.kelly_pct > 0 else "-",
                str(s.volume) if s.volume else "-",
                s.last_update,
            )

        # Add alerts at bottom
        if self.alerts:
            table.add_section()
            for alert in self.alerts[-5:]:
                table.add_row(alert, "", "", "", "", "", "", "", "", "", "", "")

        return table

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

    def _compute_model_prob(self, ticker: str) -> None:
        """Compute pre-match model probability for a market."""
        state = self.markets.get(ticker)
        if not state or not state.title or not self._db or not self._tracker:
            return

        # Extract both player names from title
        title = state.title
        player = self._extract_player(title)
        if not player:
            return

        # Extract opponent from "X vs Y"
        vs = re.search(r"the\s+(.+?)\s+vs\s+(.+?):", title, re.IGNORECASE)
        if not vs:
            return

        name1, name2 = vs.group(1).strip(), vs.group(2).strip()
        if player.lower().split()[-1] == name1.lower().split()[-1]:
            opponent = name2
        else:
            opponent = name1

        state.opponent_name = opponent

        p1_id = self._resolve_player(player)
        p2_id = self._resolve_player(opponent)
        if not p1_id or not p2_id:
            return

        # Get ratings
        r1 = self._tracker.get_rating(p1_id, date.today())
        r2 = self._tracker.get_rating(p2_id, date.today())

        # Serve probabilities: sp1 = player's serve%, sp2 = opponent's serve%
        # InPlayModel(sp1, sp2) → P(player wins), which is what we want for this YES market
        sp1, sp2 = serve_prob_from_glicko(r1.mu, r2.mu)
        model = InPlayModel(sp1, sp2)
        pre_match = model.win_probability(MatchScore(best_of=3))

        # pre_match is already P(player wins) — no flip needed
        state.model_prob = pre_match
        state.pre_match_prob = pre_match
