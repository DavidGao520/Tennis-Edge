"""CLI entry points for tennis-edge."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import load_config
from .utils.logging import setup_logging

console = Console()


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(),
    default=None,
    help="Path to config YAML file.",
)
@click.pass_context
def main(ctx: click.Context, config: str | None) -> None:
    """Tennis-Edge: ATP Tennis Prediction Market Trading Bot."""
    ctx.ensure_object(dict)

    if config is None:
        # Auto-detect config relative to this package
        pkg_dir = Path(__file__).parent.parent.parent
        default = pkg_dir / "config" / "default.yaml"
        config = str(default) if default.exists() else None

    cfg = load_config(config)
    setup_logging(cfg.logging.level, str(Path(cfg.project_root) / cfg.logging.file))
    ctx.obj["config"] = cfg


@main.command()
@click.option("--force", is_flag=True, help="Re-download all files even if cached.")
@click.pass_context
def ingest(ctx: click.Context, force: bool) -> None:
    """Download and ingest Sackmann ATP data into SQLite."""
    from .data.ingest import ingest_all

    cfg = ctx.obj["config"]
    if force:
        import shutil
        raw_dir = Path(cfg.project_root) / cfg.data.raw_dir
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            console.print("[yellow]Cleared cached data files.[/yellow]")

    console.print("[bold]Starting data ingestion...[/bold]")
    counts = ingest_all(cfg)

    table = Table(title="Ingestion Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for cat, count in counts.items():
        table.add_row(cat, f"{count:,}")
    console.print(table)


@main.command()
@click.pass_context
def ratings(ctx: click.Context) -> None:
    """Compute Glicko-2 ratings for all players."""
    from .data.db import Database
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path

    with Database(db_path) as db:
        engine = Glicko2Engine(
            tau=cfg.ratings.tau,
            initial_mu=cfg.ratings.initial_mu,
            initial_phi=cfg.ratings.initial_phi,
            initial_sigma=cfg.ratings.initial_sigma,
        )
        tracker = RatingTracker(db, engine, period_days=cfg.ratings.rating_period_days)

        console.print("[bold]Computing Glicko-2 ratings...[/bold]")
        n_periods, n_players = tracker.compute_all_ratings()
        console.print(
            f"[green]Done.[/green] Processed {n_periods} rating periods, "
            f"{n_players} unique players rated."
        )


@main.command()
@click.option("--output", "-o", type=click.Path(), default=None)
@click.pass_context
def train(ctx: click.Context, output: str | None) -> None:
    """Train the match prediction model with proper validation."""
    from .data.db import Database
    from .model.training import ModelTrainer
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker
    from .features.builder import FeatureBuilder

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path

    with Database(db_path) as db:
        engine = Glicko2Engine(tau=cfg.ratings.tau)
        tracker = RatingTracker(db, engine, period_days=cfg.ratings.rating_period_days)
        builder = FeatureBuilder(db, tracker)
        trainer = ModelTrainer(cfg, builder)

        console.print("[bold]Training model with validation pipeline...[/bold]\n")
        report = trainer.train_and_evaluate()

        # ── Data Splits ──
        split_table = Table(title="Data Splits")
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Period", style="white")
        split_table.add_column("Samples", justify="right", style="green")
        for s in report.splits:
            split_table.add_row(s.name, f"{s.start} → {s.end}", f"{s.samples:,}")
        console.print(split_table)

        # ── Walk-Forward CV ──
        if report.cv_folds:
            cv_table = Table(title="Walk-Forward Cross-Validation")
            cv_table.add_column("Fold", justify="center")
            cv_table.add_column("Train", justify="right")
            cv_table.add_column("Val", justify="right")
            cv_table.add_column("Best C", justify="right")
            cv_table.add_column("Accuracy", justify="right")
            cv_table.add_column("Brier", justify="right")
            cv_table.add_column("AUC", justify="right")
            cv_table.add_column("ECE", justify="right")

            for f in report.cv_folds:
                cv_table.add_row(
                    str(f.fold), f"{f.train_size:,}", f"{f.val_size:,}",
                    f"{f.best_C:.3f}", f"{f.accuracy:.3f}", f"{f.brier_score:.4f}",
                    f"{f.auc_roc:.3f}", f"{f.ece:.4f}",
                )
            # Mean row
            cv_table.add_row(
                "[bold]Mean[/]", "", "", "",
                f"[bold]{report.cv_mean_accuracy:.3f}[/] ±{report.cv_std_accuracy:.3f}",
                f"[bold]{report.cv_mean_brier:.4f}[/]",
                f"[bold]{report.cv_mean_auc:.3f}[/]", "",
            )
            console.print(cv_table)

        # ── Final Test Metrics ──
        test_table = Table(title=f"Final Test Evaluation (C={report.final_best_C:.4f})")
        test_table.add_column("Metric", style="cyan")
        test_table.add_column("Value", justify="right")

        lift = report.test_accuracy - report.baseline_accuracy
        test_table.add_row("Accuracy", f"[bold]{report.test_accuracy:.4f}[/]")
        test_table.add_row("Baseline Accuracy", f"{report.baseline_accuracy:.4f}")
        test_table.add_row("Lift", f"[{'green' if lift > 0 else 'red'}]{lift:+.4f}[/]")
        test_table.add_row("AUC-ROC", f"{report.test_auc:.4f}")
        test_table.add_row("Brier Score", f"{report.test_brier:.4f}")
        test_table.add_row("Log Loss", f"{report.test_log_loss:.4f}")
        test_table.add_row("ECE (Calibration)", f"{report.test_ece:.4f}")
        console.print(test_table)

        # ── Feature Importance (Permutation) ──
        imp_table = Table(title="Feature Importance (Permutation-based)")
        imp_table.add_column("Rank", justify="center", style="dim")
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right", style="green")
        imp_table.add_column("|Coefficient|", justify="right", style="dim")

        for i, (feat, imp) in enumerate(list(report.feature_importance.items())[:15]):
            coef = report.coefficient_importance.get(feat, 0)
            imp_table.add_row(
                str(i + 1), feat, f"{imp:.4f}",
                f"{coef:.4f}",
            )
        console.print(imp_table)
        console.print(f"\n[dim]Total features: {report.total_features} | Model: {report.model_type} | Saved: {report.model_path}[/dim]")


@main.command()
@click.option("--start", type=str, required=True, help="Start date YYYY-MM-DD")
@click.option("--end", type=str, required=True, help="End date YYYY-MM-DD")
@click.option("--report", type=click.Choice(["console", "html"]), default="console")
@click.pass_context
def backtest(ctx: click.Context, start: str, end: str, report: str) -> None:
    """Run walk-forward backtest."""
    from datetime import date as Date
    from .data.db import Database
    from .backtest.engine import BacktestEngine
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker
    from .features.builder import FeatureBuilder
    from .strategy.sizing import PositionSizer

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path

    start_date = Date.fromisoformat(start)
    end_date = Date.fromisoformat(end)

    with Database(db_path) as db:
        engine = Glicko2Engine(tau=cfg.ratings.tau)
        tracker = RatingTracker(db, engine, period_days=cfg.ratings.rating_period_days)
        builder = FeatureBuilder(db, tracker)
        sizer = PositionSizer(
            bankroll=cfg.strategy.bankroll,
            kelly_fraction=cfg.strategy.kelly_fraction,
            max_bet_fraction=cfg.strategy.max_bet_fraction,
            min_edge=cfg.strategy.min_edge,
        )

        bt = BacktestEngine(cfg, builder, sizer, db)

        console.print(f"[bold]Running backtest {start} to {end}...[/bold]")
        result = bt.run(start_date, end_date)

        if report == "console":
            from .backtest.report import ReportGenerator
            ReportGenerator(result).to_console()
        else:
            from .backtest.report import ReportGenerator
            out_path = Path(cfg.project_root) / cfg.backtest.reports_dir / "report.html"
            ReportGenerator(result).to_html(out_path)
            console.print(f"[green]Report saved to {out_path}[/green]")


@main.command()
@click.pass_context
def scan(ctx: click.Context) -> None:
    """Scan Kalshi for open tennis markets and display opportunities."""
    import asyncio
    from .exchange.client import KalshiClient
    from .exchange.auth import KalshiAuth

    cfg = ctx.obj["config"]

    async def _scan() -> None:
        auth = None
        if cfg.kalshi.api_key_id and cfg.kalshi.private_key_path:
            auth = KalshiAuth(cfg.kalshi.api_key_id, cfg.kalshi.private_key_path)

        async with KalshiClient(cfg.kalshi, auth) as client:
            markets = await client.get_markets(series_ticker="TENNIS", status="open")

            if not markets:
                console.print("[yellow]No open tennis markets found.[/yellow]")
                return

            table = Table(title="Open Tennis Markets")
            table.add_column("Ticker", style="cyan")
            table.add_column("Title", style="white")
            table.add_column("Yes Bid", style="green", justify="right")
            table.add_column("Yes Ask", style="red", justify="right")
            table.add_column("Volume", justify="right")

            for m in markets:
                table.add_row(
                    m.ticker,
                    m.title or "",
                    f"{m.yes_bid:.0f}" if m.yes_bid is not None else "-",
                    f"{m.yes_ask:.0f}" if m.yes_ask is not None else "-",
                    str(m.volume or "-"),
                )
            console.print(table)

    asyncio.run(_scan())


@main.command()
@click.option("--min-edge", type=float, default=0.03, help="Minimum edge to show.")
@click.option("--category", type=click.Choice(["all", "atp", "wta", "main", "challenger"]), default="all")
@click.pass_context
def opportunities(ctx: click.Context, min_edge: float, category: str) -> None:
    """Scan live markets, run model, and show EV opportunities."""
    import asyncio
    from .data.db import Database
    from .exchange.client import KalshiClient
    from .exchange.auth import KalshiAuth
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker
    from .features.builder import FeatureBuilder
    from .model.predictor import LogisticPredictor
    from .strategy.sizing import PositionSizer
    from .scanner import EVScanner

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path
    key_path = Path(cfg.project_root) / cfg.kalshi.private_key_path

    async def _scan() -> None:
        auth = KalshiAuth(cfg.kalshi.api_key_id, str(key_path))

        with Database(db_path) as db:
            engine = Glicko2Engine(tau=cfg.ratings.tau)
            tracker = RatingTracker(db, engine, period_days=cfg.ratings.rating_period_days)
            builder = FeatureBuilder(db, tracker)
            model = LogisticPredictor.load(Path(cfg.project_root) / cfg.model.artifacts_dir / "latest.joblib")
            sizer = PositionSizer(
                bankroll=cfg.strategy.bankroll,
                kelly_fraction=cfg.strategy.kelly_fraction,
                max_bet_fraction=cfg.strategy.max_bet_fraction,
                min_edge=cfg.strategy.min_edge,
            )
            scanner = EVScanner(db, tracker, builder, model, sizer, cfg.strategy.kelly_fraction)

            async with KalshiClient(cfg.kalshi, auth) as client:
                # Fetch all tennis markets
                series = []
                if category in ("all", "atp", "main"):
                    series.append("KXATPMATCH")
                if category in ("all", "atp", "challenger"):
                    series.append("KXATPCHALLENGERMATCH")
                if category in ("all", "wta", "main"):
                    series.append("KXWTAMATCH")
                if category in ("all", "wta", "challenger"):
                    series.append("KXWTACHALLENGERMATCH")

                all_markets = []
                for s in series:
                    markets = await client.get_markets(series_ticker=s, status="open")
                    all_markets.extend(markets)

                console.print(f"[bold]Scanning {len(all_markets)} markets...[/bold]")

                # Analyze each market
                opps = []
                for market in all_markets:
                    try:
                        ob = await client.get_orderbook(market.ticker)
                    except Exception:
                        ob = None

                    opp = scanner.analyze_market_pair(market, None, ob, None)
                    if opp and abs(opp.edge) >= min_edge:
                        opps.append(opp)

                if not opps:
                    console.print("[yellow]No opportunities found above minimum edge.[/yellow]")
                    console.print(f"Markets without liquidity may not have prices yet.")
                    # Show markets count by category
                    from collections import Counter
                    cats = Counter()
                    for m in all_markets:
                        if "KXATPCHALLENGER" in m.ticker: cats["ATP Challenger"] += 1
                        elif "KXATPMATCH" in m.ticker: cats["ATP Main"] += 1
                        elif "KXWTACHALLENGER" in m.ticker: cats["WTA Challenger"] += 1
                        elif "KXWTAMATCH" in m.ticker: cats["WTA Main"] += 1
                    for cat, cnt in cats.most_common():
                        console.print(f"  {cat}: {cnt} markets")
                    return

                # Sort by absolute edge
                opps.sort(key=lambda o: abs(o.edge), reverse=True)

                # Display
                balance = await client.get_balance()
                console.print(f"\n[bold]Balance: ${balance:.2f}[/bold]")

                table = Table(title=f"EV Opportunities (edge >= {min_edge*100:.0f}%)")
                table.add_column("Signal", justify="center")
                table.add_column("Match", style="white")
                table.add_column("Category")
                table.add_column("Side", justify="center")
                table.add_column("Market", justify="right")
                table.add_column("Model", justify="right")
                table.add_column("Edge", justify="right")
                table.add_column("EV/$", justify="right")
                table.add_column("Kelly%", justify="right")
                table.add_column("Bet $", justify="right")

                for opp in opps:
                    signal_color = {"STRONG": "red bold", "MODERATE": "yellow", "WEAK": "white"}.get(opp.signal_strength, "dim")
                    side_color = "green" if opp.recommended_side == "yes" else "red" if opp.recommended_side == "no" else "dim"
                    edge_color = "green" if opp.edge > 0 else "red"

                    bet_size = opp.kelly_fraction * balance
                    player_display = opp.player_name.split()[-1]  # last name
                    opponent_display = opp.opponent_name.split()[-1]

                    table.add_row(
                        f"[{signal_color}]{opp.signal_strength}[/]",
                        f"{player_display} vs {opponent_display}",
                        opp.category,
                        f"[{side_color}]{opp.recommended_side.upper()}[/] {player_display if opp.recommended_side == 'yes' else opponent_display}",
                        f"{opp.market_implied_prob*100:.0f}%",
                        f"{opp.model_prob*100:.0f}%",
                        f"[{edge_color}]{opp.edge*100:+.1f}%[/]",
                        f"{opp.ev_per_dollar:+.2f}",
                        f"{opp.kelly_fraction*100:.1f}%",
                        f"${bet_size:.0f}",
                    )

                console.print(table)
                console.print(f"\n[dim]Model: Logistic Regression | Features: 36 | Ratings: Glicko-2[/dim]")

    asyncio.run(_scan())


@main.command(name="live")
@click.option("--min-edge", type=float, default=0.03, help="Minimum edge to show.")
@click.pass_context
def live_scan(ctx: click.Context, min_edge: float) -> None:
    """Scan LIVE in-play matches: real-time scores + model EV vs Kalshi odds."""
    import asyncio
    from .data.db import Database
    from .exchange.client import KalshiClient
    from .exchange.auth import KalshiAuth
    from .exchange.livescore import fetch_live_scores, match_live_to_kalshi
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker
    from .scanner_live import LiveEVScanner

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path
    key_path = Path(cfg.project_root) / cfg.kalshi.private_key_path

    async def _scan() -> None:
        # Fetch live scores
        console.print("[bold]Fetching live tennis scores...[/bold]")
        live_scores = await fetch_live_scores()
        live_matches = [s for s in live_scores if s.status == "live"]
        console.print(f"  Found {len(live_matches)} live matches\n")

        if not live_matches:
            console.print("[yellow]No live tennis matches right now.[/yellow]")
            # Show upcoming
            upcoming = [s for s in live_scores if s.status == "not_started"]
            if upcoming:
                console.print(f"\n[dim]Upcoming: {len(upcoming)} matches[/dim]")
                for u in upcoming[:5]:
                    console.print(f"  [dim]{u.player1} vs {u.player2} ({u.tournament})[/dim]")
            return

        # Show all live scores first
        score_table = Table(title="Live Tennis Matches")
        score_table.add_column("Match", style="white")
        score_table.add_column("Score", style="cyan")
        score_table.add_column("Serving", justify="center")
        score_table.add_column("Tournament", style="dim")

        for lm in live_matches:
            score_str = " ".join(f"{s[0]}-{s[1]}" for s in lm.sets)
            srv = "●" if lm.serving == 1 else " "
            srv2 = "●" if lm.serving == 2 else " "
            score_table.add_row(
                f"{srv} {lm.player1}\n{srv2} {lm.player2}",
                score_str,
                f"{'P1' if lm.serving == 1 else 'P2'}",
                f"{lm.tournament}\n{lm.round}",
            )
        console.print(score_table)

        # Now match against Kalshi markets
        auth = KalshiAuth(cfg.kalshi.api_key_id, str(key_path))

        with Database(db_path) as db:
            engine = Glicko2Engine(tau=cfg.ratings.tau)
            tracker = RatingTracker(db, engine, period_days=cfg.ratings.rating_period_days)
            scanner = LiveEVScanner(db, tracker, kelly_frac=cfg.strategy.kelly_fraction, min_edge=min_edge)

            async with KalshiClient(cfg.kalshi, auth) as client:
                balance = await client.get_balance()
                console.print(f"\n[bold]Kalshi Balance: ${balance:.2f}[/bold]")

                # Get all tennis markets
                all_markets = []
                for series in ["KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH"]:
                    try:
                        ms = await client.get_markets(series_ticker=series, status="open")
                        all_markets.extend(ms)
                    except Exception:
                        pass

                # Match live scores to Kalshi markets and compute EV
                opps = []
                for lm in live_matches:
                    for market in all_markets:
                        if not market.title:
                            continue
                        if not match_live_to_kalshi(lm, market.title):
                            continue

                        try:
                            ob = await client.get_orderbook(market.ticker)
                        except Exception:
                            ob = None

                        opp = scanner.analyze_live_match(lm, market, ob)
                        if opp and abs(opp.edge) >= min_edge:
                            opps.append(opp)

                if not opps:
                    console.print(f"\n[yellow]No in-play opportunities with edge >= {min_edge*100:.0f}%[/yellow]")
                    console.print("[dim]Markets may not have liquidity, or no edge detected.[/dim]")
                    return

                # Sort by absolute edge
                opps.sort(key=lambda o: abs(o.edge), reverse=True)

                # Display EV table
                ev_table = Table(title="In-Play EV Opportunities")
                ev_table.add_column("Signal", justify="center")
                ev_table.add_column("Match", style="white")
                ev_table.add_column("Score", style="cyan")
                ev_table.add_column("Side", justify="center")
                ev_table.add_column("Market", justify="right")
                ev_table.add_column("Model", justify="right")
                ev_table.add_column("Pre-match", justify="right", style="dim")
                ev_table.add_column("Edge", justify="right")
                ev_table.add_column("EV/$", justify="right")
                ev_table.add_column("Kelly$", justify="right")

                for opp in opps:
                    sig_style = {"STRONG": "red bold", "MODERATE": "yellow", "WEAK": "white"}.get(opp.signal, "dim")
                    side_style = "green" if opp.side == "yes" else "red"
                    edge_style = "green" if opp.edge > 0 else "red"
                    bet_size = opp.kelly_fraction * balance

                    p_last = opp.player_name.split()[-1]
                    o_last = opp.opponent_name.split()[-1]

                    ev_table.add_row(
                        f"[{sig_style}]{opp.signal}[/]",
                        f"{p_last} vs {o_last}",
                        opp.score_display,
                        f"[{side_style}]{opp.side.upper()}[/] {p_last if opp.side == 'yes' else o_last}",
                        f"{opp.market_price*100:.0f}%",
                        f"{opp.model_prob*100:.0f}%",
                        f"{opp.pre_match_prob*100:.0f}%",
                        f"[{edge_style}]{opp.edge*100:+.1f}%[/]",
                        f"{opp.ev_per_dollar:+.2f}",
                        f"${bet_size:.0f}",
                    )

                console.print(ev_table)
                console.print(f"\n[dim]In-play model: serve prob from Glicko-2 → score state → win prob[/dim]")


@main.command()
@click.pass_context
def monitor(ctx: click.Context) -> None:
    """Real-time WebSocket monitor: live prices → EV → alerts."""
    import asyncio
    from .realtime import RealtimeMonitor

    cfg = ctx.obj["config"]
    mon = RealtimeMonitor(cfg)

    console.print("[bold]Starting real-time WebSocket monitor...[/bold]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        asyncio.run(mon.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")


@main.command(name="log-ticks")
@click.pass_context
def log_ticks(ctx: click.Context) -> None:
    """Stream Kalshi WebSocket tennis ticks to SQLite (market_ticks table).

    Run continuously (e.g. inside tmux/screen). Each day this is not running
    is a day of real backtest data we can never recover.
    """
    import asyncio
    from .tick_logger import TickLogger

    cfg = ctx.obj["config"]
    tl = TickLogger(cfg)

    console.print("[bold]Starting Kalshi tick logger (tennis only)...[/bold]")
    console.print("[dim]Writing to market_ticks table. Press Ctrl+C to stop.[/dim]\n")

    try:
        asyncio.run(tl.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tick logger stopped.[/yellow]")


@main.command()
@click.argument("player_name")
@click.pass_context
def player(ctx: click.Context, player_name: str) -> None:
    """Look up a player's rating and stats."""
    from .data.db import Database

    cfg = ctx.obj["config"]
    db_path = Path(cfg.project_root) / cfg.database.path

    with Database(db_path) as db:
        rows = db.query_all(
            "SELECT * FROM players WHERE LOWER(last_name) LIKE ? OR LOWER(first_name) LIKE ? LIMIT 10",
            (f"%{player_name.lower()}%", f"%{player_name.lower()}%"),
        )
        if not rows:
            console.print(f"[red]No players found matching '{player_name}'[/red]")
            return

        for row in rows:
            console.print(f"\n[bold]{row['first_name']} {row['last_name']}[/bold] (ID: {row['player_id']})")
            console.print(f"  Country: {row['country_code']}  Hand: {row['hand']}  Height: {row['height_cm'] or '?'}cm")

            # Latest Glicko-2 rating
            rating = db.query_one(
                "SELECT * FROM glicko2_ratings WHERE player_id = ? ORDER BY as_of_date DESC LIMIT 1",
                (row["player_id"],),
            )
            if rating:
                console.print(
                    f"  Glicko-2: mu={rating['mu']:.0f}  phi={rating['phi']:.0f}  "
                    f"sigma={rating['sigma']:.4f}  (as of {rating['as_of_date']})"
                )

            # Recent match count
            match_count = db.query_one(
                "SELECT COUNT(*) as cnt FROM matches WHERE winner_id = ? OR loser_id = ?",
                (row["player_id"], row["player_id"]),
            )
            if match_count:
                console.print(f"  Total matches in DB: {match_count['cnt']}")


@main.command()
@click.option("--export-dir", type=click.Path(), default=None, help="Path to kalshi_export dir.")
@click.pass_context
def history(ctx: click.Context, export_dir: str | None) -> None:
    """Analyze Kalshi trading history."""
    from .data.history import load_trading_history, analyze_history

    if export_dir is None:
        # Auto-detect
        candidates = [
            Path(ctx.obj["config"].project_root) / ".." / "Kalshi" / "kalshi_export",
            Path("Kalshi/kalshi_export"),
        ]
        for c in candidates:
            if c.exists():
                export_dir = str(c)
                break

    if not export_dir or not Path(export_dir).exists():
        console.print("[red]kalshi_export directory not found. Use --export-dir.[/red]")
        return

    trades = load_trading_history(Path(export_dir))
    stats = analyze_history(trades)

    if not stats:
        console.print("[red]No tennis trades found.[/red]")
        return

    table = Table(title="Trading History Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    pnl = stats["net_pnl"]
    table.add_row("Total Trades", str(stats["total_trades"]))
    table.add_row("Wins / Losses", f"{stats['wins']} / {stats['losses']}")
    table.add_row("Win Rate", f"{stats['win_rate']*100:.1f}%")
    table.add_row("Net P&L", f"[{'green' if pnl>=0 else 'red'}]${pnl:,.2f}[/]")
    table.add_row("Total Wagered", f"${stats['total_wagered']:,.2f}")
    table.add_row("ROI", f"{stats['roi']*100:.1f}%")
    table.add_row("Total Fees", f"[red]${stats['total_fees']:,.2f}[/]")
    table.add_row("Tennis P&L", f"[{'green' if stats['tennis_pnl']>=0 else 'red'}]${stats['tennis_pnl']:,.2f}[/]")
    table.add_row("Date Range", stats["date_range"])
    console.print(table)

    # Category breakdown
    cat_table = Table(title="P&L by Category")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("P&L", justify="right")
    cat_table.add_column("Trades", justify="right")
    cat_table.add_column("Win%", justify="right")

    for cat, data in stats["categories"].items():
        p = data["pnl"]
        style = "green" if p >= 0 else "red"
        cat_table.add_row(
            cat,
            f"[{style}]${p:,.2f}[/]",
            str(data["trades"]),
            f"{data['win_rate']*100:.0f}%",
        )
    console.print(cat_table)


@main.group()
def agent() -> None:
    """Phase 2 shadow-trade agent (Option 3).

    Sub-commands:
      start   — spawn the daemon (runs foreground; use tmux/systemd)
      pause   — pause the running daemon (reversible)
      resume  — clear the pause flag
      flatten — stop the daemon and close agent-opened positions (terminal)
      status  — print run-state, decision count, budget, daily P&L
    """


def _control_dir(cfg) -> Path:
    """Resolved absolute control dir under project_root."""
    return Path(cfg.project_root) / "data" / "agent_control"


def _decisions_paths(cfg) -> tuple[Path, Path]:
    root = Path(cfg.project_root)
    return (
        root / "data" / "agent_decisions.jsonl",
        root / "data" / "agent_settlements.jsonl",
    )


@agent.command("start")
@click.option(
    "--mode",
    type=click.Choice(["shadow"]),
    default="shadow",
    help="Phase 3A only supports shadow. 3B/3C modes come in later lanes.",
)
@click.option("--min-edge", type=float, default=0.08, show_default=True)
@click.option("--cooldown", type=float, default=300.0, show_default=True,
              help="Per-ticker cooldown seconds.")
@click.option("--gemini-budget", type=float, default=50.0, show_default=True,
              help="Monthly Gemini budget cap in USD.")
@click.option("--model",
              default="gemini-3.1-pro-preview", show_default=True,
              help="Gemini model ID for the LLM provider.")
@click.pass_context
def agent_start(
    ctx: click.Context,
    mode: str,
    min_edge: float,
    cooldown: float,
    gemini_budget: float,
    model: str,
) -> None:
    """Start the agent daemon in the foreground.

    Run inside tmux so SSH disconnects don't kill it:

        tmux new -s agent
        tennis-edge agent start
        # Ctrl+B D to detach

    The daemon runs three coroutines concurrently:
      - MarketTickReader + LLM worker (AgentLoop)
      - SettlementPoller (15 min scan)
      - SafetyMonitor.watchdog_loop (30s scan)

    Any of the five kill switches tripping exits the daemon cleanly.
    """
    import asyncio
    import os
    import signal

    from .agent.decisions import DecisionLog
    from .agent.llm import BudgetTracker, GeminiProvider, PricingRates
    from .agent.loop import AgentLoop, AgentLoopConfig
    from .agent.runtime import AgentRuntime, MarketCache
    from .agent.safety import SafetyConfig, SafetyMonitor
    from .agent.settlement import SettlementConfig, SettlementPoller
    from .data.db import Database
    from .exchange.auth import KalshiAuth
    from .exchange.client import KalshiClient
    from .features.builder import FeatureBuilder
    from .model.predictor import LogisticPredictor
    from .ratings.glicko2 import Glicko2Engine
    from .ratings.tracker import RatingTracker

    cfg = ctx.obj["config"]

    # Gemini API key: prefer env, fall back to project .env so the CLI
    # "just works" without the user exporting every time.
    key = os.environ.get("TENNIS_EDGE_GEMINI_KEY")
    if not key:
        env_file = Path(cfg.project_root) / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("TENNIS_EDGE_GEMINI_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["TENNIS_EDGE_GEMINI_KEY"] = key
                    break
    if not key:
        console.print("[red]TENNIS_EDGE_GEMINI_KEY not set (.env or env var).[/red]")
        return

    async def _run() -> None:
        db_path = Path(cfg.project_root) / cfg.database.path
        key_path = Path(cfg.project_root) / cfg.kalshi.private_key_path

        decisions_path, settlements_path = _decisions_paths(cfg)
        decisions = DecisionLog(decisions_path, settlements_path)

        # Conservative Gemini 3.x preview-tier pricing. Thinking tokens
        # billed at output-token rate. Update when real rates confirmed.
        rates = PricingRates(
            input_per_1m_usd=2.50,
            output_per_1m_usd=10.00,
            thinking_per_1m_usd=10.00,
        )
        budget = BudgetTracker(
            Path(cfg.project_root) / "data" / "agent_budget.json",
            monthly_cap_usd={model: gemini_budget},
        )
        llm = GeminiProvider(
            model=model, rates=rates, budget=budget, api_key=key,
        )

        safety = SafetyMonitor(SafetyConfig(control_dir=str(_control_dir(cfg))))

        loop_cfg = AgentLoopConfig(
            min_edge=min_edge,
            cooldown_s=cooldown,
            mode=mode,
        )

        with Database(db_path) as db:
            engine = Glicko2Engine(tau=cfg.ratings.tau)
            tracker = RatingTracker(
                db, engine, period_days=cfg.ratings.rating_period_days,
            )
            builder = FeatureBuilder(db, tracker)
            model_artifact = (
                Path(cfg.project_root) / cfg.model.artifacts_dir / "latest.joblib"
            )
            predictor = LogisticPredictor.load(model_artifact)

            auth = KalshiAuth(cfg.kalshi.api_key_id, str(key_path))
            async with KalshiClient(cfg.kalshi, auth) as client:
                runtime = AgentRuntime(
                    db=db, tracker=tracker, builder=builder, model=predictor,
                    market_cache=MarketCache(client),
                )

                agent_loop = AgentLoop(
                    config=loop_cfg,
                    db_path=db_path,
                    safety=safety,
                    llm=llm,
                    decisions=decisions,
                    model_prob_fn=runtime.model_prob_fn,
                    context_builder=runtime.context_builder,
                )
                settlement = SettlementPoller(
                    log=decisions, exchange=client,
                    config=SettlementConfig(),
                )

                # Live-match predicate: a match is live if we have any
                # tick with received_at within the last 5 minutes. This
                # lets the WS_STALE watchdog fire only when we would
                # actually expect traffic.
                def is_live_match() -> bool:
                    row = db.query_one(
                        "SELECT MAX(received_at) AS m FROM market_ticks "
                        "WHERE received_at > ?",
                        (int(__import__("time").time()) - 300,),
                    )
                    return bool(row and row["m"])

                # Wire SIGINT/SIGTERM to clean shutdown.
                loop_handle = asyncio.get_running_loop()

                def _graceful(_sig=None, _frame=None):
                    agent_loop.request_stop()
                    settlement.request_stop()

                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop_handle.add_signal_handler(sig, _graceful)
                    except NotImplementedError:
                        pass  # windows

                console.print(
                    f"[bold green]Agent daemon starting[/bold green] "
                    f"mode={mode} min_edge={min_edge} cooldown={cooldown}s "
                    f"model={model} budget=${gemini_budget}/mo"
                )
                console.print(f"[dim]Control dir: {_control_dir(cfg)}[/dim]")
                console.print(f"[dim]Decisions: {decisions_path}[/dim]")

                await asyncio.gather(
                    agent_loop.run(),
                    settlement.run(),
                    safety.watchdog_loop(
                        ws=_DummyWS(),   # no WS in pure shadow; tick-DB proxy is enough
                        db_path=db_path,
                        budget=budget,
                        providers=[model],
                        risk=_NullRisk(),   # no executor in 3A, daily P&L check is a no-op
                        live_match_fn=is_live_match,
                        interval_s=30.0,
                    ),
                )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("[yellow]Agent stopped.[/yellow]")

    # If the daemon exited because of a USER_FLATTEN trip, clear the
    # flatten flag so the next start does not immediately re-trip.
    # (Executor is not wired in 3A so there are no positions to close.)
    try:
        from .agent.safety import clear_flatten_flag
        clear_flatten_flag(_control_dir(cfg))
    except Exception:
        pass


class _DummyWS:
    """Phase 3A shadow mode: agent does not own a live WS. The tick
    logger on the Mac mini is the sole writer; the agent only reads
    market_ticks. Fake the timestamps so the WS watchdog never trips.

    Lane C invariant: seconds_since_last_connect > 60 = link down.
    By returning a tiny value here we opt out of that check entirely;
    the tick-logger-stale switch is the real health signal for 3A.
    """

    def seconds_since_last_message(self):
        return 0.0

    def seconds_since_last_connect(self):
        return 0.0


class _NullRisk:
    """No-op risk manager for Phase 3A (executor not wired yet).
    daily_pnl=0 so the DAILY_LOSS_LIMIT switch never trips in shadow."""

    class _State:
        daily_pnl = 0.0

    state = _State()


@agent.command("pause")
@click.pass_context
def agent_pause(ctx: click.Context) -> None:
    """Pause the running agent (reversible). Daemon stops processing
    candidates but stays up so `resume` can flip it back."""
    from .agent.safety import touch_pause_flag
    p = touch_pause_flag(_control_dir(ctx.obj["config"]))
    console.print(f"[yellow]Paused[/yellow] — flag at {p}")


@agent.command("resume")
@click.pass_context
def agent_resume(ctx: click.Context) -> None:
    """Resume a paused agent by clearing the pause flag."""
    from .agent.safety import clear_pause_flag
    clear_pause_flag(_control_dir(ctx.obj["config"]))
    console.print("[green]Resumed[/green]")


@agent.command("flatten")
@click.pass_context
def agent_flatten(ctx: click.Context) -> None:
    """Stop the daemon (terminal). Trips USER_FLATTEN; the daemon
    exits after closing agent-opened positions (3B/3C) — in 3A shadow
    mode there are no positions, so the daemon simply exits."""
    from .agent.safety import touch_flatten_flag
    p = touch_flatten_flag(_control_dir(ctx.obj["config"]))
    console.print(f"[red]Flatten signaled[/red] — flag at {p}")


@agent.command("status")
@click.pass_context
def agent_status(ctx: click.Context) -> None:
    """Print decision counts, budget, daily P&L. Safe to run while the
    daemon is running — this command never writes."""
    import json

    from .agent.decisions import DecisionLog

    cfg = ctx.obj["config"]
    dec_path, set_path = _decisions_paths(cfg)
    ctrl = _control_dir(cfg)

    log = DecisionLog(dec_path, set_path)
    n_decisions = log.count_decisions()
    n_settlements = sum(1 for _ in log.iter_settlements())

    table = Table(title="Agent Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Decisions logged", str(n_decisions))
    table.add_row("Settlements", str(n_settlements))
    table.add_row("Unresolved", str(n_decisions - n_settlements))
    table.add_row("Pause flag", "present" if (ctrl / "pause").exists() else "—")
    table.add_row("Flatten flag", "present" if (ctrl / "flatten").exists() else "—")

    budget_path = Path(cfg.project_root) / "data" / "agent_budget.json"
    if budget_path.exists():
        try:
            blob = json.loads(budget_path.read_text())
            for provider, s in blob.get("providers", {}).items():
                table.add_row(
                    f"Budget {provider}",
                    f"${s.get('total_cost_usd', 0):.4f} "
                    f"(calls={s.get('call_count', 0)})",
                )
        except Exception:
            pass

    # Counterfactual shadow P&L summary (only when settlements exist).
    if n_settlements > 0:
        wins = losses = voids = 0
        pnl = 0.0
        for s in log.iter_settlements():
            if s.outcome == "won":
                wins += 1
            elif s.outcome == "lost":
                losses += 1
            else:
                voids += 1
            pnl += s.realized_pnl
        table.add_row("Shadow wins/losses/void", f"{wins}/{losses}/{voids}")
        pnl_color = "green" if pnl >= 0 else "red"
        table.add_row("Counterfactual P&L", f"[{pnl_color}]${pnl:,.2f}[/]")

    console.print(table)


if __name__ == "__main__":
    main()
