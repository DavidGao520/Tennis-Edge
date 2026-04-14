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


if __name__ == "__main__":
    main()
