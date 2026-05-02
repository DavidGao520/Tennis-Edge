"""Interactive launch-screen UI for ``tennis-edge``.

Three layers:

1. Launch screen: dashboard + horizontal mode selector (Arbitrage /
   Monitor / Agent / Status / Settings / Exit). Shown when
   ``tennis-edge`` is invoked with no subcommand.

2. First-run onboarding: detects missing API keys / missing model /
   empty DB and walks the user through filling them in. Saves to
   .env (gitignored, never uploaded).

3. Settings sub-menu: rotate keys, validate credentials with real
   API calls, view full config (with secrets masked).

Design notes
------------

- Powered by Rich (already a dependency). No new TUI framework.
- Input uses ``rich.prompt.Prompt`` / ``Confirm`` for forms, plus a
  small raw-key selector for the top-level mode carousel.
- ``_update_dotenv`` preserves existing comments and other lines.
- ``_mask`` redacts secrets when displaying current values.
- The Agent menu does NOT spawn the daemon directly — it prints the
  exact ``tennis-edge agent start`` command for the user to run
  inside tmux. Production deployment pattern stays correct;
  beginners still see the right command.
- pause/resume/flatten flag-file IPC is wired through this menu so
  users do not have to remember the subcommands.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import termios
import time
import tty
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


# LLM provider env-var registry. Single source of truth for which
# providers exist and how to configure each one. Adding a new
# provider = one entry here + a corresponding onboarding helper.
LLM_PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "label": "Gemini",
        "vendor": "Google AI Studio",
        "env_var": "TENNIS_EDGE_GEMINI_KEY",
        "url": "https://aistudio.google.com/apikey",
        "prefix": "AIza",
        "status": "active",  # the agent's grounded provider runs Gemini today
    },
    "openai": {
        "label": "OpenAI",
        "vendor": "OpenAI Platform",
        "env_var": "TENNIS_EDGE_OPENAI_KEY",
        "url": "https://platform.openai.com/api-keys",
        "prefix": "sk-",
        "status": "saved",  # key saved; provider wiring to land in a future PR
    },
    "anthropic": {
        "label": "Claude",
        "vendor": "Anthropic Console",
        "env_var": "TENNIS_EDGE_ANTHROPIC_KEY",
        "url": "https://console.anthropic.com/settings/keys",
        "prefix": "sk-ant-",
        "status": "saved",
    },
}


MAIN_MENU_OPTIONS = [
    ("1", "Arbitrage", "Cross-market pricing gaps"),
    ("2", "Monitor", "Human-led EV review"),
    ("3", "Agent", "Auto-research + paper/live trading"),
    ("4", "Status", "Decisions, P&L, kill switches"),
    ("5", "Settings", "Configure API keys"),
    ("6", "Exit", "Leave Tennis-Edge"),
]


def _llm_present() -> bool:
    """True if ANY supported LLM provider has a key configured."""
    return any(os.environ.get(p["env_var"]) for p in LLM_PROVIDERS.values())


def _active_llm_label() -> str | None:
    """Which provider the agent is wired to use today, if its key is set.

    Returns the label of the first 'status=active' provider whose env
    var is populated, or None if no active provider has a key. Display
    helper for the launch screen.
    """
    for p in LLM_PROVIDERS.values():
        if p["status"] == "active" and os.environ.get(p["env_var"]):
            return p["label"]
    return None


def check_setup(cfg) -> dict[str, bool]:
    """Return setup-completeness flags. Used by launch screen and
    onboarding to decide whether to nag.

    Keys:
      llm     Any supported LLM provider key set in env (Gemini /
              OpenAI / Anthropic). Note: only Gemini is wired into
              the agent's grounded path today; OpenAI / Anthropic
              keys are saved for the upcoming multi-provider PR.
      kalshi  Kalshi api_key_id set AND PEM file present
      model   data/models/latest.joblib exists and is non-empty
      data    players table has rows
    """
    project_root = Path(cfg.project_root)
    return {
        "llm": _llm_present(),
        "kalshi": _kalshi_auth_present(cfg),
        "model": _model_present(project_root, cfg),
        "data": _player_count(project_root / cfg.database.path) > 0,
    }


def _kalshi_auth_present(cfg) -> bool:
    if not cfg.kalshi.api_key_id:
        return False
    pem = Path(cfg.project_root) / cfg.kalshi.private_key_path
    return pem.is_file() and pem.stat().st_size > 0


def _model_present(project_root: Path, cfg) -> bool:
    artifact = project_root / cfg.model.artifacts_dir / "latest.joblib"
    return artifact.is_file() and artifact.stat().st_size > 0


def _player_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            n = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            return int(n)
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return 0


def _match_date_range(db_path: Path) -> tuple[str | None, str | None]:
    if not db_path.is_file():
        return None, None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT MIN(tourney_date), MAX(tourney_date) FROM matches"
            ).fetchone()
            if row is None:
                return None, None
            return row[0], row[1]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return None, None


def _market_tick_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM market_ticks").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# .env helper
# ---------------------------------------------------------------------------


def _update_dotenv(env_path: Path, key: str, value: str) -> None:
    """Atomically replace or append KEY=value in .env.

    Preserves all other lines (comments, blanks, unrelated keys). If
    the file does not exist, creates it.
    """
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.is_file():
        lines = env_path.read_text().splitlines()
    else:
        lines = []

    new_line = f"{key}={value}"
    replaced = False
    for i, line in enumerate(lines):
        # Match "KEY=..." but not commented-out lines.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key == key:
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        lines.append(new_line)

    env_path.write_text("\n".join(lines) + "\n")


def _mask(value: str, show: int = 4) -> str:
    """Redact secrets for display. Keeps first/last few chars only."""
    if not value:
        return "(not set)"
    if len(value) <= show * 2:
        return "•" * len(value)
    return f"{value[:show]}{'•' * 6}{value[-show:]}"


# ---------------------------------------------------------------------------
# Launch screen — top level
# ---------------------------------------------------------------------------


def show_launch_screen(cfg) -> None:
    """Top-level menu loop. Returns when user picks Exit (or Ctrl-C)."""
    _show_boot_animation()
    setup = check_setup(cfg)

    # Auto-trigger onboarding when the critical bits are missing.
    if not setup["llm"] or not setup["model"] or not setup["data"]:
        _print_setup_status(setup)
        try:
            if Confirm.ask("\nRun setup wizard now?", default=True):
                run_onboarding(cfg)
                setup = check_setup(cfg)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            return

    while True:
        try:
            choice = _select_main_menu(cfg, setup, default_index=1)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return

        if choice == "1":
            _route_arbitrage(cfg)
        elif choice == "2":
            _route_monitor(cfg)
        elif choice == "3":
            _agent_submenu(cfg)
        elif choice == "4":
            _show_status(cfg)
        elif choice == "5":
            _settings_submenu(cfg)
        elif choice == "6":
            console.print("[dim]Goodbye.[/dim]")
            return

        # Refresh in case the user changed keys / ran a command.
        setup = check_setup(cfg)


def _show_boot_animation() -> None:
    """Short pixel-art startup animation for interactive terminals."""
    if os.environ.get("TENNIS_EDGE_SKIP_BOOT_ANIMATION") == "1":
        return
    if not console.is_terminal and os.environ.get("TENNIS_EDGE_BOOT_ANIMATION") != "1":
        return

    frames = [_boot_frame(i) for i in range(40)]
    with Live(
        frames[0],
        console=console,
        refresh_per_second=10,
        transient=False,
    ) as live:
        for frame in frames[1:]:
            time.sleep(0.11)
            live.update(frame)


def _boot_frame(step: int) -> str:
    width = max(36, min(console.width, 82))
    travel_frames = 30
    if step >= travel_frames:
        return _boot_title_frame(width, step - travel_frames)

    ball = _tennis_ball_ascii().splitlines()
    ball_width = max(len(row) for row in ball)
    x = round((width - ball_width) * (step / (travel_frames - 1)))
    trail_start = max(0, x - 18)
    trail = " " * trail_start + "-" * max(0, x - trail_start)

    rows = ["", trail]
    rows.extend(f"{' ' * x}{row}" for row in ball)
    return "\n".join(rows)


def _tennis_ball_ascii() -> str:
    return "\n".join(
        [
            r"   ###########",
            r" ##.........####",
            r"#############..##",
            r"#############...##",
            r"#############..##",
            r" ##.........####",
            r"   ###########",
        ]
    )


def _boot_title_frame(width: int, step: int) -> str:
    title = "TENNIS-EDGE"
    subtitle = "Kalshi tennis trading assistant"
    reveal = min(len(title), step + 2)
    title_line = title[:reveal]
    if reveal < len(title):
        title_line += "-" * (len(title) - reveal)
    return "\n".join(
        [
            "",
            _center_text("+" + "-" * 15 + "+", width),
            _center_text(f"|  {title_line}  |", width),
            _center_text("+" + "-" * 15 + "+", width),
            "",
            _center_text(subtitle, width),
        ]
    )


def _center_text(text: str, width: int) -> str:
    return f"{text:^{width}}"


def _render_main_menu(cfg, setup: dict[str, bool], selected_index: int = 1) -> None:
    status_parts = []
    for label, key in [("LLM", "llm"), ("Kalshi", "kalshi"),
                       ("Model", "model"), ("Data", "data")]:
        icon = "[green]✓[/green]" if setup[key] else "[red]✗[/red]"
        status_parts.append(f"{icon} {label}")
    status_line = "  ".join(status_parts)

    n_decisions = _today_decisions(cfg)
    bal_line = _bankroll_line(cfg, setup)
    active = _active_llm_label()

    body = (
        "[bold]Tennis-Edge[/bold] — Kalshi Tennis Trading Assistant\n\n"
        f"  [dim]Setup     [/dim] {status_line}\n"
    )
    if active:
        body += f"  [dim]Agent LLM [/dim] {active} (grounded)\n"
    if bal_line:
        body += f"  [dim]Bankroll  [/dim] {bal_line}\n"
    body += f"  [dim]Today     [/dim] {n_decisions} decisions logged"

    console.print(Panel(body, expand=False, padding=(1, 2)))
    console.print(Panel(
        _main_menu_selector_frame(selected_index, _mode_dashboard_stats(cfg)),
        title="Choose Mode",
        expand=False,
        padding=(1, 2),
    ))
    console.print()


def _select_main_menu(cfg, setup: dict[str, bool], default_index: int = 1) -> str:
    if not console.is_terminal or not sys.stdin.isatty():
        _render_main_menu(cfg, setup, default_index)
        return Prompt.ask(
            "Select",
            choices=[key for key, _, _ in MAIN_MENU_OPTIONS],
            default=MAIN_MENU_OPTIONS[default_index][0],
        )

    index = default_index
    while True:
        console.clear()
        _render_main_menu(cfg, setup, index)
        key = _read_menu_key()
        if key in {"right", "d", "tab"}:
            index = (index + 1) % len(MODE_ROWS)
        elif key in {"left", "a"}:
            index = (index - 1) % len(MODE_ROWS)
        elif key in {"enter", "space"}:
            return str(index + 1)
        elif key in {"1", "2", "3"}:
            return key
        elif key in {"4", "s"}:
            return "4"
        elif key in {"5", "c"}:
            return "5"
        elif key in {"6", "q", "esc"}:
            return "6"
        elif key == "ctrl-c":
            raise KeyboardInterrupt


def _main_menu_selector_frame(
    index: int,
    stats: dict[str, dict[str, int | float]] | None = None,
) -> str:
    cells = []
    for i, (mode, label, _) in enumerate(MODE_ROWS):
        if i == index:
            cells.append(f"[reverse bold cyan] {label} [/reverse bold cyan]")
        else:
            cells.append(f"[dim]{label}[/dim]")

    mode, label, desc = MODE_ROWS[index]
    row = stats[mode] if stats else {
        "settled": 0, "wins": 0, "losses": 0, "voids": 0, "pnl": 0.0,
    }
    pnl = float(row["pnl"])
    pnl_color = "green" if pnl >= 0 else "red"
    return "\n".join(
        [
            "  " + "  |  ".join(cells),
            "",
            f"[bold]{label}[/bold]  [dim]{desc}[/dim]",
            (
                f"Settled: [bold]{row['settled']}[/bold]   "
                f"Earned: [{pnl_color}]${pnl:,.2f}[/{pnl_color}]   "
                f"Success: [bold]{_success_rate(row)}[/bold]"
            ),
            "",
            "[dim]←/→ or A/D move • Enter opens mode • S status • C settings • Q exit[/dim]",
        ]
    )


def _read_menu_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x03":
            return "ctrl-c"
        if ch in {"\r", "\n"}:
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\t":
            return "tab"
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[C":
                return "right"
            if seq == "[D":
                return "left"
            return "esc"
        if ch in {"a", "A"}:
            return "a"
        if ch in {"d", "D"}:
            return "d"
        if ch in {"s", "S"}:
            return "s"
        if ch in {"c", "C"}:
            return "c"
        if ch in {"q", "Q"}:
            return "q"
        if ch in {option[0] for option in MAIN_MENU_OPTIONS}:
            return ch
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _print_setup_status(setup: dict[str, bool]) -> None:
    console.print("[yellow]Setup incomplete:[/yellow]")
    labels = {
        "llm":    "LLM API key (Gemini / OpenAI / Claude)",
        "kalshi": "Kalshi API key (optional for paper mode)",
        "model":  "Trained model artifact",
        "data":   "Player database populated",
    }
    for k, v in setup.items():
        icon = "[green]✓[/green]" if v else "[red]✗[/red]"
        console.print(f"  {icon} {labels[k]}")


def _today_decisions(cfg) -> int:
    path = Path(cfg.project_root) / "data" / "agent_decisions.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _bankroll_line(cfg, setup: dict[str, bool]) -> str:
    """Return Kalshi balance line for the dashboard, or empty string
    if auth is not set up. Keep it cheap — this runs every menu refresh."""
    if not setup["kalshi"]:
        return ""
    # We avoid a Kalshi REST call on every menu refresh. Just show
    # "Kalshi auth configured". Live balance is one click away in
    # Settings → Validate.
    return "[green]Kalshi auth configured[/green] (balance via Settings → Validate)"


# ---------------------------------------------------------------------------
# Mode performance dashboard
# ---------------------------------------------------------------------------


MODE_ROWS = [
    ("arbitrage", "Arbitrage", "pricing gaps"),
    ("monitor", "Monitor", "human review"),
    ("agent", "Agent", "auto research"),
]


def _render_mode_performance_table(cfg) -> None:
    stats = _mode_dashboard_stats(cfg)
    tbl = Table(title="Bet Modes", expand=False)
    tbl.add_column("Mode", style="cyan")
    tbl.add_column("Use", style="dim")
    tbl.add_column("Settled", justify="right")
    tbl.add_column("Earned", justify="right")
    tbl.add_column("Success", justify="right")

    for key, label, use in MODE_ROWS:
        row = stats[key]
        pnl = float(row["pnl"])
        pnl_color = "green" if pnl >= 0 else "red"
        tbl.add_row(
            label,
            use,
            str(row["settled"]),
            f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
            _success_rate(row),
        )
    console.print(tbl)


def _mode_dashboard_stats(cfg) -> dict[str, dict[str, int | float]]:
    stats: dict[str, dict[str, int | float]] = {
        key: {"decisions": 0, "settled": 0, "wins": 0, "losses": 0, "voids": 0, "pnl": 0.0}
        for key, _, _ in MODE_ROWS
    }
    root = Path(cfg.project_root)
    decisions_path = root / "data" / "agent_decisions.jsonl"
    settlements_path = root / "data" / "agent_settlements.jsonl"

    decision_modes: dict[str, str] = {}
    for rec in _read_jsonl(decisions_path):
        mode = _dashboard_mode_key(rec)
        if mode is None:
            mode = "agent"
        decision_id = str(rec.get("decision_id", ""))
        if decision_id:
            decision_modes[decision_id] = mode
        stats[mode]["decisions"] += 1

    for rec in _read_jsonl(settlements_path):
        decision_id = str(rec.get("decision_id", ""))
        mode = _dashboard_mode_key(rec) or decision_modes.get(decision_id, "agent")
        row = stats[mode]
        row["settled"] += 1
        outcome = str(rec.get("outcome", "")).lower()
        if outcome == "won":
            row["wins"] += 1
        elif outcome == "lost":
            row["losses"] += 1
        else:
            row["voids"] += 1
        row["pnl"] += _safe_float(rec.get("realized_pnl", 0.0))

    return stats


def _dashboard_mode_key(rec: dict) -> str | None:
    raw = str(rec.get("strategy_mode") or rec.get("mode") or "").lower()
    if raw in {"arbitrage", "arb"}:
        return "arbitrage"
    if raw in {"monitor", "pre_bet", "human_in_loop"}:
        return "monitor"
    if raw in {"agent", "shadow", "auto"}:
        return "agent"
    return None


def _success_rate(row: dict[str, int | float]) -> str:
    graded = int(row["wins"]) + int(row["losses"])
    if graded == 0:
        return "—"
    return f"{int(row['wins']) / graded:.0%}"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _invoke_click_command(command: click.Command, cfg, **kwargs) -> None:
    ctx = click.Context(command)
    ctx.obj = {"config": cfg}
    with ctx:
        command.callback(**kwargs)


def _mode_requires_kalshi(cfg, mode_label: str) -> bool:
    if check_setup(cfg)["kalshi"]:
        return True
    console.print(Panel(
        f"{mode_label} needs Kalshi auth to fetch live markets.\n\n"
        "Open [bold]Settings[/bold] from the dashboard and set your "
        "Kalshi API key first.",
        title=f"{mode_label} unavailable",
        expand=False,
    ))
    try:
        Prompt.ask("Press Enter to return to dashboard", default="")
    except KeyboardInterrupt:
        pass
    return False


# ---------------------------------------------------------------------------
# [1] Arbitrage
# ---------------------------------------------------------------------------


def _route_arbitrage(cfg) -> None:
    if not _mode_requires_kalshi(cfg, "Arbitrage"):
        return

    console.print(Panel(
        "Entering Arbitrage mode from the dashboard.\n\n"
        "Today this uses the existing live EV opportunity scanner. "
        "Dedicated cross-book arbitrage tracking can plug into this "
        "same dashboard mode next.",
        title="[1] Arbitrage",
        expand=False,
    ))
    try:
        from .cli import opportunities
        _invoke_click_command(opportunities, cfg, min_edge=cfg.strategy.min_edge, category="all")
    except KeyboardInterrupt:
        console.print("\n[yellow]Arbitrage scan stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Arbitrage scan failed: {e}[/red]")

    try:
        Prompt.ask("Press Enter to return to dashboard", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [2] Monitor
# ---------------------------------------------------------------------------


def _route_monitor(cfg) -> None:
    if not _mode_requires_kalshi(cfg, "Monitor"):
        return

    console.print(Panel(
        "Entering Monitor mode from the dashboard.\n\n"
        "Ctrl-C stops the live monitor and returns you here.",
        title="[2] Monitor",
        expand=False,
    ))
    try:
        from .cli import monitor
        _invoke_click_command(monitor, cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")
    try:
        Prompt.ask("Press Enter to return to dashboard", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [3] Agent submenu
# ---------------------------------------------------------------------------


def _agent_submenu(cfg) -> None:
    from .agent.safety import (
        clear_pause_flag, touch_flatten_flag, touch_pause_flag,
    )

    ctrl = Path(cfg.project_root) / "data" / "agent_control"

    while True:
        paused = (ctrl / "pause").exists()
        flatten = (ctrl / "flatten").exists()
        if flatten:
            state = "[red]FLATTEN signaled[/red]"
        elif paused:
            state = "[yellow]PAUSED[/yellow]"
        else:
            state = "[green]ready[/green]"

        body = (
            f"[bold]Agent[/bold]   state: {state}\n\n"
            "  [cyan][1][/cyan] Show start command (paper mode)\n"
            "  [cyan][2][/cyan] Show start command (live mode)\n"
            "  [cyan][3][/cyan] Pause running agent\n"
            "  [cyan][4][/cyan] Resume agent\n"
            "  [cyan][5][/cyan] Flatten + stop agent\n"
            "  [cyan][6][/cyan] View agent status\n"
            "  [cyan][7][/cyan] Back"
        )
        console.print(Panel(body, expand=False))
        try:
            choice = Prompt.ask(
                "Select",
                choices=["1", "2", "3", "4", "5", "6", "7"],
                default="7",
            )
        except KeyboardInterrupt:
            return

        if choice == "1":
            _print_agent_start_command(cfg, mode="paper")
        elif choice == "2":
            _print_agent_start_command(cfg, mode="live")
        elif choice == "3":
            touch_pause_flag(ctrl)
            console.print("[yellow]✓ Pause flag set. Daemon will pause on next tick.[/yellow]")
        elif choice == "4":
            clear_pause_flag(ctrl)
            console.print("[green]✓ Pause flag cleared. Daemon will resume.[/green]")
        elif choice == "5":
            try:
                confirmed = Confirm.ask(
                    "[red]Flatten will stop the agent and (in live mode) close all "
                    "agent-opened positions. Confirm?[/red]",
                    default=False,
                )
            except KeyboardInterrupt:
                continue
            if confirmed:
                touch_flatten_flag(ctrl)
                console.print("[red]✓ Flatten flag set. Daemon will stop on next tick.[/red]")
        elif choice == "6":
            _show_status(cfg)
        elif choice == "7":
            return


def _has_tmux() -> bool:
    """Detect tmux on PATH. Used to choose between recommending
    'tmux new -s agent' (preferred) vs running the agent foreground
    directly (fallback when tmux isn't installed)."""
    import shutil
    return shutil.which("tmux") is not None


def _print_agent_start_command(cfg, *, mode: str) -> None:
    """Show the user a one-line `tennis-edge agent start` command they
    can copy verbatim. Displayed OUTSIDE any Rich panel so terminal
    box-border characters do not contaminate the copy buffer."""
    if mode == "live":
        if not check_setup(cfg)["kalshi"]:
            console.print(Panel(
                "[red]Cannot start live mode: Kalshi auth not configured.[/red]\n\n"
                "Use Settings → 'Set/update Kalshi API key' first.",
                expand=False,
            ))
            try:
                Prompt.ask("Press Enter to return", default="")
            except KeyboardInterrupt:
                pass
            return
        cmd_parts = [
            "tennis-edge agent start",
            "--executor live",
            "--whitelist atp-wta-main",
            "--min-prematch-ev 0.08",
            "--gemini-budget 5.00",
            "--max-position 1.00",
            "--max-total-exposure 20.00",
            "--daily-loss-limit 10.00",
            "--bankroll 100.00",
            "--mode auto",
        ]
        header = (
            "[bold red]LIVE MODE — real money.[/bold red] "
            "Default caps: $1/order, $20 total exposure, $10 daily loss kill."
        )
    else:
        cmd_parts = [
            "tennis-edge agent start",
            "--executor paper",
            "--whitelist atp-wta-main",
            "--min-prematch-ev 0.08",
            "--gemini-budget 2.00",
            "--mode shadow",
        ]
        header = "[green]Paper mode — no real money at risk.[/green]"

    cmd_oneline = " ".join(cmd_parts)

    # Header inside a panel for visual emphasis (caps, mode warning).
    console.print(Panel(header, title=f"Agent → Start ({mode})", expand=False))

    # Plain delimiter + single-line command. NO Rich panel around the
    # command itself — box-border glyphs were getting copied into the
    # paste buffer, which is the user-reported bug.
    console.print()
    console.print(
        "[dim]──────────── copy the line below (single command) ────────────[/dim]"
    )
    console.print(cmd_oneline, style="bold cyan", soft_wrap=True)
    console.print("[dim]───────────────────────── end ─────────────────────────[/dim]")
    console.print()

    # tmux instructions — fallback for systems without tmux installed.
    if _has_tmux():
        console.print("[bold]Recommended[/bold]: run inside tmux so SSH disconnects "
                      "don't kill the daemon.")
        console.print("  [cyan]tmux new -s agent[/cyan]   then paste the command above")
        console.print("  [dim]Ctrl+B D to detach. Re-attach: tmux attach -t agent[/dim]")
    else:
        console.print("[yellow]tmux not detected on PATH.[/yellow] Options:")
        console.print("  • Install: [cyan]brew install tmux[/cyan]   (recommended)")
        console.print("  • Or paste the command directly into your shell — it will run "
                      "in the foreground; closing the terminal stops it.")

    try:
        Prompt.ask("\nPress Enter to return", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [4] Status
# ---------------------------------------------------------------------------


def _show_status(cfg) -> None:
    from .agent.decisions import DecisionLog

    log = DecisionLog(
        Path(cfg.project_root) / "data" / "agent_decisions.jsonl",
        Path(cfg.project_root) / "data" / "agent_settlements.jsonl",
    )
    n = log.count_decisions()

    tbl = Table(title="Agent Status")
    tbl.add_column("Field", style="cyan")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Decisions logged", str(n))

    settlements = list(log.iter_settlements())
    tbl.add_row("Settlements", str(len(settlements)))

    if settlements:
        wins = sum(1 for s in settlements if s.outcome == "won")
        losses = sum(1 for s in settlements if s.outcome == "lost")
        voids = sum(1 for s in settlements if s.outcome == "void")
        pnl = sum(s.realized_pnl for s in settlements)
        tbl.add_row("Wins / Losses / Void", f"{wins}/{losses}/{voids}")
        color = "green" if pnl >= 0 else "red"
        tbl.add_row("P&L (counterfactual)", f"[{color}]${pnl:,.2f}[/{color}]")

    ctrl = Path(cfg.project_root) / "data" / "agent_control"
    tbl.add_row("Pause flag", "present" if (ctrl / "pause").exists() else "—")
    tbl.add_row("Flatten flag", "present" if (ctrl / "flatten").exists() else "—")

    budget_path = Path(cfg.project_root) / "data" / "agent_budget.json"
    if budget_path.is_file():
        try:
            blob = json.loads(budget_path.read_text())
            for provider, s in blob.get("providers", {}).items():
                tbl.add_row(
                    f"Budget {provider}",
                    f"${s.get('total_cost_usd', 0):.4f} (calls={s.get('call_count', 0)})",
                )
        except (OSError, json.JSONDecodeError):
            pass

    console.print(tbl)
    try:
        Prompt.ask("\nPress Enter to return to menu", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [5] Settings
# ---------------------------------------------------------------------------


def _settings_submenu(cfg) -> None:
    while True:
        llm_display = _llm_summary_for_settings()
        kal_display = _mask(cfg.kalshi.api_key_id)

        body = (
            "[bold]Settings[/bold]\n\n"
            f"  [cyan][1][/cyan] Set/update LLM API key      ({llm_display})\n"
            f"  [cyan][2][/cyan] Set/update Kalshi API key   (current: {kal_display})\n"
            "  [cyan][3][/cyan] Validate credentials        (real API calls)\n"
            "  [cyan][4][/cyan] View full config            (read-only)\n"
            "  [cyan][5][/cyan] Back"
        )
        console.print(Panel(body, expand=False))
        try:
            choice = Prompt.ask(
                "Select",
                choices=["1", "2", "3", "4", "5"],
                default="5",
            )
        except KeyboardInterrupt:
            return

        if choice == "1":
            _llm_provider_submenu(cfg)
        elif choice == "2":
            _set_kalshi_key(cfg)
        elif choice == "3":
            _validate_credentials(cfg)
        elif choice == "4":
            _show_config(cfg)
        elif choice == "5":
            return


def _llm_summary_for_settings() -> str:
    """One-liner for the Settings menu showing which providers are
    configured. e.g. 'Gemini ✓, OpenAI ○, Claude ○'."""
    parts = []
    for provider in LLM_PROVIDERS.values():
        configured = bool(os.environ.get(provider["env_var"]))
        icon = "[green]✓[/green]" if configured else "[dim]○[/dim]"
        parts.append(f"{provider['label']} {icon}")
    return ", ".join(parts)


def _llm_provider_submenu(cfg) -> None:
    """Pick a provider, then configure its key. Today only Gemini is
    actually used by the agent's grounded path; OpenAI and Claude
    keys are saved for the upcoming multi-provider PR.
    """
    while True:
        body = ["[bold]LLM provider[/bold]\n"]
        body.append("Select which provider to configure:\n")
        for i, (key, p) in enumerate(LLM_PROVIDERS.items(), start=1):
            current = _mask(os.environ.get(p["env_var"], ""))
            tag = (
                "[green](used by agent)[/green]"
                if p["status"] == "active"
                else "[dim](key saved for future use)[/dim]"
            )
            body.append(
                f"  [cyan][{i}][/cyan] {p['label']:8s} {tag}\n"
                f"      key: {current}"
            )
        back_idx = len(LLM_PROVIDERS) + 1
        body.append(f"\n  [cyan][{back_idx}][/cyan] Back")
        console.print(Panel("\n".join(body), expand=False))

        choices = [str(i) for i in range(1, back_idx + 1)]
        try:
            choice = Prompt.ask("Select", choices=choices, default=str(back_idx))
        except KeyboardInterrupt:
            return

        idx = int(choice)
        if idx == back_idx:
            return

        provider_key = list(LLM_PROVIDERS.keys())[idx - 1]
        _set_llm_key_for(cfg, provider_key)


def _set_llm_key_for(cfg, provider_key: str) -> None:
    """Walk the user through obtaining an API key for the chosen
    provider and save it to .env. Same shape for every provider so
    adding more is one entry in LLM_PROVIDERS away.
    """
    p = LLM_PROVIDERS[provider_key]

    panel_body = (
        f"[bold]{p['label']} API key[/bold]\n\n"
        f"Get a key at [cyan]{p['url']}[/cyan]\n"
        f"Vendor: {p['vendor']}\n"
    )
    if p["status"] == "active":
        panel_body += (
            "\n[green]This provider is wired into the agent today.[/green]"
        )
    else:
        panel_body += (
            "\n[yellow]Key will be saved to .env. Note: the agent's "
            "grounded path currently uses Gemini; multi-provider support "
            "is on the roadmap.[/yellow]"
        )
    console.print(Panel(panel_body, expand=False))

    try:
        key = Prompt.ask("Paste key (input visible)", default="").strip()
    except KeyboardInterrupt:
        return
    if not key:
        console.print("[yellow]Skipped.[/yellow]")
        return

    expected_prefix = p["prefix"]
    if not key.startswith(expected_prefix):
        try:
            ok = Confirm.ask(
                f"[yellow]That doesn't look like a {p['label']} key "
                f"(should start with '{expected_prefix}'). Save anyway?[/yellow]",
                default=False,
            )
        except KeyboardInterrupt:
            return
        if not ok:
            return

    env_path = Path(cfg.project_root) / ".env"
    _update_dotenv(env_path, p["env_var"], key)
    os.environ[p["env_var"]] = key
    console.print(f"[green]✓ Saved to {env_path} as {p['env_var']}[/green]")


def _set_kalshi_key(cfg) -> None:
    console.print(Panel(
        "[bold]Kalshi API key[/bold]\n\n"
        "Get a key pair at [cyan]https://kalshi.com → Account → API[/cyan]\n\n"
        "[bold]You need TWO things:[/bold]\n"
        "  • API Key ID (UUID format, e.g. abc12345-1234-...)\n"
        "  • Private key file (.pem) — Kalshi lets you download it once\n",
        expand=False,
    ))

    try:
        api_key_id = Prompt.ask(
            "Paste API Key ID (UUID)", default="",
        ).strip()
    except KeyboardInterrupt:
        return
    if not api_key_id:
        console.print("[yellow]Skipped Kalshi setup.[/yellow]")
        return

    default_pem = "config/kalshi_private_key.pem"
    pem_full = Path(cfg.project_root) / default_pem
    if not pem_full.is_file():
        console.print(
            f"\n[yellow]PEM file not found at {default_pem}.[/yellow]\n"
            f"Move your downloaded .pem there:\n"
            f"  [dim]mv ~/Downloads/kalshi_private_key.pem {default_pem}[/dim]\n"
            f"  [dim]chmod 600 {default_pem}[/dim]\n",
        )
        try:
            ready = Confirm.ask(
                "PEM file is at the default location now?", default=False,
            )
        except KeyboardInterrupt:
            return
        if not ready:
            console.print(
                "[yellow]Skipped Kalshi setup. The API Key ID was NOT saved. "
                "Re-run after placing the .pem file.[/yellow]"
            )
            return

    env_path = Path(cfg.project_root) / ".env"
    _update_dotenv(env_path, "TENNIS_EDGE__KALSHI__API_KEY_ID", api_key_id)
    _update_dotenv(env_path, "TENNIS_EDGE__KALSHI__PRIVATE_KEY_PATH", default_pem)
    os.environ["TENNIS_EDGE__KALSHI__API_KEY_ID"] = api_key_id
    os.environ["TENNIS_EDGE__KALSHI__PRIVATE_KEY_PATH"] = default_pem
    console.print(f"[green]✓ Saved to {env_path}[/green]")


def _validate_credentials(cfg) -> None:
    console.print("\n[bold]Validating credentials...[/bold]\n")

    # --- Gemini (wired into agent today) ---
    gem_key = os.environ.get("TENNIS_EDGE_GEMINI_KEY")
    if not gem_key:
        console.print("  [yellow]○ Gemini: not set[/yellow]")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=gem_key)
            count = sum(1 for _ in client.models.list())
            console.print(f"  [green]✓ Gemini: {count} models accessible[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Gemini: {e}[/red]")

    # --- OpenAI (key saved; provider not wired yet) ---
    oai_key = os.environ.get("TENNIS_EDGE_OPENAI_KEY")
    if not oai_key:
        console.print("  [dim]○ OpenAI: not set[/dim]")
    else:
        # Cheapest validation: format check only. We do not import
        # the SDK here because openai is not a project dependency
        # (only google-genai is). Multi-provider PR will swap this
        # for a real models.list() call.
        if oai_key.startswith("sk-"):
            console.print(
                "  [green]✓ OpenAI: key saved (format OK; "
                "provider wiring pending)[/green]"
            )
        else:
            console.print(
                "  [yellow]⚠ OpenAI: key saved but doesn't look like "
                "an OpenAI key (should start with 'sk-')[/yellow]"
            )

    # --- Anthropic / Claude (key saved; provider not wired yet) ---
    ant_key = os.environ.get("TENNIS_EDGE_ANTHROPIC_KEY")
    if not ant_key:
        console.print("  [dim]○ Claude: not set[/dim]")
    else:
        if ant_key.startswith("sk-ant-"):
            console.print(
                "  [green]✓ Claude: key saved (format OK; "
                "provider wiring pending)[/green]"
            )
        else:
            console.print(
                "  [yellow]⚠ Claude: key saved but doesn't look like "
                "an Anthropic key (should start with 'sk-ant-')[/yellow]"
            )

    # --- Kalshi ---
    if not _kalshi_auth_present(cfg):
        console.print(
            "  [yellow]○ Kalshi: not set "
            "(paper mode does not require auth)[/yellow]"
        )
    else:
        import asyncio

        from .exchange.auth import KalshiAuth
        from .exchange.client import KalshiClient

        async def _check_kalshi():
            auth = KalshiAuth(
                cfg.kalshi.api_key_id,
                str(Path(cfg.project_root) / cfg.kalshi.private_key_path),
            )
            async with KalshiClient(cfg.kalshi, auth) as c:
                bal = await c.get_balance()
                positions = await c.get_positions()
                return bal, len(positions)

        try:
            bal, n_pos = asyncio.run(_check_kalshi())
            console.print(
                f"  [green]✓ Kalshi: balance ${bal:.2f}, "
                f"{n_pos} open positions[/green]"
            )
        except Exception as e:
            console.print(f"  [red]✗ Kalshi: {e}[/red]")

    try:
        Prompt.ask("\nPress Enter to return", default="")
    except KeyboardInterrupt:
        pass


def _show_config(cfg) -> None:
    tbl = Table(title="Active configuration (secrets masked)")
    tbl.add_column("Section.Field", style="cyan")
    tbl.add_column("Value", style="white")

    for section_name in [
        "database", "data", "ratings", "model", "strategy", "risk", "kalshi",
    ]:
        section = getattr(cfg, section_name)
        for field_name in section.__dataclass_fields__:
            value = getattr(section, field_name)
            if (
                "key" in field_name.lower()
                or "secret" in field_name.lower()
                or "id" in field_name.lower()
            ):
                value = _mask(str(value))
            tbl.add_row(f"{section_name}.{field_name}", str(value))

    console.print(tbl)
    try:
        Prompt.ask("\nPress Enter to return", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Onboarding wizard
# ---------------------------------------------------------------------------


def run_onboarding(cfg) -> None:
    """First-run wizard. Walks the user through Gemini + Kalshi setup.

    Idempotent: re-running it just updates whatever the user fills in.
    """
    console.print(Panel(
        "[bold]Welcome to Tennis-Edge.[/bold]\n\n"
        "We'll set up the API keys you need.\n"
        "Both keys are saved to [cyan].env[/cyan] (gitignored, never uploaded).",
        expand=False,
    ))

    setup = check_setup(cfg)

    if not setup["data"] or not setup["model"]:
        console.print(
            "\n[yellow]Note: Player database and trained model are also missing.[/yellow]\n"
            "After API keys, run these from your shell:\n"
            "  [cyan]tennis-edge ingest[/cyan]    [dim]# downloads tennis match data[/dim]\n"
            "  [cyan]tennis-edge ratings[/cyan]   [dim]# computes Glicko-2[/dim]\n"
            "  [cyan]tennis-edge train[/cyan]     [dim]# trains logistic model[/dim]\n",
        )

    db_path = Path(cfg.project_root) / cfg.database.path
    first_match, last_match = _match_date_range(db_path)
    if last_match:
        console.print(
            f"\n[dim]Local match data currently covers {first_match} → {last_match}.[/dim]"
        )
    if _market_tick_count(db_path) == 0:
        console.print(
            "[dim]No local Kalshi ticks yet. For real backtests, run "
            "`tennis-edge log-ticks` on an always-on machine.[/dim]"
        )

    # LLM key is required for the agent path. Skip prompt only if
    # at least one provider is already configured.
    if not setup["llm"]:
        console.print(
            "\n═════ Step 1/2: LLM API key (required for agent) ═════\n"
        )
        console.print(
            "The agent's grounded research uses an LLM with web access. "
            "Currently the agent runs Gemini; OpenAI and Claude support "
            "is on the roadmap (you can save those keys now too).\n"
        )
        _llm_provider_submenu(cfg)
    else:
        active = _active_llm_label() or "an LLM provider"
        console.print(f"\n[green]✓ {active} key already configured.[/green]")

    # Kalshi optional — paper mode does not require it.
    if not setup["kalshi"]:
        console.print(
            "\n═════ Step 2/2: Kalshi API key (optional — skip for paper-only) ═════\n"
        )
        try:
            do_it = Confirm.ask(
                "Configure Kalshi for live trading? [y/n]", default=False,
            )
        except KeyboardInterrupt:
            do_it = False
        if do_it:
            _set_kalshi_key(cfg)
        else:
            console.print(
                "[yellow]Skipped. You can set it later via Settings → Kalshi.[/yellow]"
            )
    else:
        console.print("\n[green]✓ Kalshi key already configured.[/green]")

    console.print()
    console.print(Panel(
        "[green]Setup complete.[/green] Returning to main menu.",
        expand=False,
    ))
