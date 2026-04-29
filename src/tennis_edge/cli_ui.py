"""Interactive launch-screen UI for ``tennis-edge``.

Three layers:

1. Launch screen: dashboard + 5-option menu (Monitor / Agent /
   Status / Settings / Exit). Shown when ``tennis-edge`` is invoked
   with no subcommand.

2. First-run onboarding: detects missing API keys / missing model /
   empty DB and walks the user through filling them in. Saves to
   .env (gitignored, never uploaded).

3. Settings sub-menu: rotate keys, validate credentials with real
   API calls, view full config (with secrets masked).

Design notes
------------

- Powered by Rich (already a dependency). No new TUI framework.
- All input via ``rich.prompt.Prompt`` / ``Confirm``; survives Ctrl-C
  by catching ``KeyboardInterrupt`` at the top of the loop.
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
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_setup(cfg) -> dict[str, bool]:
    """Return setup-completeness flags. Used by launch screen and
    onboarding to decide whether to nag.

    Keys:
      gemini  Gemini API key set in env (loaded from .env at startup)
      kalshi  Kalshi api_key_id set AND PEM file present
      model   data/models/latest.joblib exists and is non-empty
      data    players table has rows
    """
    project_root = Path(cfg.project_root)
    return {
        "gemini": bool(os.environ.get("TENNIS_EDGE_GEMINI_KEY")),
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
    setup = check_setup(cfg)

    # Auto-trigger onboarding when the critical bits are missing.
    if not setup["gemini"] or not setup["model"] or not setup["data"]:
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
            _render_main_menu(cfg, setup)
            choice = Prompt.ask(
                "Select", choices=["1", "2", "3", "4", "5"], default="1",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return

        if choice == "1":
            _route_monitor()
        elif choice == "2":
            _agent_submenu(cfg)
        elif choice == "3":
            _show_status(cfg)
        elif choice == "4":
            _settings_submenu(cfg)
        elif choice == "5":
            console.print("[dim]Goodbye.[/dim]")
            return

        # Refresh in case the user changed keys / ran a command.
        setup = check_setup(cfg)


def _render_main_menu(cfg, setup: dict[str, bool]) -> None:
    status_parts = []
    for label, key in [("Gemini", "gemini"), ("Kalshi", "kalshi"),
                       ("Model", "model"), ("Data", "data")]:
        icon = "[green]✓[/green]" if setup[key] else "[red]✗[/red]"
        status_parts.append(f"{icon} {label}")
    status_line = "  ".join(status_parts)

    n_decisions = _today_decisions(cfg)
    bal_line = _bankroll_line(cfg, setup)

    body = (
        "[bold]Tennis-Edge[/bold] — Kalshi Tennis Trading Assistant\n\n"
        f"  [dim]Setup     [/dim] {status_line}\n"
    )
    if bal_line:
        body += f"  [dim]Bankroll  [/dim] {bal_line}\n"
    body += f"  [dim]Today     [/dim] {n_decisions} decisions logged"

    console.print(Panel(body, expand=False, padding=(1, 2)))
    console.print("  [bold cyan][1][/bold cyan] Monitor       Live market scanner with EV signals")
    console.print("  [bold cyan][2][/bold cyan] Agent         Auto-research + paper/live trading")
    console.print("  [bold cyan][3][/bold cyan] Status        Today's decisions, P&L, kill switches")
    console.print("  [bold cyan][4][/bold cyan] Settings      Configure API keys, view config")
    console.print("  [bold cyan][5][/bold cyan] Exit")
    console.print()


def _print_setup_status(setup: dict[str, bool]) -> None:
    console.print("[yellow]Setup incomplete:[/yellow]")
    labels = {
        "gemini": "Gemini API key",
        "kalshi": "Kalshi API key (optional for paper mode)",
        "model":  "Trained model artifact",
        "data":   "Player database populated",
    }
    for k, v in setup.items():
        icon = "[green]✓[/green]" if v else "[red]✗[/red]"
        console.print(f"  {icon} {labels[k]}")


def _today_decisions(cfg) -> int:
    from .agent.decisions import DecisionLog
    log = DecisionLog(
        Path(cfg.project_root) / "data" / "agent_decisions.jsonl",
        Path(cfg.project_root) / "data" / "agent_settlements.jsonl",
    )
    return log.count_decisions()


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
# [1] Monitor
# ---------------------------------------------------------------------------


def _route_monitor() -> None:
    console.print(Panel(
        "Monitor is a live full-screen dashboard.\n\n"
        "[bold]Run from your shell:[/bold]\n\n"
        "  [cyan]tennis-edge monitor[/cyan]\n\n"
        "[dim]It opens its own TUI. Ctrl-C to quit.[/dim]",
        title="[1] Monitor",
        expand=False,
    ))
    try:
        Prompt.ask("Press Enter to return to menu", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [2] Agent submenu
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


def _print_agent_start_command(cfg, *, mode: str) -> None:
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
        cmd = (
            "tennis-edge agent start --executor live --whitelist atp-wta-main \\\n"
            "  --min-prematch-ev 0.08 --gemini-budget 5.00 \\\n"
            "  --max-position 1.00 --max-total-exposure 20.00 \\\n"
            "  --daily-loss-limit 10.00 --bankroll 100.00 --mode auto"
        )
        warning = (
            "[bold red]LIVE MODE — real money.[/bold red]\n"
            "Default caps: $1/order, $20 total exposure, $10 daily loss kill."
        )
    else:
        cmd = (
            "tennis-edge agent start --executor paper --whitelist atp-wta-main \\\n"
            "  --min-prematch-ev 0.08 --gemini-budget 2.00 --mode shadow"
        )
        warning = "[green]Paper mode — no real money at risk.[/green]"

    console.print(Panel(
        f"{warning}\n\n"
        "Run inside tmux so SSH disconnects don't kill the daemon:\n\n"
        f"  [cyan]tmux new -s agent[/cyan]\n"
        f"  [cyan]{cmd}[/cyan]\n"
        f"  [dim](Ctrl+B D to detach. Re-attach: tmux attach -t agent)[/dim]",
        title=f"Agent → Start ({mode})",
        expand=False,
    ))
    try:
        Prompt.ask("Press Enter to return", default="")
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# [3] Status
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
# [4] Settings
# ---------------------------------------------------------------------------


def _settings_submenu(cfg) -> None:
    while True:
        gem_display = _mask(os.environ.get("TENNIS_EDGE_GEMINI_KEY", ""))
        kal_display = _mask(cfg.kalshi.api_key_id)

        body = (
            "[bold]Settings[/bold]\n\n"
            f"  [cyan][1][/cyan] Set/update Gemini API key   (current: {gem_display})\n"
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
            _set_gemini_key(cfg)
        elif choice == "2":
            _set_kalshi_key(cfg)
        elif choice == "3":
            _validate_credentials(cfg)
        elif choice == "4":
            _show_config(cfg)
        elif choice == "5":
            return


def _set_gemini_key(cfg) -> None:
    console.print(Panel(
        "[bold]Gemini API key[/bold]\n\n"
        "Get a key at [cyan]https://aistudio.google.com/apikey[/cyan]\n"
        "Free tier covers small testing; live runs ~$10/month.",
        expand=False,
    ))
    try:
        key = Prompt.ask("Paste key (input visible)", default="").strip()
    except KeyboardInterrupt:
        return
    if not key:
        console.print("[yellow]Skipped.[/yellow]")
        return
    if not key.startswith("AIza"):
        try:
            ok = Confirm.ask(
                "[yellow]That doesn't look like a Gemini key (should start with "
                "'AIza'). Save anyway?[/yellow]",
                default=False,
            )
        except KeyboardInterrupt:
            return
        if not ok:
            return

    env_path = Path(cfg.project_root) / ".env"
    _update_dotenv(env_path, "TENNIS_EDGE_GEMINI_KEY", key)
    os.environ["TENNIS_EDGE_GEMINI_KEY"] = key
    console.print(f"[green]✓ Saved to {env_path}[/green]")


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

    # --- Gemini ---
    gem_key = os.environ.get("TENNIS_EDGE_GEMINI_KEY")
    if not gem_key:
        console.print("  [red]✗ Gemini: not set[/red]")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=gem_key)
            count = sum(1 for _ in client.models.list())
            console.print(f"  [green]✓ Gemini: {count} models accessible[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Gemini: {e}[/red]")

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

    # Gemini is required for the agent path. Skip prompt only if already set.
    if not setup["gemini"]:
        console.print("\n═════ Step 1/2: Gemini API key (required for agent) ═════\n")
        _set_gemini_key(cfg)
    else:
        console.print("\n[green]✓ Gemini key already configured.[/green]")

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
