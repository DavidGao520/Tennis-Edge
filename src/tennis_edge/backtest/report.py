"""Report generation for backtest results."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .engine import BacktestResult
from .metrics import compute_monthly_pnl, summary_dict

console = Console()

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Tennis-Edge Backtest Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
th {{ background: #161b22; color: #58a6ff; }}
tr:nth-child(even) {{ background: #161b22; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
.metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.metric-value {{ font-size: 24px; font-weight: bold; }}
.metric-label {{ color: #8b949e; font-size: 14px; }}
</style>
</head>
<body>
<h1>Tennis-Edge Backtest Report</h1>

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-value {pnl_class}">{total_pnl}</div>
  <div class="metric-label">Total PnL</div>
</div>
<div class="metric-card">
  <div class="metric-value">{roi}</div>
  <div class="metric-label">ROI</div>
</div>
<div class="metric-card">
  <div class="metric-value">{sharpe}</div>
  <div class="metric-label">Sharpe Ratio</div>
</div>
<div class="metric-card">
  <div class="metric-value">{num_bets}</div>
  <div class="metric-label">Total Bets</div>
</div>
<div class="metric-card">
  <div class="metric-value">{win_rate}</div>
  <div class="metric-label">Win Rate</div>
</div>
<div class="metric-card">
  <div class="metric-value">{max_dd}</div>
  <div class="metric-label">Max Drawdown</div>
</div>
</div>

<h2>Monthly Performance</h2>
{monthly_table}

<h2>Recent Bets (last 50)</h2>
{bets_table}

</body>
</html>
"""


class ReportGenerator:
    def __init__(self, result: BacktestResult):
        self.result = result

    def to_console(self) -> None:
        """Print summary to terminal using rich tables."""
        summary = summary_dict(self.result)

        table = Table(title="Backtest Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        for k, v in summary.items():
            style = None
            if isinstance(v, str) and v.startswith("-"):
                style = "red"
            table.add_row(k, str(v), style=style)
        console.print(table)

        # Monthly PnL
        monthly = compute_monthly_pnl(self.result)
        if not monthly.empty:
            mt = Table(title="Monthly PnL")
            mt.add_column("Month")
            mt.add_column("PnL", justify="right")
            mt.add_column("ROI", justify="right")
            mt.add_column("Bets", justify="right")
            mt.add_column("Win%", justify="right")

            for _, row in monthly.iterrows():
                pnl_str = f"${row['pnl']:.2f}"
                style = "green" if row["pnl"] >= 0 else "red"
                mt.add_row(
                    str(row["month"]),
                    pnl_str,
                    f"{row['roi'] * 100:.1f}%",
                    str(int(row["num_bets"])),
                    f"{row['win_rate'] * 100:.0f}%",
                    style=style,
                )
            console.print(mt)

    def to_html(self, output_path: Path) -> None:
        """Generate HTML report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        monthly = compute_monthly_pnl(self.result)
        monthly_rows = ""
        if not monthly.empty:
            monthly_rows = "<table><tr><th>Month</th><th>PnL</th><th>ROI</th><th>Bets</th><th>Win%</th></tr>"
            for _, row in monthly.iterrows():
                cls = "positive" if row["pnl"] >= 0 else "negative"
                monthly_rows += (
                    f'<tr><td>{row["month"]}</td>'
                    f'<td class="{cls}">${row["pnl"]:.2f}</td>'
                    f'<td>{row["roi"] * 100:.1f}%</td>'
                    f'<td>{int(row["num_bets"])}</td>'
                    f'<td>{row["win_rate"] * 100:.0f}%</td></tr>'
                )
            monthly_rows += "</table>"

        bets_rows = "<table><tr><th>Date</th><th>P1</th><th>P2</th><th>Side</th><th>Edge</th><th>Bet</th><th>PnL</th></tr>"
        for b in self.result.bets[-50:]:
            cls = "positive" if b.pnl >= 0 else "negative"
            bets_rows += (
                f'<tr><td>{b.match_date}</td><td>{b.p1_name}</td><td>{b.p2_name}</td>'
                f'<td>{b.side}</td><td>{b.edge * 100:.1f}%</td>'
                f'<td>${b.bet_amount:.2f}</td>'
                f'<td class="{cls}">${b.pnl:.2f}</td></tr>'
            )
        bets_rows += "</table>"

        html = HTML_TEMPLATE.format(
            total_pnl=f"${self.result.total_pnl:.2f}",
            pnl_class="positive" if self.result.total_pnl >= 0 else "negative",
            roi=f"{self.result.roi * 100:.1f}%",
            sharpe=f"{self.result.sharpe_ratio:.2f}",
            num_bets=self.result.num_bets,
            win_rate=f"{self.result.win_rate * 100:.1f}%",
            max_dd=f"{self.result.max_drawdown * 100:.1f}%",
            monthly_table=monthly_rows,
            bets_table=bets_rows,
        )

        output_path.write_text(html, encoding="utf-8")
