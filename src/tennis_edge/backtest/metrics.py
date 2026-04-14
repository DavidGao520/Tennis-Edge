"""Backtesting metrics computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def compute_monthly_pnl(result: BacktestResult) -> pd.DataFrame:
    """Aggregate PnL by month."""
    if not result.bets:
        return pd.DataFrame()

    df = pd.DataFrame([
        {"date": b.match_date, "pnl": b.pnl, "wagered": b.bet_amount}
        for b in result.bets
    ])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month").agg(
        pnl=("pnl", "sum"),
        wagered=("wagered", "sum"),
        num_bets=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()
    monthly["roi"] = monthly["pnl"] / monthly["wagered"]

    return monthly


def compute_streak_stats(result: BacktestResult) -> dict:
    """Compute win/loss streak statistics."""
    if not result.bets:
        return {"max_win_streak": 0, "max_loss_streak": 0}

    max_win = 0
    max_loss = 0
    current_win = 0
    current_loss = 0

    for b in result.bets:
        if b.pnl > 0:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)

    return {"max_win_streak": max_win, "max_loss_streak": max_loss}


def summary_dict(result: BacktestResult) -> dict:
    """Create a comprehensive summary dictionary."""
    streaks = compute_streak_stats(result)

    return {
        "Total Bets": result.num_bets,
        "Total PnL": f"${result.total_pnl:.2f}",
        "Total Wagered": f"${result.total_wagered:.2f}",
        "ROI": f"{result.roi * 100:.1f}%",
        "Win Rate": f"{result.win_rate * 100:.1f}%",
        "Avg Edge": f"{result.avg_edge * 100:.1f}%",
        "Brier Score": f"{result.brier_score:.4f}",
        "Max Drawdown": f"{result.max_drawdown * 100:.1f}%",
        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
        "Max Win Streak": streaks["max_win_streak"],
        "Max Loss Streak": streaks["max_loss_streak"],
    }
