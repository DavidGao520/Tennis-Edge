"""Import and analyze Kalshi trading history with correct P&L calculation.

Key insight: when action=sell, side=no, you're selling a YES position.
The revenue is yes_price (not no_price). P&L must be computed from
fill-level cash flows + settlement payouts.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(
    r"KX(ATP|WTA)(CHALLENGER)?MATCH-"
    r"(\d{2})([A-Z]{3})(\d{2})"
    r"([A-Z]{2,5})([A-Z]{2,5})-"
    r"([A-Z]{2,5})"
)

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


@dataclass
class TradeResult:
    ticker: str
    category: str  # ATP Main, ATP Challenger, WTA Main, WTA Challenger, Crypto, etc.
    match_date: str
    pnl: float
    cost: float  # total cash spent (gross buys)
    revenue: float  # cash received (sells + settlement payout)
    fees: float
    result: str  # "yes" or "no"
    won: bool
    yes_held: float  # net YES contracts at settlement
    no_held: float  # net NO contracts at settlement


def _categorize_ticker(ticker: str) -> str:
    if "KXATPCHALLENGER" in ticker:
        return "ATP Challenger"
    elif "KXATPMATCH" in ticker:
        return "ATP Main"
    elif "KXWTACHALLENGER" in ticker:
        return "WTA Challenger"
    elif "KXWTAMATCH" in ticker:
        return "WTA Main"
    elif "BTC" in ticker or "ETH" in ticker:
        return "Crypto"
    elif "NBA" in ticker:
        return "NBA"
    elif "NCAA" in ticker:
        return "NCAA"
    else:
        return "Other"


def _parse_match_date(ticker: str) -> str | None:
    m = TICKER_PATTERN.match(ticker)
    if not m:
        return None
    yy, mon, dd = m.group(3), MONTH_MAP.get(m.group(4), "01"), m.group(5)
    return f"20{yy}-{mon}-{dd}"


def load_trading_history(export_dir: Path) -> list[TradeResult]:
    """Load and compute correct P&L from Kalshi export CSVs."""
    fills_path = export_dir / "kalshi_fills.csv"
    settlements_path = export_dir / "kalshi_settlements.csv"

    if not fills_path.exists() or not settlements_path.exists():
        logger.warning("Missing fills or settlements CSV in %s", export_dir)
        return []

    with open(fills_path) as f:
        fills = list(csv.DictReader(f))
    with open(settlements_path) as f:
        settlements = list(csv.DictReader(f))

    # Compute per-ticker cash flows from fills
    ticker_cash = defaultdict(float)  # net cash (negative = spent)
    ticker_fees = defaultdict(float)
    ticker_yes = defaultdict(float)  # net YES contracts held
    ticker_no = defaultdict(float)  # net NO contracts held

    for fill in fills:
        ticker = fill["ticker"]
        count = float(fill["count_fp"])
        yes_p = float(fill["yes_price_dollars"])
        no_p = float(fill.get("no_price_dollars", 0) or 0)
        fee = float(fill.get("fee_cost", 0) or 0)
        action = fill["action"]
        side = fill["side"]

        ticker_fees[ticker] += fee

        if action == "buy" and side == "yes":
            ticker_cash[ticker] -= count * yes_p
            ticker_yes[ticker] += count
        elif action == "buy" and side == "no":
            ticker_cash[ticker] -= count * no_p
            ticker_no[ticker] += count
        elif action == "sell" and side == "no":
            # Selling YES position: receive yes_price per contract
            ticker_cash[ticker] += count * yes_p
            ticker_yes[ticker] -= count
        elif action == "sell" and side == "yes":
            ticker_cash[ticker] += count * yes_p
            ticker_yes[ticker] -= count

    # Process settlements
    trades: list[TradeResult] = []

    for s in settlements:
        ticker = s["ticker"]
        result = s["market_result"]

        # Settlement payout
        if result == "yes":
            payout = max(0, ticker_yes.get(ticker, 0)) * 1.0
        elif result == "no":
            payout = max(0, ticker_no.get(ticker, 0)) * 1.0
        else:
            payout = 0

        cash = ticker_cash.get(ticker, 0)
        fees = ticker_fees.get(ticker, 0)
        pnl = cash + payout - fees

        trades.append(TradeResult(
            ticker=ticker,
            category=_categorize_ticker(ticker),
            match_date=_parse_match_date(ticker) or s.get("settled_time", "")[:10],
            pnl=pnl,
            cost=abs(min(0, cash)),  # total spent
            revenue=max(0, cash) + payout,  # total received
            fees=fees,
            result=result,
            won=pnl > 0,
            yes_held=ticker_yes.get(ticker, 0),
            no_held=ticker_no.get(ticker, 0),
        ))

    logger.info("Loaded %d trades (P&L: $%.2f)", len(trades), sum(t.pnl for t in trades))
    return trades


def analyze_history(trades: list[TradeResult]) -> dict:
    """Compute summary stats from trading history."""
    if not trades:
        return {}

    total_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    wins = sum(1 for t in trades if t.won)
    total_wagered = sum(t.cost for t in trades)

    # Category breakdown
    cat_pnl: dict[str, float] = defaultdict(float)
    cat_count: dict[str, int] = defaultdict(int)
    cat_wins: dict[str, int] = defaultdict(int)

    for t in trades:
        cat_pnl[t.category] += t.pnl
        cat_count[t.category] += 1
        if t.won:
            cat_wins[t.category] += 1

    tennis_trades = [t for t in trades if "ATP" in t.category or "WTA" in t.category]
    tennis_pnl = sum(t.pnl for t in tennis_trades)

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": wins / len(trades),
        "total_wagered": total_wagered,
        "total_fees": total_fees,
        "net_pnl": total_pnl,
        "roi": total_pnl / total_wagered if total_wagered > 0 else 0,
        "tennis_trades": len(tennis_trades),
        "tennis_pnl": tennis_pnl,
        "categories": {
            cat: {
                "pnl": cat_pnl[cat],
                "trades": cat_count[cat],
                "win_rate": cat_wins[cat] / cat_count[cat] if cat_count[cat] > 0 else 0,
            }
            for cat in sorted(cat_pnl.keys(), key=lambda x: -cat_pnl[x])
        },
        "date_range": f"{min(t.match_date for t in trades)} to {max(t.match_date for t in trades)}",
    }
