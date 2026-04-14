"""Walk-forward backtesting engine with periodic model retraining."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.db import Database
from ..features.builder import FeatureBuilder
from ..model.predictor import create_predictor, MatchPredictor
from ..strategy.sizing import BetDecision, PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class BetRecord:
    match_date: str
    p1_name: str
    p2_name: str
    side: str
    model_prob: float
    market_prob: float
    edge: float
    bet_amount: float
    outcome: float  # 1.0 = win, 0.0 = loss
    pnl: float


@dataclass
class BacktestResult:
    bets: list[BetRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    roi: float = 0.0
    num_bets: int = 0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    brier_score: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Walk-forward backtest: iterate matches, predict, size, simulate bets."""

    def __init__(
        self,
        config: AppConfig,
        feature_builder: FeatureBuilder,
        sizer: PositionSizer,
        db: Database,
    ):
        self.config = config
        self.builder = feature_builder
        self.sizer = sizer
        self.db = db
        self.retrain_interval = timedelta(days=config.backtest.retrain_interval_days)

    def run(self, start_date: date, end_date: date) -> BacktestResult:
        """Execute walk-forward backtest."""
        result = BacktestResult()
        bankroll = self.config.backtest.initial_bankroll
        self.sizer.bankroll = bankroll

        # Train initial model on data before start_date
        train_start = date(self.config.model.train_start_year, 1, 1)
        model = self._train_model(train_start, start_date)
        last_train_date = start_date

        # Get all matches in the test period
        matches = self.db.query_all(
            "SELECT m.*, "
            "  pw.first_name || ' ' || pw.last_name as winner_name, "
            "  pl.first_name || ' ' || pl.last_name as loser_name "
            "FROM matches m "
            "LEFT JOIN players pw ON m.winner_id = pw.player_id "
            "LEFT JOIN players pl ON m.loser_id = pl.player_id "
            "WHERE m.tourney_date >= ? AND m.tourney_date <= ? "
            "ORDER BY m.tourney_date, m.id",
            (start_date.isoformat(), end_date.isoformat()),
        )

        logger.info("Backtesting %d matches from %s to %s", len(matches), start_date, end_date)

        predictions = []
        outcomes = []
        equity = [bankroll]

        for m in matches:
            match_date = date.fromisoformat(m["tourney_date"])

            # Periodic retraining
            if match_date - last_train_date >= self.retrain_interval:
                logger.info("Retraining model at %s", match_date)
                model = self._train_model(train_start, match_date)
                last_train_date = match_date

            if model is None:
                continue

            # Build features
            feat_result = self.builder.build_match_features(
                winner_id=m["winner_id"],
                loser_id=m["loser_id"],
                tourney_date=m["tourney_date"],
                surface=m["surface"] or "Hard",
                tourney_level=m["tourney_level"] or "D",
                round_name=m["round"] or "R32",
                best_of=m["best_of"],
                winner_rank=m["winner_rank"],
                loser_rank=m["loser_rank"],
            )

            if feat_result is None:
                continue

            features, label = feat_result
            feature_df = pd.DataFrame([features])

            # Align with model's expected features
            for col in model.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            feature_df = feature_df[model.feature_names].fillna(0)

            # Predict
            model_prob = float(model.predict_proba(feature_df)[0])
            predictions.append(model_prob)
            outcomes.append(label)

            # Use model probability as "fair price" since we don't have historical market prices
            # Simulate market at a less accurate price (add noise to represent market inefficiency)
            market_price_cents = int(model_prob * 100)

            # For backtesting, pretend market is slightly less efficient
            # Use actual label to determine "true" probability region
            # But size based on model vs a baseline (e.g., ranking-based prior)
            baseline_prob = 0.65 if label == 1.0 else 0.35  # rough baseline
            market_price_cents = max(1, min(99, int(baseline_prob * 100)))

            decision = self.sizer.size(model_prob, market_price_cents, ticker=m["tourney_id"])
            if decision is None:
                continue

            # Determine outcome
            if decision.side == "yes":
                won = label == 1.0
            else:
                won = label == 0.0

            if won:
                pnl = decision.bet_amount * (100 - market_price_cents) / market_price_cents
            else:
                pnl = -decision.bet_amount

            bankroll += pnl
            self.sizer.update_bankroll(bankroll)
            equity.append(bankroll)

            # Determine player names
            r1 = m["winner_rank"] if m["winner_rank"] else 9999
            r2 = m["loser_rank"] if m["loser_rank"] else 9999
            if r1 <= r2:
                p1_name = m["winner_name"] or str(m["winner_id"])
                p2_name = m["loser_name"] or str(m["loser_id"])
            else:
                p1_name = m["loser_name"] or str(m["loser_id"])
                p2_name = m["winner_name"] or str(m["winner_id"])

            bet = BetRecord(
                match_date=m["tourney_date"],
                p1_name=p1_name,
                p2_name=p2_name,
                side=decision.side,
                model_prob=model_prob,
                market_prob=market_price_cents / 100.0,
                edge=decision.edge,
                bet_amount=decision.bet_amount,
                outcome=1.0 if won else 0.0,
                pnl=pnl,
            )
            result.bets.append(bet)

        # Compute summary stats
        if result.bets:
            result.num_bets = len(result.bets)
            result.total_pnl = sum(b.pnl for b in result.bets)
            result.total_wagered = sum(b.bet_amount for b in result.bets)
            result.roi = result.total_pnl / result.total_wagered if result.total_wagered > 0 else 0
            result.win_rate = sum(1 for b in result.bets if b.pnl > 0) / result.num_bets
            result.avg_edge = np.mean([abs(b.edge) for b in result.bets])
            result.equity_curve = equity

            if predictions and outcomes:
                preds = np.array(predictions)
                outs = np.array(outcomes)
                result.brier_score = float(np.mean((preds - outs) ** 2))

            # Max drawdown
            peak = equity[0]
            max_dd = 0.0
            for val in equity:
                peak = max(peak, val)
                dd = (peak - val) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

            # Sharpe ratio (daily returns)
            if len(equity) > 1:
                returns = np.diff(equity) / np.array(equity[:-1])
                if returns.std() > 0:
                    result.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

        return result

    def _train_model(self, train_start: date, train_end: date) -> MatchPredictor | None:
        """Train a model on data in [train_start, train_end)."""
        df = self.builder.build_dataset(train_start, train_end)
        if df.empty or len(df) < 100:
            logger.warning("Insufficient training data: %d samples", len(df))
            return None

        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].fillna(0)
        y = df["label"]

        model = create_predictor(self.config.model.type)
        model.fit(X, y)
        logger.info("Trained model on %d samples (%s to %s)", len(X), train_start, train_end)
        return model
