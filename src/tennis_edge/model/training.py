"""Model training with proper train/val/test splits, hyperparameter tuning, and walk-forward CV."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.inspection import permutation_importance

from ..config import AppConfig
from ..features.builder import FeatureBuilder
from .calibration import brier_score, calibration_error
from .predictor import MatchPredictor, LogisticPredictor, create_predictor

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    name: str
    start: str
    end: str
    samples: int


@dataclass
class CVFoldResult:
    fold: int
    train_size: int
    val_size: int
    train_period: str
    val_period: str
    best_C: float
    accuracy: float
    brier_score: float
    auc_roc: float
    log_loss: float
    ece: float


@dataclass
class TrainingReport:
    """Complete training report with all metrics and diagnostics."""
    # Data splits
    splits: list[SplitInfo] = field(default_factory=list)

    # Walk-forward CV results
    cv_folds: list[CVFoldResult] = field(default_factory=list)
    cv_mean_accuracy: float = 0.0
    cv_mean_brier: float = 0.0
    cv_mean_auc: float = 0.0
    cv_std_accuracy: float = 0.0

    # Final model (retrained on train+val, evaluated on test)
    final_best_C: float = 1.0
    test_accuracy: float = 0.0
    test_brier: float = 0.0
    test_auc: float = 0.0
    test_log_loss: float = 0.0
    test_ece: float = 0.0
    baseline_accuracy: float = 0.0

    # Feature importance (permutation-based)
    feature_importance: dict[str, float] = field(default_factory=dict)
    coefficient_importance: dict[str, float] = field(default_factory=dict)

    # Model info
    model_type: str = "logistic"
    total_features: int = 0
    model_path: str = ""


class ModelTrainer:
    """Train and evaluate match prediction models with proper validation."""

    # Hyperparameter grid
    C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    def __init__(self, config: AppConfig, feature_builder: FeatureBuilder):
        self.config = config
        self.builder = feature_builder

    def train_and_evaluate(self) -> TrainingReport:
        """Full training pipeline:

        1. Build dataset
        2. Split: Train (2005-2017) / Val (2017-2020) / Test (2020-2026)
        3. Walk-forward CV on Train+Val with hyperparameter tuning
        4. Select best hyperparameters
        5. Retrain final model on Train+Val
        6. Evaluate on held-out Test set
        7. Compute permutation importance
        """
        report = TrainingReport(model_type=self.config.model.type)

        # ── Step 1: Define temporal splits ──
        # Train 2005-2020, Validation 2021-2023, Test 2024-2026
        train_start = date(self.config.model.train_start_year, 1, 1)  # 2005
        val_start = date(2021, 1, 1)
        test_start = date(2024, 1, 1)
        test_end = date.today()

        # ── Step 2: Build datasets ──
        logger.info("Building train set: %s to %s", train_start, val_start)
        train_df = self.builder.build_dataset(train_start, val_start)

        logger.info("Building val set: %s to %s", val_start, test_start)
        val_df = self.builder.build_dataset(val_start, test_start)

        logger.info("Building test set: %s to %s", test_start, test_end)
        test_df = self.builder.build_dataset(test_start, test_end)

        if train_df.empty or val_df.empty or test_df.empty:
            logger.error("Insufficient data: train=%d val=%d test=%d",
                        len(train_df), len(val_df), len(test_df))
            return report

        feature_cols = [c for c in train_df.columns if c != "label"]
        report.total_features = len(feature_cols)

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["label"]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["label"]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["label"]

        report.splits = [
            SplitInfo("Train", str(train_start), str(val_start), len(X_train)),
            SplitInfo("Validation", str(val_start), str(test_start), len(X_val)),
            SplitInfo("Test", str(test_start), str(test_end), len(X_test)),
        ]

        logger.info("Splits: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

        # ── Step 3: Hyperparameter tuning on val set ──
        logger.info("Tuning hyperparameters on validation set...")
        best_C, best_brier, tuning_results = self._tune_hyperparams(
            X_train, y_train, X_val, y_val
        )
        logger.info("Best C=%.4f (val Brier=%.4f)", best_C, best_brier)

        for C, metrics in tuning_results:
            logger.info("  C=%-8.4f  acc=%.3f  brier=%.4f  auc=%.3f",
                       C, metrics["accuracy"], metrics["brier"], metrics["auc"])

        # ── Step 4: Walk-forward CV ──
        logger.info("Running walk-forward cross-validation...")
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        report.cv_folds = self._walk_forward_cv(trainval_df, feature_cols, n_splits=5)

        if report.cv_folds:
            accs = [f.accuracy for f in report.cv_folds]
            briers = [f.brier_score for f in report.cv_folds]
            aucs = [f.auc_roc for f in report.cv_folds]
            report.cv_mean_accuracy = float(np.mean(accs))
            report.cv_mean_brier = float(np.mean(briers))
            report.cv_mean_auc = float(np.mean(aucs))
            report.cv_std_accuracy = float(np.std(accs))

        # ── Step 5: Train final model on train+val with best C ──
        logger.info("Training final model on train+val (C=%.4f)...", best_C)
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)

        final_model = LogisticPredictor(C=best_C)
        final_model.fit(X_trainval, y_trainval)

        # ── Step 6: Evaluate on held-out test set ──
        y_prob = final_model.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        report.final_best_C = best_C
        report.test_accuracy = float(accuracy_score(y_test, y_pred))
        report.test_brier = brier_score(y_test.values, y_prob)
        report.test_auc = float(roc_auc_score(y_test, y_prob))
        report.test_log_loss = float(log_loss(y_test, y_prob))
        report.test_ece = calibration_error(y_test.values, y_prob)
        report.baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))

        # ── Step 7: Feature importance ──
        # Coefficient-based (after scaling)
        report.coefficient_importance = final_model.feature_importance()

        # Permutation-based (more reliable, measures actual impact on predictions)
        logger.info("Computing permutation importance on test set...")
        perm_result = permutation_importance(
            final_model.pipeline, X_test, y_test,
            n_repeats=10, random_state=42, scoring="neg_brier_score",
        )
        perm_imp = dict(zip(feature_cols, perm_result.importances_mean))
        report.feature_importance = dict(sorted(perm_imp.items(), key=lambda x: -x[1]))

        # ── Step 8: Save model ──
        artifacts_dir = Path(self.config.project_root) / self.config.model.artifacts_dir
        model_path = artifacts_dir / "latest.joblib"
        final_model.save(model_path)
        report.model_path = str(model_path)
        logger.info("Model saved to %s", model_path)

        return report

    def _tune_hyperparams(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> tuple[float, float, list[tuple[float, dict]]]:
        """Grid search over C values, evaluated on validation set.

        Returns: (best_C, best_brier, [(C, metrics_dict), ...])
        """
        results: list[tuple[float, dict]] = []
        best_C = 1.0
        best_brier = 1.0

        for C in self.C_VALUES:
            model = LogisticPredictor(C=C)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "brier": brier_score(y_val.values, y_prob),
                "auc": float(roc_auc_score(y_val, y_prob)),
                "log_loss": float(log_loss(y_val, y_prob)),
                "ece": calibration_error(y_val.values, y_prob),
            }
            results.append((C, metrics))

            if metrics["brier"] < best_brier:
                best_brier = metrics["brier"]
                best_C = C

        return best_C, best_brier, results

    def _walk_forward_cv(
        self,
        full_df: pd.DataFrame,
        feature_cols: list[str],
        n_splits: int = 5,
    ) -> list[CVFoldResult]:
        """Expanding-window walk-forward CV with per-fold hyperparameter tuning.

        Each fold:
        - Train on data[:split_point]
        - Tune C on a small val window within training data
        - Evaluate on data[split_point:next_split_point]
        """
        n = len(full_df)
        min_train = n // (n_splits + 1)  # minimum training size

        folds: list[CVFoldResult] = []

        for i in range(n_splits):
            split = min_train + (i * (n - min_train) // n_splits)
            next_split = min(split + (n - min_train) // n_splits, n)

            if next_split <= split:
                continue

            X_fold_train = full_df.iloc[:split][feature_cols].fillna(0)
            y_fold_train = full_df.iloc[:split]["label"]
            X_fold_val = full_df.iloc[split:next_split][feature_cols].fillna(0)
            y_fold_val = full_df.iloc[split:next_split]["label"]

            if len(X_fold_val) == 0:
                continue

            # Quick tune: use last 20% of training as internal val
            internal_split = int(len(X_fold_train) * 0.8)
            X_it = X_fold_train.iloc[:internal_split]
            y_it = y_fold_train.iloc[:internal_split]
            X_iv = X_fold_train.iloc[internal_split:]
            y_iv = y_fold_train.iloc[internal_split:]

            best_C = 1.0
            best_brier = 1.0
            for C in [0.01, 0.1, 1.0, 10.0]:
                m = LogisticPredictor(C=C)
                m.fit(X_it, y_it)
                prob = m.predict_proba(X_iv)
                b = brier_score(y_iv.values, prob)
                if b < best_brier:
                    best_brier = b
                    best_C = C

            # Train on full fold training data with best C
            model = LogisticPredictor(C=best_C)
            model.fit(X_fold_train, y_fold_train)
            y_prob = model.predict_proba(X_fold_val)
            y_pred = (y_prob >= 0.5).astype(int)

            # Date ranges for reporting
            train_dates = full_df.iloc[:split]["label"].index
            val_dates = full_df.iloc[split:next_split]["label"].index

            fold_result = CVFoldResult(
                fold=i + 1,
                train_size=len(X_fold_train),
                val_size=len(X_fold_val),
                train_period=f"rows 0-{split}",
                val_period=f"rows {split}-{next_split}",
                best_C=best_C,
                accuracy=float(accuracy_score(y_fold_val, y_pred)),
                brier_score=brier_score(y_fold_val.values, y_prob),
                auc_roc=float(roc_auc_score(y_fold_val, y_prob)),
                log_loss=float(log_loss(y_fold_val, y_prob)),
                ece=calibration_error(y_fold_val.values, y_prob),
            )
            folds.append(fold_result)

            logger.info(
                "  Fold %d: train=%d val=%d C=%.2f acc=%.3f brier=%.4f auc=%.3f",
                fold_result.fold, fold_result.train_size, fold_result.val_size,
                fold_result.best_C, fold_result.accuracy, fold_result.brier_score,
                fold_result.auc_roc,
            )

        return folds
