"""Match prediction models with pluggable backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MatchPredictor(ABC):
    """Abstract base class for match prediction models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> MatchPredictor: ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]: ...


class LogisticPredictor(MatchPredictor):
    """Logistic regression with StandardScaler pipeline."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
        ])
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.pipeline.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(label=1) for each sample."""
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "features": self._feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> LogisticPredictor:
        data = joblib.load(path)
        predictor = cls()
        predictor.pipeline = data["pipeline"]
        predictor._feature_names = data["features"]
        return predictor

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def feature_importance(self) -> dict[str, float]:
        """Get absolute coefficient values as importance proxy."""
        clf = self.pipeline.named_steps["clf"]
        coefs = np.abs(clf.coef_[0])
        return dict(sorted(zip(self._feature_names, coefs), key=lambda x: -x[1]))


def create_predictor(model_type: str = "logistic") -> MatchPredictor:
    """Factory function to create a predictor by type."""
    if model_type == "logistic":
        return LogisticPredictor()
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Install xgboost: pip install tennis-edge[boost]")
        # XGBoost predictor would go here
        raise NotImplementedError("XGBoost predictor not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
