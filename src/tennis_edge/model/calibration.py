"""Model calibration: Brier score and reliability diagrams."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better, 0 = perfect)."""
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Compute calibration data: predicted vs actual rates per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    rows = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        count = mask.sum()
        if count == 0:
            rows.append({
                "bin_center": bin_centers[i],
                "mean_predicted": bin_centers[i],
                "mean_actual": np.nan,
                "count": 0,
            })
        else:
            rows.append({
                "bin_center": bin_centers[i],
                "mean_predicted": y_prob[mask].mean(),
                "mean_actual": y_true[mask].mean(),
                "count": int(count),
            })

    return pd.DataFrame(rows)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): weighted average of |predicted - actual| per bin."""
    table = calibration_table(y_true, y_prob, n_bins)
    valid = table.dropna(subset=["mean_actual"])
    if valid.empty:
        return 1.0
    total = valid["count"].sum()
    ece = (valid["count"] / total * (valid["mean_predicted"] - valid["mean_actual"]).abs()).sum()
    return float(ece)
