"""Shared utilities: paths, reproducibility, and discrimination metrics.

Used by every stage of the pipeline so that a metric computed in
``train_pd.py`` means exactly the same thing in ``monitoring.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def ensure_outputs() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation of the two CDFs."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * ROC-AUC - 1 (computed directly from the AUC)."""
    from sklearn.metrics import roc_auc_score

    return float(2.0 * roc_auc_score(y_true, y_score) - 1.0)


def decile_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Risk deciles (1 = lowest predicted PD, 10 = highest)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    # qcut on rank so ties cannot create empty deciles.
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=False) + 1
    grouped = (
        df.groupby("decile")
        .agg(
            n=("y", "size"),
            defaults=("y", "sum"),
            avg_pd=("score", "mean"),
            min_pd=("score", "min"),
            max_pd=("score", "max"),
        )
        .reset_index()
    )
    grouped["default_rate"] = grouped["defaults"] / grouped["n"]
    grouped["share_of_defaults"] = grouped["defaults"] / grouped["defaults"].sum()
    grouped["cum_share_of_defaults"] = grouped["share_of_defaults"].cumsum()
    grouped["lift"] = grouped["default_rate"] / df["y"].mean()
    return grouped[
        ["decile", "n", "defaults", "default_rate", "avg_pd", "min_pd", "max_pd",
         "share_of_defaults", "cum_share_of_defaults", "lift"]
    ]
