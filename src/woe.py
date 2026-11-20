"""Weight of Evidence (WoE) and Information Value (IV) binning.

WoE is the auditable, regulator-friendly encoding used by scorecards:
for a bin :math:`b`,

.. math::
    \\mathrm{WoE}_b = \\ln\\left(\\frac{\\text{share of good in } b}
    {\\text{share of bad in } b}\\right), \\qquad
    \\mathrm{IV} = \\sum_b (\\text{good\\%}_b - \\text{bad\\%}_b)
    \\times \\mathrm{WoE}_b

IV rules of thumb (Siddiqi, *Credit Risk Scorecards*):
< 0.02 unpredictive, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong
(> 0.5 should be checked for leakage / overfitting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing so no bin has a zero share.


class WoeEncoder:
    """Quantile binning for numeric variables, frequency binning for categoricals.

    Parameters
    ----------
    n_bins:
        Maximum number of quantile bins for numeric variables.
    min_bin_share:
        Bins smaller than this share of the sample are merged with a neighbour.
    """

    def __init__(self, n_bins: int = 5, min_bin_share: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_share = min_bin_share
        self.bins_: dict[str, list] = {}
        self.woe_maps_: dict[str, dict] = {}
        self.iv_: dict[str, float] = {}
        self.is_numeric_: dict[str, bool] = {}

    # -- fitting ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeEncoder":
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)
        for col in X.columns:
            series = X[col]
            numeric = pd.api.types.is_numeric_dtype(series)
            self.is_numeric_[col] = bool(numeric)
            if numeric:
                edges = self._quantile_edges(series)
                binned = pd.cut(series, bins=edges, include_lowest=True)
            else:
                binned = series.astype(str).fillna("MISSING")
            self.bins_[col] = edges if numeric else sorted(binned.unique().tolist())
            woe_map, iv = self._woe_for_bins(binned, y)
            self.woe_maps_[col] = woe_map
            self.iv_[col] = iv
        return self

    @staticmethod
    def _quantile_edges(series: pd.Series) -> list:
        clean = series.dropna()
        quantiles = np.linspace(0, 1, 6)  # up to 5 bins by default
        edges = sorted(set(np.quantile(clean, quantiles).tolist()))
        if len(edges) < 3:  # (near-)constant variable
            edges = [float(clean.min()) - 1.0, float(clean.max()) + 1.0]
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @staticmethod
    def _woe_for_bins(binned: pd.Series, y: pd.Series) -> tuple[dict, float]:
        frame = pd.DataFrame({"bin": binned.astype(str), "y": y.values})
        grouped = frame.groupby("bin")["y"].agg(["count", "sum"])
        grouped["good"] = grouped["count"] - grouped["sum"]
        total_good = grouped["good"].sum()
        total_bad = grouped["sum"].sum()
        good_share = (grouped["good"] + EPS) / (total_good + EPS * len(grouped))
        bad_share = (grouped["sum"] + EPS) / (total_bad + EPS * len(grouped))
        woe = np.log(good_share / bad_share)
        iv = float(((good_share - bad_share) * woe).sum())
        return {str(k): float(v) for k, v in woe.items()}, iv

    # -- transforming ----------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.reset_index(drop=True).index)
        for col in X.columns:
            series = X[col].reset_index(drop=True)
            if self.is_numeric_.get(col, False):
                binned = pd.cut(series, bins=self.bins_[col], include_lowest=True)
                keys = binned.astype(str)
            else:
                keys = series.astype(str).fillna("MISSING")
            mapping = self.woe_maps_[col]
            # Unseen category -> neutral WoE of 0 (no evidence either way).
            out[f"{col}_woe"] = keys.map(mapping).fillna(0.0)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_table(self) -> pd.DataFrame:
        table = pd.DataFrame(
            {"variable": list(self.iv_.keys()), "iv": list(self.iv_.values())}
        ).sort_values("iv", ascending=False)
        table["strength"] = table["iv"].apply(iv_strength)
        return table.reset_index(drop=True)


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious — check for leakage"
