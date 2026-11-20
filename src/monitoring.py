"""Population Stability Index (PSI) and model monitoring.

PSI compares the distribution of a score (here the predicted PD,
binned into deciles fixed on the training / reference population)
between a reference period and a later period:

.. math::
    PSI = \\sum_b (\\text{actual\\%}_b - \\text{expected\\%}_b)
          \\times \\ln\\left(\\frac{\\text{actual\\%}_b}
          {\\text{expected\\%}_b}\\right)

Interpretation (standard industry thresholds):
PSI < 0.10 — no significant change; 0.10-0.25 — moderate shift,
investigate; > 0.25 — significant shift, consider recalibration.

In this project the "reference" is the test set's earliest
origination months and each later month is compared against it,
using the same decile edges, so a rising PSI means the incoming
population (or the score's behaviour on it) is drifting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import OUTPUTS_DIR, save_json

EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two distributions over the same bins (shares, sum to 1)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_by_period(scores: pd.Series, periods: pd.Series, bins: np.ndarray) -> pd.DataFrame:
    """PSI per period vs the first (reference) period, using fixed bins."""
    ref_period = periods.min()
    ref = scores[periods == ref_period]
    ref_hist, _ = np.histogram(ref, bins=bins)
    ref_share = ref_hist / max(ref_hist.sum(), 1)
    rows = []
    for period in sorted(periods.unique()):
        actual = scores[periods == period]
        hist, _ = np.histogram(actual, bins=bins)
        share = hist / max(hist.sum(), 1)
        value = psi(ref_share, share)
        rows.append({
            "period": str(period),
            "n": int(len(actual)),
            "psi": value,
            "status": "stable" if value < 0.10 else (
                "investigate" if value < 0.25 else "significant shift"),
            "mean_pd": float(actual.mean()),
            "default_rate": None,  # filled by caller if y is available
        })
    return pd.DataFrame(rows)


def run(predictions_path: Path, outputs: Path) -> dict[str, object]:
    outputs.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(predictions_path)
    preds["origination_month"] = pd.PeriodIndex(preds["origination_month"], freq="M")
    # Decile edges fixed on the reference (earliest) period.
    ref = preds[preds["origination_month"] == preds["origination_month"].min()]
    bins = np.quantile(ref["pd_logistic"], np.linspace(0, 1, 11))
    bins = np.unique(bins)
    bins[0], bins[-1] = 0.0, 1.0
    table = psi_by_period(preds["pd_logistic"], preds["origination_month"], bins)
    # Attach observed default rate per period for context.
    rates = preds.groupby("origination_month")["y_true"].mean()
    table["default_rate"] = [float(rates[pd.Period(p, freq="M")]) for p in table["period"]]
    table.to_csv(outputs / "psi_monthly.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(table["period"], table["psi"], "o-")
    plt.axhline(0.10, color="orange", linestyle="--", label="0.10 investigate")
    plt.axhline(0.25, color="red", linestyle="--", label="0.25 significant shift")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("PSI")
    plt.title("Monthly PSI vs reference period (synthetic, illustrative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs / "psi_monthly.png", dpi=150)
    plt.close()

    result = {
        "reference_period": str(preds["origination_month"].min()),
        "max_psi": float(table["psi"].max()),
        "mean_psi": float(table["psi"].mean()),
        "months_flagged": int((table["psi"] >= 0.10).sum()),
        "monthly": table.to_dict(orient="records"),
        "note": "Synthetic data — PSI values are illustrative.",
    }
    save_json(result, outputs / "monitoring.json")
    print(table.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUTPUTS_DIR / "test_predictions.csv")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    run(args.predictions, args.outputs)


if __name__ == "__main__":
    main()
