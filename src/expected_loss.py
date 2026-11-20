"""Expected loss under IFRS 9 / Basel logic: EL = PD x LGD x EAD.

Formulas (LaTeX in the README)
-------------------------------
For a single exposure:

.. math::
    EL = PD \\times LGD \\times EAD

* **PD** — 12-month probability of default from the calibrated
  scorecard (``train_pd.py``). For lifetime / Stage 2 the same
  formula is applied with a lifetime PD; here the 12-month PD is
  used and the staging logic is documented as a simplification.
* **LGD** — loss given default. Estimated here as the realised LGD
  mean by grade on defaulted loans in train (a real implementation
  would fit a workout-LGD model on recoveries, discounted cash
  flows). Floored at 5% and capped at 95%.
* **EAD** — exposure at default. Here the outstanding balance at
  the latest snapshot (``loan_amnt`` at origination is the upper
  bound; the pipeline uses the outstanding balance at month 12
  from ``snapshots.csv`` when available, otherwise ``loan_amnt``).

Stress testing
--------------
Macro scenarios scale PD and shift LGD, as a bank's stress test
would: base (x1.0), adverse (PD x1.5, LGD +10 pp), severe
(PD x2.0, LGD +20 pp). The point is the *sensitivity* of EL to
the scenario, not a forecast of any real macro path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, OUTPUTS_DIR, save_json

SCENARIOS = {
    "base": {"pd_mult": 1.0, "lgd_add": 0.00},
    "adverse": {"pd_mult": 1.5, "lgd_add": 0.10},
    "severe": {"pd_mult": 2.0, "lgd_add": 0.20},
}


def estimate_lgd_by_grade(loans: pd.DataFrame) -> tuple[pd.Series, float]:
    """Mean realised LGD per grade on defaulted loans; global mean as fallback."""
    defaulted = loans[loans["default_12m"] == 1]
    lgd = defaulted.groupby("grade")["lgd_realised"].mean()
    global_lgd = float(defaulted["lgd_realised"].mean())
    return lgd, global_lgd


def compute_el(df: pd.DataFrame, pd_col: str = "pd_logistic",
               lgd_by_grade: pd.Series | None = None,
               global_lgd: float = 0.45) -> pd.DataFrame:
    """Add ``lgd``, ``ead`` and ``el`` columns. Pure function — unit-tested."""
    out = df.copy()
    if lgd_by_grade is not None:
        out["lgd"] = out["grade"].map(lgd_by_grade).fillna(global_lgd)
    else:
        out["lgd"] = global_lgd
    out["lgd"] = out["lgd"].clip(0.05, 0.95)
    out["ead"] = out["ead"] if "ead" in out else out["loan_amnt"]
    out["pd"] = out[pd_col].clip(0.0, 0.9999) if pd_col in out else out["pd"]
    out["el"] = out["pd"] * out["lgd"] * out["ead"]
    return out


def stress_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, scen in SCENARIOS.items():
        stressed = df.copy()
        stressed["pd"] = (stressed["pd"] * scen["pd_mult"]).clip(upper=0.99)
        stressed["lgd"] = (stressed["lgd"] + scen["lgd_add"]).clip(0.05, 0.95)
        stressed["el"] = stressed["pd"] * stressed["lgd"] * stressed["ead"]
        rows.append({
            "scenario": name,
            "pd_multiplier": scen["pd_mult"],
            "lgd_add_pp": scen["lgd_add"] * 100,
            "total_el": float(stressed["el"].sum()),
            "total_ead": float(stressed["ead"].sum()),
            "el_rate_bps": float(stressed["el"].sum() / stressed["ead"].sum() * 10_000),
            "mean_pd": float(stressed["pd"].mean()),
        })
    return pd.DataFrame(rows)


def run(loans_path: Path, predictions_path: Path, snapshots_path: Path | None,
        outputs: Path) -> dict[str, object]:
    outputs.mkdir(parents=True, exist_ok=True)
    loans = pd.read_csv(loans_path)
    preds = pd.read_csv(predictions_path)
    lgd_by_grade, global_lgd = estimate_lgd_by_grade(loans)

    # EAD: outstanding balance at month 12 if snapshots exist, else loan amount.
    ead = loans.set_index("loan_id")["loan_amnt"]
    if snapshots_path and snapshots_path.exists():
        snaps = pd.read_csv(snapshots_path)
        m12 = snaps[snaps["months_on_book"] == 12].set_index("loan_id")["outstanding_balance"]
        ead = m12.reindex(ead.index).fillna(ead)
    df = preds.copy()
    df["ead"] = df["loan_id"].map(ead)
    df = compute_el(df, pd_col="pd_logistic", lgd_by_grade=lgd_by_grade,
                    global_lgd=global_lgd)

    by_grade = (df.groupby("grade")
                  .agg(n=("el", "size"), ead=("ead", "sum"), el=("el", "sum"),
                       mean_pd=("pd", "mean"), mean_lgd=("lgd", "mean"))
                  .reset_index())
    by_grade["el_rate_bps"] = by_grade["el"] / by_grade["ead"] * 10_000
    by_grade = by_grade.sort_values("grade")
    by_grade.to_csv(outputs / "el_by_grade.csv", index=False)

    stress = stress_table(df)
    stress.to_csv(outputs / "el_stress.csv", index=False)

    result = {
        "total_ead": float(df["ead"].sum()),
        "total_el": float(df["el"].sum()),
        "el_rate_bps": float(df["el"].sum() / df["ead"].sum() * 10_000),
        "mean_pd": float(df["pd"].mean()),
        "mean_lgd": float(df["lgd"].mean()),
        "global_lgd_fallback": float(global_lgd),
        "lgd_by_grade": {k: float(v) for k, v in lgd_by_grade.items()},
        "stress": stress.to_dict(orient="records"),
        "note": "Synthetic data — EL figures are illustrative, not a real portfolio's.",
    }
    save_json(result, outputs / "expected_loss.json")
    print(f"Total EAD: ${result['total_ead']:,.0f}  |  Total EL: ${result['total_el']:,.0f}  "
          f"|  EL rate: {result['el_rate_bps']:.0f} bps")
    print(stress.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loans", type=Path, default=DATA_DIR / "synthetic" / "loans.csv")
    parser.add_argument("--predictions", type=Path, default=OUTPUTS_DIR / "test_predictions.csv")
    parser.add_argument("--snapshots", type=Path, default=DATA_DIR / "synthetic" / "snapshots.csv")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    run(args.loans, args.predictions, args.snapshots, args.outputs)


if __name__ == "__main__":
    main()
