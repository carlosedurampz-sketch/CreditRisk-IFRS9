"""PD scorecard: WoE logistic regression (auditable baseline) vs LightGBM.

Design choices, and why
-----------------------
* **Out-of-time split** (train: originated <= 2024-06, test: after) instead
  of a random split. A random split leaks the future into training and
  flatters every metric; in production a model is always scored on
  loans originated *after* it was built.
* **WoE encoding** for the logistic model: monotone, interpretable,
  and the standard a regulator can audit line by line. The WoE maps
  are fitted on train only.
* **Calibration**: the logistic model is wrapped in
  ``CalibratedClassifierCV`` (sigmoid / Platt scaling). A PD that is
  used in ``EL = PD x LGD x EAD`` must be a *probability*, not just a
  ranking — an uncalibrated score over- or under-states provisions.
* **LightGBM** is the challenger: stronger ranking on non-linear
  interactions, but a black box. It is compared on the *same* test
  set and the same metrics; if it wins on ranking but loses on
  calibration, the README says so explicitly.
* Metrics reported: ROC-AUC, PR-AUC (informative under class
  imbalance), KS, Gini, Brier score, and the full decile table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

from src.utils import (DATA_DIR, OUTPUTS_DIR, RANDOM_STATE, decile_table,
                       gini_coefficient, ks_statistic, save_json)
from src.woe import WoeEncoder

FEATURES_NUM = ["annual_income", "dti", "revol_util", "bureau_score",
                "emp_length_years", "delinq_2yrs", "int_rate", "loan_amnt"]
FEATURES_CAT = ["grade", "purpose", "term_months"]
TARGET = "default_12m"
CUTOFF = "2024-06"  # out-of-time split: train <= cutoff, test > cutoff


def load_loans(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "synthetic" / "loans.csv"
    df = pd.read_csv(path)
    df["origination_month"] = pd.PeriodIndex(df["origination_month"], freq="M")
    return df


def evaluate(y_true, y_score, name: str) -> dict:
    return {
        "model": name,
        "n": int(len(y_true)),
        "default_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": float(ks_statistic(np.asarray(y_true), np.asarray(y_score))),
        "gini": float(gini_coefficient(np.asarray(y_true), np.asarray(y_score))),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def plot_roc(y_true, scores: dict[str, np.ndarray], out: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve — out-of-time test set (synthetic, illustrative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_deciles(deciles: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(deciles["decile"], deciles["default_rate"])
    plt.plot(deciles["decile"], deciles["avg_pd"], "o-", color="darkred",
             label="Mean predicted PD")
    plt.xlabel("Risk decile (1 = lowest PD)")
    plt.ylabel("Observed default rate / predicted PD")
    plt.title("Decile calibration — observed vs predicted (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def train_and_evaluate(loans: pd.DataFrame, outputs: Path) -> dict:
    outputs.mkdir(parents=True, exist_ok=True)
    loans = loans.copy()
    loans["origination_month"] = pd.PeriodIndex(loans["origination_month"], freq="M")
    cutoff = pd.Period(CUTOFF, freq="M")
    train = loans[loans["origination_month"] <= cutoff].copy()
    test = loans[loans["origination_month"] > cutoff].copy()
    print(f"train: {len(train):,} loans (<= {CUTOFF}), "
          f"test: {len(test):,} loans (> {CUTOFF})")
    print(f"train default rate: {train[TARGET].mean():.2%}, "
          f"test default rate: {test[TARGET].mean():.2%}")

    features = FEATURES_NUM + FEATURES_CAT
    X_train_raw, y_train = train[features], train[TARGET]
    X_test_raw, y_test = test[features], test[TARGET]

    # --- WoE logistic regression (auditable baseline) ---
    encoder = WoeEncoder(n_bins=5)
    X_train_woe = encoder.fit_transform(X_train_raw, y_train)
    X_test_woe = encoder.transform(X_test_raw)
    iv_table = encoder.iv_table()
    iv_table.to_csv(outputs / "iv_table.csv", index=False)

    base_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    lr = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    lr.fit(X_train_woe, y_train)
    pred_lr = lr.predict_proba(X_test_woe)[:, 1]

    results: dict = {}
    results["logistic_woe"] = evaluate(y_test, pred_lr, "logistic_woe")
    scores = {"Logistic (WoE)": pred_lr}

    # --- LightGBM challenger (raw features, native categorical handling) ---
    try:
        import lightgbm as lgb

        X_train_lgb = X_train_raw.copy()
        X_test_lgb = X_test_raw.copy()
        for col in FEATURES_CAT:
            X_train_lgb[col] = X_train_lgb[col].astype("category")
            X_test_lgb[col] = X_test_lgb[col].astype("category")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbose=-1)
        lgb_model.fit(X_train_lgb, y_train)
        pred_lgb = lgb_model.predict_proba(X_test_lgb)[:, 1]
        results["lightgbm"] = evaluate(y_test, pred_lgb, "lightgbm")
        scores["LightGBM"] = pred_lgb
        importance = pd.DataFrame({
            "feature": features,
            "importance": lgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(outputs / "lightgbm_importance.csv", index=False)
    except Exception as exc:  # LightGBM is a challenger, not a hard dependency
        print(f"LightGBM skipped ({exc}); logistic model still reported.")
        results["lightgbm"] = {"model": "lightgbm", "status": "skipped", "error": str(exc)}

    # --- Champion: the calibrated logistic model (auditable, used for EL) ---
    deciles = decile_table(y_test.values, pred_lr)
    deciles.to_csv(outputs / "deciles_logistic.csv", index=False)
    plot_roc(y_test, scores, outputs / "roc_curve.png")
    plot_deciles(deciles, outputs / "decile_calibration.png")

    metrics = {
        "split": {"cutoff": CUTOFF, "train_n": int(len(train)), "test_n": int(len(test)),
                  "train_default_rate": float(train[TARGET].mean()),
                  "test_default_rate": float(test[TARGET].mean())},
        "models": results,
        "champion": "logistic_woe",
        "note": "Synthetic data — metrics are illustrative, not a real portfolio's.",
    }
    save_json(metrics, outputs / "pd_metrics.json")
    # Save test predictions so expected_loss.py and monitoring.py reuse them.
    test_out = test[["loan_id", "origination_month", "grade", "loan_amnt"]].copy()
    test_out["origination_month"] = test_out["origination_month"].astype(str)
    test_out["y_true"] = y_test.values
    test_out["pd_logistic"] = pred_lr
    if "LightGBM" in scores:
        test_out["pd_lightgbm"] = scores["LightGBM"]
    test_out.to_csv(outputs / "test_predictions.csv", index=False)
    for name, m in results.items():
        if "roc_auc" in m:
            print(f"{name:15s} ROC-AUC={m['roc_auc']:.3f}  KS={m['ks']:.3f}  "
                  f"Gini={m['gini']:.3f}  PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.3f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    train_and_evaluate(load_loans(args.data), args.outputs)


if __name__ == "__main__":
    main()
