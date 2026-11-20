"""Generate realistic *synthetic* credit data — no real borrowers.

Why synthetic?
    The suggested real datasets (LendingClub, Home Credit) require a
    Kaggle / external download and contain real borrower data. This
    project must run offline, reproducibly, with ``python -m src.run_pipeline``.
    The generator below encodes the qualitative relationships a risk
    analyst expects (higher DTI / utilisation / rate -> higher default
    risk; higher income / bureau score -> lower risk) without claiming
    any real portfolio's numbers.

Data-generating process (documented, so it can be audited)
----------------------------------------------------------
1. ``n_loans`` loans originated monthly between 2022-01 and 2025-06.
2. A latent log-odds of 12-month default is a linear combination of
   standardised risk drivers plus noise; ``default_12m`` is a Bernoulli
   draw from that probability. Overall 12-month default rate ≈ 8-12%.
3. Monthly delinquency snapshots (months on book 0-24) follow a
   simple Markov-style process: the probability of worsening a bucket
   rises with the loan's latent risk and with months on book, and
   90+ DPD is (almost) absorbing, as in real roll-rate matrices.
4. Realised LGD for defaulted loans is drawn around 45% (secured-like
   low / unsecured-like high by grade), clipped to [5%, 95%].

Outputs
-------
``data/synthetic/loans.csv``      — one row per loan (origination + outcome)
``data/synthetic/snapshots.csv``  — one row per loan-month (for SQL vintage / roll-rate work)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, RANDOM_STATE

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_RATE = {"A": 0.07, "B": 0.10, "C": 0.13, "D": 0.17, "E": 0.21, "F": 0.25, "G": 0.29}
GRADE_LGD_MEAN = {"A": 0.35, "B": 0.40, "C": 0.45, "D": 0.50, "E": 0.55, "F": 0.60, "G": 0.65}
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "other"]
BUCKETS = ["Current", "30", "60", "90+"]


def generate_loans(n_loans: int = 20_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    orig_months = pd.period_range("2022-01", "2025-06", freq="M")
    origination = rng.choice(orig_months, size=n_loans)
    annual_income = np.clip(rng.lognormal(mean=np.log(55_000), sigma=0.55, size=n_loans),
                            12_000, 400_000)
    loan_amnt = np.clip(rng.lognormal(mean=np.log(15_000), sigma=0.6, size=n_loans),
                        1_000, 40_000)
    dti = np.clip(rng.normal(18, 8, n_loans) + (loan_amnt / annual_income) * 30, 0, 45)
    revol_util = np.clip(rng.beta(2.5, 5.0, n_loans) * 100, 0, 100)
    bureau_score = np.clip(
        rng.normal(690, 55, n_loans) - 1.0 * (dti - 18) - 0.4 * (revol_util - 35),
        500, 850).astype(int)
    # Grade is assigned FROM the bureau score (as in a real book), so the
    # grade ranking is monotone in risk by construction — and SQL KPI 3
    # in sql/03_portfolio_kpis.sql can verify that monotonicity.
    grade = np.select(
        [bureau_score >= 750, bureau_score >= 720, bureau_score >= 690,
         bureau_score >= 660, bureau_score >= 630, bureau_score >= 600],
        ["A", "B", "C", "D", "E", "F"], default="G")
    int_rate = np.array([GRADE_RATE[g] for g in grade]) + rng.normal(0, 0.015, n_loans)
    int_rate = np.clip(int_rate, 0.05, 0.35)
    emp_length = rng.integers(0, 21, n_loans)
    delinq_2yrs = rng.poisson(0.3, n_loans)
    term_months = rng.choice([36, 60], size=n_loans, p=[0.7, 0.3])
    purpose = rng.choice(PURPOSES, size=n_loans,
                         p=[0.45, 0.20, 0.12, 0.08, 0.05, 0.10])

    # Latent log-odds — signs are the economically expected ones.
    logit = (
        -3.0
        + 0.045 * (dti - 18)
        + 0.012 * (revol_util - 35)
        + 0.020 * (850 - bureau_score) / 10
        + 0.25 * delinq_2yrs
        - 0.03 * emp_length
        + 2.0 * (int_rate - 0.13)
        + rng.normal(0, 0.40, n_loans)  # unobserved heterogeneity
    )
    pd_default = 1 / (1 + np.exp(-logit))
    default_12m = (rng.random(n_loans) < pd_default).astype(int)

    lgd = np.where(
        default_12m == 1,
        np.clip(
            np.array([GRADE_LGD_MEAN[g] for g in grade])
            + rng.normal(0, 0.12, n_loans), 0.05, 0.95),
        np.nan,
    )
    months_to_default = np.where(
        default_12m == 1, rng.integers(3, 13, n_loans), np.nan)

    loans = pd.DataFrame({
        "loan_id": np.arange(1, n_loans + 1),
        "origination_month": pd.PeriodIndex(origination, freq="M").astype(str),
        "grade": grade,
        "term_months": term_months,
        "loan_amnt": np.round(loan_amnt, 2),
        "int_rate": np.round(int_rate, 4),
        "annual_income": np.round(annual_income, 2),
        "dti": np.round(dti, 2),
        "revol_util": np.round(revol_util, 2),
        "bureau_score": bureau_score,
        "emp_length_years": emp_length,
        "delinq_2yrs": delinq_2yrs,
        "purpose": purpose,
        "pd_latent": np.round(pd_default, 4),
        "default_12m": default_12m,
        "months_to_default": months_to_default,
        "lgd_realised": np.round(lgd, 4),
    })
    return loans


def generate_snapshots(loans: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Monthly DPD snapshots, months on book 0..24, per loan."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    # Worsening probability grows with latent risk; 90+ is near-absorbing.
    for loan in loans.itertuples(index=False):
        risk = float(loan.pd_latent)
        bucket = 0  # 0=Current, 1=30, 2=60, 3=90+
        for mob in range(0, 25):
            p_worse = np.clip(0.02 + 0.35 * risk + 0.002 * mob, 0, 0.45)
            p_better = 0.45 if bucket > 0 else 0.0
            roll = rng.random()
            if bucket < 3 and roll < p_worse:
                bucket += 1
            elif bucket > 0 and roll < p_worse + p_better:
                bucket -= 1
            if loan.default_12m == 1 and mob >= int(loan.months_to_default):
                bucket = max(bucket, 3)
            rows.append({
                "loan_id": int(loan.loan_id),
                "origination_month": loan.origination_month,
                "months_on_book": mob,
                "dpd_bucket": BUCKETS[bucket],
                "dpd_days": [0, 30, 60, 95][bucket],
                "outstanding_balance": round(
                    float(loan.loan_amnt) * max(0.0, 1 - mob / (int(loan.term_months) + 6)), 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-loans", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "synthetic")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loans = generate_loans(args.n_loans, args.seed)
    snapshots = generate_snapshots(loans, args.seed)
    loans.to_csv(args.out_dir / "loans.csv", index=False)
    snapshots.to_csv(args.out_dir / "snapshots.csv", index=False)
    print(f"loans:     {len(loans):,} rows  -> {args.out_dir / 'loans.csv'}")
    print(f"snapshots: {len(snapshots):,} rows  -> {args.out_dir / 'snapshots.csv'}")
    print(f"12m default rate: {loans['default_12m'].mean():.2%}")
    print(f"mean loan amount: ${loans['loan_amnt'].mean():,.0f}")


if __name__ == "__main__":
    main()
