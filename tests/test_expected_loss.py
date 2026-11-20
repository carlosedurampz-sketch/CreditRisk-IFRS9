"""Tests for src/expected_loss.py — EL = PD x LGD x EAD and stress scenarios."""
import pandas as pd

from src.expected_loss import compute_el, stress_table
from src.monitoring import psi


def _portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "grade": ["A", "B", "A"],
        "loan_amnt": [10_000.0, 20_000.0, 5_000.0],
        "pd_logistic": [0.02, 0.10, 0.05],
    })


def test_el_formula_is_pd_times_lgd_times_ead() -> None:
    df = compute_el(_portfolio(), lgd_by_grade=pd.Series({"A": 0.4, "B": 0.5}))
    # Loan 1: 0.02 * 0.40 * 10,000 = 80 ; loan 2: 0.10 * 0.50 * 20,000 = 1,000
    assert df["el"].iloc[0] == 80.0
    assert df["el"].iloc[1] == 1_000.0
    assert df["el"].iloc[2] == 0.05 * 0.4 * 5_000


def test_el_zero_pd_means_zero_loss_and_lgd_is_clipped() -> None:
    df = compute_el(_portfolio(), lgd_by_grade=pd.Series({"A": 2.0, "B": -1.0}))
    assert df["lgd"].between(0.05, 0.95).all()
    zero_pd = pd.DataFrame({"grade": ["A"], "loan_amnt": [10_000.0], "pd_logistic": [0.0]})
    assert compute_el(zero_pd, lgd_by_grade=pd.Series({"A": 0.4}))["el"].iloc[0] == 0.0


def test_stress_scenarios_increase_el_monotonically() -> None:
    df = compute_el(_portfolio(), lgd_by_grade=pd.Series({"A": 0.4, "B": 0.5}))
    stress = stress_table(df)
    assert list(stress["scenario"]) == ["base", "adverse", "severe"]
    els = stress["total_el"].tolist()
    assert els[0] < els[1] < els[2]
    # Severe EL rate must be higher than base EL rate.
    assert stress["el_rate_bps"].iloc[2] > stress["el_rate_bps"].iloc[0]


def test_psi_zero_for_identical_distributions_and_positive_for_shift() -> None:
    expected = [0.1] * 10
    assert psi(expected, expected) == 0.0
    shifted = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91]
    assert psi(expected, shifted) > 0.25  # a large shift
