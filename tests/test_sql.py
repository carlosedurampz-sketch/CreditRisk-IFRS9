"""Tests for scripts/check_sql.py — SQL validation integrated into pytest.

Each test builds a small loans/snapshots fixture in a temporary directory
and runs the real sql/*.sql files against it in an in-memory SQLite
database, so CI validates the SQL on every push without the full
synthetic dataset.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from scripts.check_sql import build_database, run_sql_file, validate_sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"

LOAN_COLUMNS = [
    "loan_id", "origination_month", "grade", "term_months", "loan_amnt",
    "int_rate", "annual_income", "dti", "revol_util", "bureau_score",
    "emp_length_years", "delinq_2yrs", "purpose", "pd_latent",
    "default_12m", "months_to_default", "lgd_realised",
]
SNAPSHOT_COLUMNS = [
    "loan_id", "origination_month", "months_on_book", "dpd_bucket",
    "dpd_days", "outstanding_balance",
]


def _fixture_data(data_dir: Path) -> Path:
    """Write a minimal loans/snapshots fixture; return the data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    loans = [
        [1, "2022-01", "A", 36, 10_000, 0.07, 60_000, 10, 20, 780, 5, 0,
         "credit_card", 0.02, 0, "", ""],
        [2, "2022-01", "B", 36, 15_000, 0.10, 50_000, 18, 35, 720, 3, 0,
         "debt_consolidation", 0.05, 0, "", ""],
        [3, "2022-01", "C", 60, 20_000, 0.13, 45_000, 25, 50, 680, 2, 1,
         "home_improvement", 0.12, 1, 6, 0.45],
        [4, "2022-01", "G", 36, 8_000, 0.29, 30_000, 35, 70, 580, 1, 2,
         "small_business", 0.25, 1, 9, 0.65],
        [5, "2022-02", "A", 36, 12_000, 0.07, 70_000, 12, 15, 790, 8, 0,
         "credit_card", 0.02, 0, "", ""],
        [6, "2022-02", "B", 60, 18_000, 0.10, 55_000, 20, 40, 710, 4, 0,
         "debt_consolidation", 0.06, 0, "", ""],
        [7, "2022-02", "C", 36, 9_000, 0.13, 40_000, 28, 55, 660, 2, 1,
         "other", 0.15, 1, 5, 0.50],
        [8, "2022-02", "G", 36, 11_000, 0.29, 32_000, 38, 75, 570, 0, 3,
         "major_purchase", 0.30, 1, 8, 0.70],
    ]
    snapshots = []
    for loan in loans:
        loan_id, orig_month = loan[0], loan[1]
        defaulted = loan[14] == 1
        for mob, bucket, dpd in [
            (0, "Current", 0), (1, "30", 30), (2, "Current", 0),
            (12, "90+" if defaulted else "Current", 95 if defaulted else 0),
        ]:
            snapshots.append([loan_id, orig_month, mob, bucket, dpd, 10_000])
    for name, columns, rows in [
        ("loans.csv", LOAN_COLUMNS, loans),
        ("snapshots.csv", SNAPSHOT_COLUMNS, snapshots),
    ]:
        with open(data_dir / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
    return data_dir


def test_all_sql_files_execute_and_return_rows(tmp_path: Path) -> None:
    data_dir = _fixture_data(tmp_path / "data")
    result = validate_sql(data_dir, SQL_DIR)
    assert sorted(result) == [
        "01_vintage_analysis.sql",
        "02_roll_rates.sql",
        "03_portfolio_kpis.sql",
    ]
    # Row counts per statement, for the 8-loan / 2-vintage fixture:
    # 01: one row per vintage at MOB 12; 02: one matrix row per bucket
    # observed as "from" (Current, 30); 03: six KPI queries.
    assert result["01_vintage_analysis.sql"] == [2]
    assert result["02_roll_rates.sql"] == [2]
    assert result["03_portfolio_kpis.sql"] == [2, 2, 4, 2, 1, 2]
    assert all(count > 0 for counts in result.values() for count in counts)


def test_roll_rate_matrix_rows_sum_to_100(tmp_path: Path) -> None:
    data_dir = _fixture_data(tmp_path / "data")
    conn = build_database(data_dir)
    try:
        cursor = conn.execute(
            "SELECT bucket_from, "
            "ROUND(100.0 * SUM(CASE WHEN bucket_to = 'Current' THEN 1 ELSE 0 END) "
            "/ COUNT(*), 2) AS to_current_pct, "
            "ROUND(100.0 * SUM(CASE WHEN bucket_to = '30' THEN 1 ELSE 0 END) "
            "/ COUNT(*), 2) AS to_30_pct "
            "FROM (SELECT a.dpd_bucket AS bucket_from, b.dpd_bucket AS bucket_to "
            "FROM snapshots AS a JOIN snapshots AS b "
            "ON b.loan_id = a.loan_id AND b.months_on_book = a.months_on_book + 1) "
            "GROUP BY bucket_from"
        )
        for _from, to_current, to_30 in cursor.fetchall():
            assert to_current + to_30 == pytest.approx(100.0)
    finally:
        conn.close()


def test_missing_data_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_sql(tmp_path / "empty", SQL_DIR)


def test_invalid_sql_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sql"
    bad.write_text("SELECT * FROM nonexistent_table;", encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.Error):
            run_sql_file(conn, bad)
    finally:
        conn.close()
