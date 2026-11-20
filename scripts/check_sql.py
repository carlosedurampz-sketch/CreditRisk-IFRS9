"""Validate sql/*.sql against the synthetic data using SQLite (stdlib only).

Loads data/synthetic/loans.csv and snapshots.csv into an in-memory
SQLite database and executes each .sql file, printing row counts.
This proves the SQL is not just documentation — it runs and returns
sensible results on the same data the Python pipeline uses.

The validation logic is importable so pytest can exercise it
(tests/test_sql.py) without invoking the CLI: use
:func:`validate_sql` for a whole directory or :func:`run_sql_file`
for a single file against an existing connection.

Usage: python scripts/check_sql.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csv(conn: sqlite3.Connection, table: str, path: Path) -> int:
    """Load a CSV file into a SQLite table; return the number of rows loaded."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0
    cols = list(rows[0].keys())
    conn.execute(f"CREATE TABLE {table} ({', '.join(f'\"{c}\" TEXT' for c in cols)})")
    conn.executemany(
        f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
        [[r[c] for c in cols] for r in rows],
    )
    return len(rows)


def build_database(data_dir: Path) -> sqlite3.Connection:
    """Create an in-memory SQLite database from loans.csv and snapshots.csv."""
    conn = sqlite3.connect(":memory:")
    for table, name in [("loans", "loans.csv"), ("snapshots", "snapshots.csv")]:
        load_csv(conn, table, data_dir / name)
    return conn


def _statements(text: str) -> list[str]:
    """Split a SQL file into executable statements, dropping comment lines."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def run_sql_file(conn: sqlite3.Connection, sql_file: Path) -> list[int]:
    """Execute every statement in a .sql file; return row counts per statement.

    Raises sqlite3.Error if any statement fails.
    """
    counts: list[int] = []
    for statement in _statements(sql_file.read_text(encoding="utf-8")):
        cursor = conn.execute(statement)
        counts.append(len(cursor.fetchall()))
    return counts


def validate_sql(data_dir: Path, sql_dir: Path) -> dict[str, list[int]]:
    """Validate every .sql file in sql_dir against the data in data_dir.

    Returns {filename: [row counts per statement]}. Raises FileNotFoundError
    if the data files are missing and sqlite3.Error on invalid SQL.
    """
    if not (data_dir / "loans.csv").exists() or not (data_dir / "snapshots.csv").exists():
        raise FileNotFoundError(f"synthetic data not found in {data_dir}")
    conn = build_database(data_dir)
    try:
        return {
            sql_file.name: run_sql_file(conn, sql_file)
            for sql_file in sorted(sql_dir.glob("*.sql"))
        }
    finally:
        conn.close()


def main() -> int:
    data_dir = PROJECT_ROOT / "data" / "synthetic"
    if not (data_dir / "loans.csv").exists():
        print("No synthetic data yet — run: python -m src.run_pipeline")
        return 1
    conn = sqlite3.connect(":memory:")
    for table, name in [("loans", "loans.csv"), ("snapshots", "snapshots.csv")]:
        n = load_csv(conn, table, data_dir / name)
        print(f"loaded {table}: {n:,} rows")
    ok = True
    for sql_file in sorted((PROJECT_ROOT / "sql").glob("*.sql")):
        try:
            counts = run_sql_file(conn, sql_file)
            print(f"OK   {sql_file.name} ({len(counts)} statements, "
                  f"{sum(counts):,} rows returned)")
        except Exception as exc:  # noqa: BLE001 - CLI reports any SQL failure
            ok = False
            print(f"FAIL {sql_file.name}: {exc}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
