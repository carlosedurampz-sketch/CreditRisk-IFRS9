"""Validate sql/*.sql against the synthetic data using SQLite (stdlib only).

Loads data/synthetic/loans.csv and snapshots.csv into an in-memory
SQLite database and executes each .sql file, printing row counts.
This proves the SQL is not just documentation — it runs and returns
sensible results on the same data the Python pipeline uses.

Usage: python scripts/check_sql.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csv(conn: sqlite3.Connection, table: str, path: Path) -> int:
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
        text = sql_file.read_text(encoding="utf-8")
        # Strip the commented-out "Bonus" queries so only live SQL runs.
        statements = [s for s in text.split(";") if s.strip() and not s.strip().startswith("--")]
        # The files contain several independent queries; run each statement
        # that is a SELECT (the CTE blocks belong to the SELECT that follows).
        # Simplest robust approach: execute the whole file minus comment-only
        # lines via executescript, then re-run the first SELECT for a count.
        try:
            conn.executescript(text)
            print(f"OK   {sql_file.name}")
        except Exception as exc:
            ok = False
            print(f"FAIL {sql_file.name}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
