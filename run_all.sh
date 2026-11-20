#!/usr/bin/env bash
# CreditRisk-IFRS9 — run the full pipeline end-to-end on synthetic data.
# Usage:  bash run_all.sh [--n-loans 20000]
# Creates .venv on first run (if missing) and installs requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

echo "Running tests..."
.venv/bin/python -m pytest tests/ -q

echo "Running pipeline..."
.venv/bin/python -m src.run_pipeline "$@"

echo ""
echo "Done. See outputs/ (pd_metrics.json, deciles_logistic.csv,"
echo "expected_loss.json, el_stress.csv, psi_monthly.csv, *.png)."
