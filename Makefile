# CreditRisk-IFRS9
VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: setup test run notebook sql-check clean

setup:  ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

test:   ## Run pytest
	$(PY) -m pytest tests/ -q

run:    ## Run the full pipeline on synthetic data
	$(PY) -m src.run_pipeline

notebook:  ## Execute the EDA notebook in place
	$(VENV)/bin/jupyter nbconvert --to notebook --execute \
		--inplace notebooks/01_eda_vintage.ipynb

sql-check:  ## Validate the SQL files against the synthetic data (SQLite)
	$(PY) scripts/check_sql.py

clean:  ## Remove regenerable data/outputs (kept out of git)
	rm -rf data outputs __pycache__ src/__pycache__ tests/__pycache__
