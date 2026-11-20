# CreditRisk-IFRS9 — Portfolio Management, Vintage Analysis & Expected Loss

> **Resumen en español.** Sistema integral de análisis y modelado de riesgo de crédito:
> auditoría de mora en SQL (curvas de cosechas y matrices de transición), scorecard
> de probabilidad de incumplimiento (PD) con Weight of Evidence, pérdida esperada
> bajo lógica IFRS 9 / Basilea, y monitoreo de estabilidad poblacional (PSI).
> Corre end-to-end con datos sintéticos realistas — sin descargar nada — y documenta
> cómo reproducirlo con datos reales (LendingClub / Home Credit).
> **Todas las métricas de este README provienen de datos sintéticos y son
> ilustrativas**: muestran que el pipeline funciona y cómo se interpretan los
> resultados, no el desempeño de ninguna cartera real.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## About the author / Sobre el autor

I am **Carlos Eduardo Ramírez Pérez**, a physicist in the process of obtaining my
degree at UAM (Universidad Autónoma Metropolitana, 2022–2026), with a prior
technical degree in Electronics (CBTis No. 003, Tlaxcala). I am building this
portfolio to move into credit-risk / data roles in banking and fintech in Mexico.
**I have not worked in a bank** — this project is applied learning, and I say so
plainly, because in risk modelling credibility starts with stating what you do
and do not know.

What I do bring, and what this project is built on:

- **Mathematical and statistical modelling from physics.** My degree trained me
  in stochastic processes, numerical methods, and fitting models to noisy data
  and then *validating* them — the same discipline a PD model needs (calibration,
  out-of-time testing, stability monitoring). I program in Python (advanced),
  R (advanced), C/C++ (advanced) and Fortran 90 (intermediate).
- **Calibration-grade rigour from metrology.** In 2026 I worked at **INCYMED**,
  calibrating pressure, temperature and electrical instruments in a laboratory
  accredited under **ISO/IEC 17025 (PJLA)**, authorised to execute and review
  calibrations. That work is traceability, reproducibility and auditable
  documentation — exactly what model validation in a regulated bank demands:
  every number in this repo can be traced to the code and seed that produced it.
- **Teaching.** As State Delegate for the Mexican Physics Olympiad (CMF,
  2021–2022) I explained quantitative ideas to students — a skill I use here in
  the heavily commented SQL and in the README itself.
- **SQL is the skill I am deliberately strengthening with this project**
  (my starting level was basic). The queries in `sql/` are written as a learning
  artefact: every CTE and window function is commented in plain language, and
  `scripts/check_sql.py` proves they run and return sensible results on the
  synthetic data. I am not presenting myself as a SQL expert — I am showing
  the work of becoming one.
- **Languages & courses.** English B2; Data Science with R (UNAM), Python I
  (PUC Chile), a 4-course Big Data series, Introduction to Statistics
  (Stanford Online / Coursera).

---

## What this project does

```
synthetic loans ──► SQL audit ──► WoE scorecard ──► Expected loss ──► Monitoring
 (20k loans)      (vintage &       (logistic vs        EL = PD×LGD×EAD   (PSI, monthly)
                   roll rates)      LightGBM)          + macro stress
```

| Stage | Question it answers | Artefact |
|---|---|---|
| Portfolio audit (SQL) | Is delinquency a bad vintage, or a bad month? How fast do loans roll between DPD buckets? | `sql/01–03` |
| PD scorecard | Which borrowers default within 12 months, and is the probability calibrated? | `src/woe.py`, `src/train_pd.py` |
| Expected loss (IFRS 9) | How much should be provisioned, by grade, and under stress? | `src/expected_loss.py` |
| Monitoring | Is the scored population drifting away from the training population? | `src/monitoring.py` |

## Architecture

```
CreditRisk-IFRS9/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── Makefile / run_all.sh          # one-command end-to-end run
├── sql/
│   ├── 01_vintage_analysis.sql    # vintage (cosecha) curves by MOB
│   ├── 02_roll_rates.sql          # DPD transition matrices
│   └── 03_portfolio_kpis.sql      # monthly management-pack KPIs
├── src/
│   ├── generate_synthetic.py      # realistic synthetic DGP (documented)
│   ├── woe.py                     # WoE / IV binning
│   ├── train_pd.py                # calibrated logistic vs LightGBM, OOT split
│   ├── expected_loss.py           # EL = PD × LGD × EAD + stress scenarios
│   ├── monitoring.py              # PSI by month
│   ├── utils.py                   # KS, Gini, decile table
│   └── run_pipeline.py            # orchestrates 1 → 4
├── notebooks/
│   └── 01_eda_vintage.ipynb       # executed notebook (outputs included)
├── scripts/
│   └── check_sql.py               # validates sql/*.sql on SQLite
├── tests/
│   ├── test_woe.py                # WoE / IV tests
│   └── test_expected_loss.py      # EL formula, stress, PSI tests
├── data/synthetic/                # generated (git-ignored, regenerable)
└── outputs/                       # metrics JSON, deciles CSV, PNG charts
```

## Quick start

```bash
git clone <this-repo> && cd CreditRisk-IFRS9
bash run_all.sh            # creates .venv, installs deps, runs tests + pipeline
# — or step by step —
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m src.run_pipeline
```

Expected runtime: ~1–2 minutes (20,000 loans / 500,000 monthly snapshots).
Outputs land in `outputs/`: `pd_metrics.json`, `deciles_logistic.csv`,
`expected_loss.json`, `el_by_grade.csv`, `el_stress.csv`, `psi_monthly.csv`,
and `roc_curve.png`, `decile_calibration.png`, `psi_monthly.png`.

To validate the SQL separately (SQLite, standard library only):

```bash
.venv/bin/python scripts/check_sql.py
# loaded loans: 20,000 rows / loaded snapshots: 500,000 rows
# OK 01_vintage_analysis.sql / OK 02_roll_rates.sql / OK 03_portfolio_kpis.sql
```

## Dataset

**Default (no download needed).** The pipeline generates synthetic data with a
documented data-generating process in `src/generate_synthetic.py`: 20,000 loans
originated monthly 2022-01 → 2025-06, with grade assigned from the bureau score,
interest rate by grade, and a latent log-odds of 12-month default driven by DTI,
revolving utilisation, bureau score, delinquencies, employment length and rate.
Monthly delinquency snapshots (MOB 0–24) follow a Markov-style process in which
90+ DPD is near-absorbing. Overall 12-month default rate: **10.08%** (synthetic).

**With real data.** The suggested real datasets are
[LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(accepted loans, `loan_status` → default) and
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(`application_train.csv` → `TARGET`). To use one: export it to
`data/synthetic/loans.csv` with columns `loan_id, origination_month, grade,
term_months, loan_amnt, int_rate, annual_income, dti, revol_util, bureau_score,
emp_length_years, delinq_2yrs, purpose, default_12m, lgd_realised` (and monthly
snapshots to `snapshots.csv` with `loan_id, origination_month, months_on_book,
dpd_bucket, dpd_days, outstanding_balance`), then run with
`python -m src.run_pipeline --skip-generate`. The WoE binning, split and metrics
are dataset-agnostic; only the column mapping needs adapting, and the LGD
estimation in `expected_loss.py` should be replaced by a workout-LGD model on
real recoveries (see Limitations).

## Methodology

### 1. Portfolio audit in SQL

- **Vintage analysis** (`sql/01`): for each origination month and months-on-book,
  the share in 30+ and 90+ DPD and the *cumulative* 90+ rate, built with
  `SUM() OVER (PARTITION BY origination_month ORDER BY months_on_book)`.
  A vintage sitting above its neighbours at the same MOB flags underwriting;
  a simultaneous rise across vintages flags macro/collections.
- **Roll rates** (`sql/02`): empirical transition matrix between DPD buckets,
  from a self-join of month `t` to month `t+1`. Rows sum to 100%.
- **KPIs** (`sql/03`): portfolio size, 12-month default rate by vintage and by
  grade, delinquency stock, and concentration in grades D–G.

### 2. PD scorecard — WoE logistic regression vs LightGBM

For a variable $x$ binned into bins $b$, Weight of Evidence and Information Value are

$$\mathrm{WoE}_b = \ln\left(\frac{\%\,\text{good in } b}{\%\,\text{bad in } b}\right),
\qquad \mathrm{IV} = \sum_b (\text{good\%}_b - \text{bad\%}_b)\cdot \mathrm{WoE}_b$$

The **champion is the calibrated logistic regression on WoE features**:
monotone, interpretable, and auditable line by line — the standard a regulator
can review. It is wrapped in `CalibratedClassifierCV` (sigmoid/Platt scaling),
because a PD fed into expected loss must be a *probability*, not just a ranking.
**LightGBM** is the challenger on raw features.

Crucially, the split is **out-of-time**, not random: train on loans originated
≤ 2024-06 (14,373 loans, 10.36% default), test on loans originated after
(5,627 loans, 9.37% default). A random split would leak the future into
training and flatter every metric.

Discrimination is measured with the Kolmogorov–Smirnov statistic

$$\mathrm{KS} = \max_s \left| F_{\text{bad}}(s) - F_{\text{good}}(s) \right|$$

and Gini $= 2\cdot\mathrm{ROC\text{-}AUC} - 1$, plus PR-AUC (more informative
than ROC-AUC under class imbalance) and the Brier score for calibration.

### 3. Expected loss (IFRS 9 / Basel)

For each exposure:

$$EL = PD \times LGD \times EAD$$

- **PD**: 12-month PD from the calibrated scorecard.
- **LGD**: mean realised LGD by grade on defaulted loans in train
  (A ≈ 35% → G ≈ 65% in the synthetic data), floored at 5% / capped at 95%.
- **EAD**: outstanding balance at month 12 from the snapshots
  (origination amount when no snapshot exists).

Stress scenarios scale PD and shift LGD, as in a bank stress test:
**base** (PD ×1.0), **adverse** (PD ×1.5, LGD +10 pp), **severe**
(PD ×2.0, LGD +20 pp). Staging (Stage 1/2/3) is *not* implemented — see
Limitations; the 12-month PD is used for all exposures and the simplification
is stated, not hidden.

### 4. Monitoring — Population Stability Index

$$\mathrm{PSI} = \sum_b (\text{actual\%}_b - \text{expected\%}_b)\cdot
\ln\left(\frac{\text{actual\%}_b}{\text{expected\%}_b}\right)$$

Decile edges are fixed on the reference period (2024-07); each later month is
compared against it. Thresholds: PSI < 0.10 stable, 0.10–0.25 investigate,
> 0.25 significant shift / consider recalibration.

## Results — synthetic data, illustrative ⚠️

> **Read this first.** Every number below was produced by
> `python -m src.run_pipeline` (seed 42, 20,000 synthetic loans) and is
> reproducible from `outputs/`. It demonstrates that the pipeline works and
> how to read the outputs. **It is not a real portfolio's performance** and
> must not be quoted as such.

### Portfolio audit (SQL)

Default rate by grade is monotone, as a sane grading must be (full table in the
notebook; A 5.53% → G 15.41% 12-month defaults). Vintage 2022-01 reaches
1.02% in 90+ DPD at MOB 3, 4.29% at MOB 6 and 12.27% at MOB 12 — curves rise
and then flatten, which justifies the 12-month default window.

**Roll-rate matrix** (% of loans, rows = bucket this month → columns = next month):

| From \ To | Current | 30 | 60 | 90+ |
|---|---:|---:|---:|---:|
| Current | **91.9** | 7.7 | 0.0 | 0.4 |
| 30 | 44.4 | 46.6 | 8.5 | 0.5 |
| 60 | 0.0 | 43.8 | 45.5 | 10.7 |
| 90+ | 0.0 | 0.0 | 2.4 | **97.6** |

Reading: 30 DPD cures 44.4% of the time and rolls to 60 only 8.5%; 90+ is
near-absorbing (97.6% stay). (Cures move one bucket at a time in the generator,
so 60 → Current is 0 by construction — a documented property of the synthetic
data, not a finding about real borrowers.)

### PD model — out-of-time test set (n = 5,627)

| Model | ROC-AUC | PR-AUC | KS | Gini | Brier |
|---|---:|---:|---:|---:|---:|
| **Logistic (WoE, calibrated) — champion** | **0.656** | **0.165** | **0.255** | **0.312** | **0.083** |
| LightGBM (challenger) | 0.633 | 0.152 | 0.192 | 0.266 | 0.084 |

The auditable model wins on this synthetic data — the honest result, and a
useful one: on noisy data with mostly monotone drivers, a regularised WoE
logistic beats the gradient-boosting challenger on ranking *and* calibration.
I report the challenger's loss rather than tuning it until it wins.

![ROC curve](outputs/roc_curve.png)

**Information Value** (top drivers): DTI 0.224 (medium), grade 0.107 (medium),
bureau score 0.101 (medium), interest rate 0.097 (weak). Full table:
`outputs/iv_table.csv`.

**Decile table** (logistic, test set; 1 = lowest PD):

| Decile | n | Defaults | Default rate | Mean PD | Lift |
|---:|---:|---:|---:|---:|---:|
| 1 | 563 | 22 | 3.91% | 3.59% | 0.42 |
| 2 | 563 | 36 | 6.39% | 4.94% | 0.68 |
| 3 | 562 | 27 | 4.80% | 6.01% | 0.51 |
| 4 | 563 | 39 | 6.93% | 7.09% | 0.74 |
| 5 | 563 | 29 | 5.15% | 8.22% | 0.55 |
| 6 | 562 | 46 | 8.19% | 9.51% | 0.87 |
| 7 | 563 | 58 | 10.30% | 11.12% | 1.10 |
| 8 | 562 | 59 | 10.50% | 13.12% | 1.12 |
| 9 | 563 | 104 | 18.47% | 16.04% | 1.97 |
| 10 | 563 | 107 | 19.01% | 21.60% | 2.03 |

The ranking is broadly monotone with small-sample inversions (deciles 2–3
and 4–5) — with ~30–60 defaults per decile, a ±2 pp wobble is sampling noise,
and I leave it visible rather than smoothing it away. The top decile captures
20.3% of all defaults; the top three deciles capture 51.2%.

![Decile calibration](outputs/decile_calibration.png)

### Expected loss and stress (test portfolio)

| Scenario | PD multiplier | LGD shift | Total EL | EL rate |
|---|---:|---:|---:|---:|
| Base | ×1.0 | +0 pp | **$4,120,897** | **576 bps** |
| Adverse | ×1.5 | +10 pp | $7,387,598 | 1,033 bps |
| Severe | ×2.0 | +20 pp | $11,458,468 | 1,603 bps |

Base case: EAD $71,499,527, mean PD 10.12%, mean LGD 48.6%.
EL rate by grade rises monotonically from 226 bps (A) to 1,077 bps (G)
(`outputs/el_by_grade.csv`). The stress table's point is sensitivity —
EL roughly triples from base to severe — not a macro forecast.

### Monitoring (PSI, monthly, reference 2024-07)

Max monthly PSI **0.059**, mean 0.024, **0 months flagged** (all < 0.10):
the scored population is stable over the test window, as expected for
synthetic data with no drift injected. The monitoring code is the deliverable
here, not the (uneventful) reading.

![Monthly PSI](outputs/psi_monthly.png)

## Limitations — stated, not buried

1. **Synthetic data.** All results are illustrative. The DGP encodes the
   qualitative relationships I expect (higher DTI → higher risk) but no real
   portfolio behaves exactly like this; in particular, the generator's
   one-bucket-at-a-time cures and near-absorbing 90+ are simplifications.
2. **No IFRS 9 staging.** There is no Stage 1/2/3 classification, no
   significant-increase-in-credit-risk (SICR) rule, and no lifetime PD —
   the EL uses a 12-month PD for every exposure. That is the single biggest
   gap versus a production IFRS 9 engine.
3. **LGD is a grade mean, not a model.** Realised LGD is averaged by grade;
   a production LGD model would use discounted workout cash flows,
   collateral, and time-to-recovery.
4. **No macro conditioning.** The stress scenarios are multipliers, not a
   model linking PD to GDP/unemployment. The "adverse/severe" labels are
   scenario names, not forecasts.
5. **Modest discrimination.** KS 0.255 / Gini 0.312 on synthetic data is a
   realistic, mid-range scorecard — not a tuned-to-impress number. With real
   data, bureau and behavioural variables would normally add more signal.
6. **SQL is a learning artefact** (see "About the author"): the queries are
   correct and validated, but written by someone strengthening SQL, not by a
   senior SQL developer. Reviewers are invited to read the comments and judge
   the reasoning.

## Next steps

- [ ] Implement IFRS 9 staging: SICR rule from roll rates + lifetime PD curves
  (survival / hazard model) for Stage 2.
- [ ] Fit a workout-LGD model (beta regression on recoveries) instead of grade means.
- [ ] Re-run the pipeline on LendingClub data (`--skip-generate`) and add a
  comparison section: synthetic vs real metrics, side by side.
- [ ] Add a macro-conditioned PD overlay (logit shift by unemployment scenario).
- [ ] Backtest the EL: compare predicted EL against realised 12-month losses
  by vintage, with a Kupiec-style coverage check.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q   # 7 passed
```

`test_woe.py` checks that IV detects a real signal, that WoE is monotone in
that signal, and that unseen categories map to a neutral WoE of 0.
`test_expected_loss.py` checks `EL = PD × LGD × EAD` on hand-computed cases,
that stress scenarios increase EL monotonically, and that PSI is 0 for
identical distributions and > 0.25 for a large shift.

## License

MIT — see [LICENSE](LICENSE).
