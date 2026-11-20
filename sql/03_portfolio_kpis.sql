-- ============================================================================
-- 03_portfolio_kpis.sql — Portfolio KPIs: the monthly management pack
-- ============================================================================
-- WHAT THIS DOES
--   The five numbers a credit-risk committee asks for every month,
--   computed directly from the loan and snapshot tables:
--     1. Portfolio size and growth (loans, EAD proxy = loan_amnt)
--     2. 12-month default rate by origination month and by grade
--     3. Delinquency stock: current share in each DPD bucket
--     4. Concentration: share of EAD in the worst grades (D-G)
--     5. Vintage-at-MOB-12 default rate (the headline vintage KPI)
--
--   Each query is independent — run the one you need. All of them
--   are cross-checked by the Python pipeline (notebooks/01_eda_vintage.ipynb).
-- ============================================================================

-- --- 1. Portfolio size and growth by origination month -------------------------
SELECT
    origination_month,
    COUNT(*) AS n_loans,
    ROUND(SUM(loan_amnt), 2) AS total_disbursed,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amnt,
    ROUND(AVG(int_rate) * 100, 2) AS avg_int_rate_pct
FROM loans
GROUP BY origination_month
ORDER BY origination_month;

-- --- 2. 12-month default rate by origination month ----------------------------
-- default_12m = 1 if the loan defaulted within 12 months of origination.
SELECT
    origination_month,
    COUNT(*) AS n_loans,
    SUM(default_12m) AS n_defaults,
    ROUND(100.0 * AVG(default_12m), 2) AS default_rate_12m_pct
FROM loans
GROUP BY origination_month
ORDER BY origination_month;

-- --- 3. 12-month default rate by grade (risk ranking check) -------------------
-- A sane scorecard / grading must be monotone: worse grade -> higher rate.
-- If this is not monotone, the grading itself is suspect.
SELECT
    grade,
    COUNT(*) AS n_loans,
    SUM(default_12m) AS n_defaults,
    ROUND(100.0 * AVG(default_12m), 2) AS default_rate_12m_pct,
    ROUND(AVG(bureau_score), 0) AS avg_bureau_score,
    ROUND(AVG(dti), 1) AS avg_dti
FROM loans
GROUP BY grade
ORDER BY grade;

-- --- 4. Delinquency stock at the latest snapshot (MOB 12) ---------------------
SELECT
    dpd_bucket,
    COUNT(*) AS n_loans,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
    -- SUM(COUNT(*)) OVER () is a window over the grouped rows: the grand total.
FROM snapshots
WHERE months_on_book = 12
GROUP BY dpd_bucket
ORDER BY CASE dpd_bucket WHEN 'Current' THEN 1 WHEN '30' THEN 2 WHEN '60' THEN 3 ELSE 4 END;

-- --- 5. Concentration: share of disbursed amount in grades D-G ----------------
SELECT
    ROUND(100.0 * SUM(CASE WHEN grade IN ('D', 'E', 'F', 'G') THEN loan_amnt ELSE 0 END)
        / SUM(loan_amnt), 2) AS pct_ead_grades_d_to_g,
    ROUND(100.0 * SUM(CASE WHEN grade IN ('F', 'G') THEN loan_amnt ELSE 0 END)
        / SUM(loan_amnt), 2) AS pct_ead_grades_f_to_g
FROM loans;

-- --- 6. Headline vintage KPI: 90+ rate at MOB 12, by vintage -------------------
SELECT
    origination_month,
    COUNT(*) AS n_loans,
    ROUND(100.0 * SUM(CASE WHEN dpd_bucket = '90+' THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS pct_90plus_at_mob12
FROM snapshots
WHERE months_on_book = 12
GROUP BY origination_month
ORDER BY origination_month;
