-- ============================================================================
-- 02_roll_rates.sql — Roll-rate (transition) matrices
-- ============================================================================
-- WHAT THIS DOES
--   A roll rate is the probability that a loan in bucket X this month
--   is in bucket Y next month. Buckets:
--     Current -> 30 -> 60 -> 90+   (days past due)
--   The matrix answers questions a static default rate cannot:
--     * How fast do 30-DPD loans cure vs roll to 60?
--     * Is 90+ really absorbing, or are loans curing from it?
--     * Is deterioration accelerating in recent months?
--   Under IFRS 9, roll rates are also a building block for staging
--   (significant increase in credit risk) and for lifetime-PD curves.
--
-- METHOD
--   1. Self-join snapshots t and t+1 on (loan_id, months_on_book + 1)
--      — each row becomes one observed transition.
--   2. Count transitions from -> to, divide by all transitions *from*
--      the same bucket. That is the empirical transition probability.
--   3. A second query repeats the matrix by calendar half-year, so a
--      drift in roll behaviour over time is visible.
--
-- READING THE MATRIX: rows = bucket this month, columns = bucket next
--   month, cells sum to 100% across each row. Diagonal = "stayed",
--   upper triangle = deteriorated, lower triangle = cured/improved.
-- ============================================================================

-- --- Step 1: build one row per observed month-to-month transition -------------
-- Self-join (t, t+1) makes the pairing explicit and auditable row by row.
WITH transitions AS (
    SELECT
        a.dpd_bucket AS bucket_from,
        b.dpd_bucket AS bucket_to
    FROM snapshots AS a
    JOIN snapshots AS b
        ON b.loan_id = a.loan_id
        AND b.months_on_book = a.months_on_book + 1
),

-- --- Step 2: count every (from, to) pair --------------------------------------
counts AS (
    SELECT bucket_from, bucket_to, COUNT(*) AS n
    FROM transitions
    GROUP BY bucket_from, bucket_to
),

-- --- Step 3: row totals, i.e. how many loans started in each bucket -----------
totals AS (
    SELECT bucket_from, SUM(n) AS n_from
    FROM counts
    GROUP BY bucket_from
)

-- --- Final: the transition matrix, pivoted ------------------------------------
SELECT
    c.bucket_from,
    t.n_from,
    ROUND(100.0 * SUM(CASE WHEN c.bucket_to = 'Current' THEN c.n ELSE 0 END) / t.n_from, 2) AS to_current_pct,
    ROUND(100.0 * SUM(CASE WHEN c.bucket_to = '30'      THEN c.n ELSE 0 END) / t.n_from, 2) AS to_30_pct,
    ROUND(100.0 * SUM(CASE WHEN c.bucket_to = '60'      THEN c.n ELSE 0 END) / t.n_from, 2) AS to_60_pct,
    ROUND(100.0 * SUM(CASE WHEN c.bucket_to = '90+'     THEN c.n ELSE 0 END) / t.n_from, 2) AS to_90plus_pct
FROM counts AS c
JOIN totals AS t ON t.bucket_from = c.bucket_from
GROUP BY c.bucket_from, t.n_from
ORDER BY
    CASE c.bucket_from WHEN 'Current' THEN 1 WHEN '30' THEN 2 WHEN '60' THEN 3 ELSE 4 END;

-- --- Bonus: roll rates by calendar period (drift check) ------------------------
-- Uncomment to run. It joins the loan's origination month to date the
-- transition, then groups by year-half.
-- WITH transitions AS (
--     SELECT
--         a.dpd_bucket AS bucket_from,
--         b.dpd_bucket AS bucket_to,
--         substr(a.origination_month, 1, 4) AS year
--     FROM snapshots AS a
--     JOIN snapshots AS b
--         ON b.loan_id = a.loan_id
--         AND b.months_on_book = a.months_on_book + 1
-- )
-- SELECT
--     year,
--     bucket_from,
--     COUNT(*) AS n_from,
--     ROUND(100.0 * SUM(CASE WHEN bucket_to IN ('60', '90+') THEN 1 ELSE 0 END) / COUNT(*), 2)
--         AS pct_worsened_to_60plus
-- FROM transitions
-- WHERE bucket_from = '30'
-- GROUP BY year, bucket_from
-- ORDER BY year;
