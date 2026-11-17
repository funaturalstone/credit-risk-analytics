-- Lending Club loan analysis in SQL (SQLite)
-- Data: 80,000 finished loans (Fully Paid or Charged Off), table `loans`
-- Build the database first:  python sql/load_data.py data/lending_club_80k.csv
-- Run everything:            sqlite3 -header -column data/lending_club.db < sql/analysis.sql
--
-- "Default" = loan_status 'Charged Off'.
-- Net cash per loan = total_pymnt - loan_amnt
--   (interest earned on a repaid loan, or the principal lost on a defaulted one)


-- Q1. How many loans, and what share of them defaulted?
SELECT
    COUNT(*)                                                         AS loans,
    SUM(loan_status = 'Charged Off')                                 AS defaults,
    ROUND(100.0 * AVG(loan_status = 'Charged Off'), 1)               AS default_rate_pct
FROM loans;


-- Q2. Default rate and average interest rate by grade, riskiest first.
SELECT
    grade,
    COUNT(*)                                                         AS loans,
    ROUND(100.0 * AVG(loan_status = 'Charged Off'), 1)               AS default_rate_pct,
    ROUND(AVG(int_rate), 2)                                          AS avg_int_rate_pct
FROM loans
GROUP BY grade
ORDER BY default_rate_pct DESC;


-- Q3. Do 60-month loans default more than 36-month loans?
SELECT
    term,
    COUNT(*)                                                         AS loans,
    ROUND(100.0 * AVG(loan_status = 'Charged Off'), 1)               AS default_rate_pct
FROM loans
GROUP BY term;


-- Q4. Default rate by debt-to-income band.
SELECT
    CASE
        WHEN dti IS NULL THEN 'missing'
        WHEN dti < 10    THEN '1) under 10'
        WHEN dti < 20    THEN '2) 10-20'
        WHEN dti < 30    THEN '3) 20-30'
        WHEN dti < 40    THEN '4) 30-40'
        ELSE                  '5) 40+'
    END                                                              AS dti_band,
    COUNT(*)                                                         AS loans,
    ROUND(100.0 * AVG(loan_status = 'Charged Off'), 1)               AS default_rate_pct
FROM loans
GROUP BY dti_band
ORDER BY dti_band;


-- Q5. Default rate by FICO band, only bands with at least 500 loans.
SELECT
    (fico_range_low / 20) * 20                                       AS fico_band_from,
    COUNT(*)                                                         AS loans,
    ROUND(100.0 * AVG(loan_status = 'Charged Off'), 1)               AS default_rate_pct
FROM loans
GROUP BY fico_band_from
HAVING COUNT(*) >= 500
ORDER BY fico_band_from;


-- Q6. Top 5 loan purposes by total money lost on defaulted loans.
SELECT
    purpose,
    COUNT(*)                                                         AS defaulted_loans,
    ROUND(SUM(loan_amnt - total_pymnt) / 1e6, 2)                     AS net_loss_musd
FROM loans
WHERE loan_status = 'Charged Off'
GROUP BY purpose
ORDER BY net_loss_musd DESC
LIMIT 5;


-- Q7. Within each grade, which purposes are riskiest?
--     Ranks purposes (with at least 100 loans) by default rate inside each grade
--     and keeps the top 2 per grade.
WITH by_grade_purpose AS (
    SELECT
        grade,
        purpose,
        COUNT(*)                                                     AS loans,
        AVG(loan_status = 'Charged Off')                             AS default_rate
    FROM loans
    GROUP BY grade, purpose
    HAVING COUNT(*) >= 100
),
ranked AS (
    SELECT
        *,
        RANK() OVER (PARTITION BY grade ORDER BY default_rate DESC)  AS risk_rank
    FROM by_grade_purpose
)
SELECT grade, risk_rank, purpose, loans, ROUND(100.0 * default_rate, 1) AS default_rate_pct
FROM ranked
WHERE risk_rank <= 2
ORDER BY grade, risk_rank;


-- Q8. Is lending to each grade profitable?
--     Expected value of approving a loan, measured against not lending (= 0):
--     average net cash per loan = P(repaid) x avg interest earned - P(default) x avg net loss.
--     Ignores funding cost and the time value of money, so it is an upper bound.
WITH grade_stats AS (
    SELECT
        grade,
        COUNT(*)                                                                    AS loans,
        AVG(loan_status = 'Charged Off')                                            AS p_default,
        AVG(CASE WHEN loan_status = 'Fully Paid'  THEN total_pymnt - loan_amnt END) AS avg_interest_earned,
        AVG(CASE WHEN loan_status = 'Charged Off' THEN loan_amnt - total_pymnt END) AS avg_net_loss
    FROM loans
    GROUP BY grade
),
grade_ev AS (
    SELECT
        *,
        (1 - p_default) * avg_interest_earned - p_default * avg_net_loss           AS ev_per_loan
    FROM grade_stats
)
SELECT
    grade,
    loans,
    ROUND(100.0 * p_default, 1)                                      AS default_rate_pct,
    ROUND(avg_interest_earned, 0)                                    AS avg_interest_usd,
    ROUND(avg_net_loss, 0)                                           AS avg_loss_usd,
    ROUND(ev_per_loan, 0)                                            AS ev_per_loan_usd,
    CASE WHEN ev_per_loan > 0 THEN 'lend' ELSE 'do not lend' END     AS decision
FROM grade_ev
ORDER BY grade;
