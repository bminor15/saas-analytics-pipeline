-- revenue by plan and account tier, with payment health metrics
-- failure_rate_pct above ~8% is worth looking into — could be billing issues
-- or customers doing passive churn (letting payments fail instead of canceling)

SELECT
    plan,
    account_tier,
    SUM(total_revenue_usd)                                  AS total_revenue_usd,
    SUM(payment_count)                                      AS payment_count,
    SUM(failed_payments)                                    AS failed_payments,
    SUM(refunded_payments)                                  AS refunded_payments,
    ROUND(
        SUM(failed_payments) * 100.0
        / NULLIF(SUM(payment_count), 0)
    , 2)                                                    AS failure_rate_pct,
    ROUND(AVG(avg_payment_usd), 2)                          AS avg_payment_usd
FROM mart_revenue_by_month
GROUP BY plan, account_tier
ORDER BY total_revenue_usd DESC;
