-- month-over-month MRR growth
-- LAG() compares each month to the prior month
-- negative pct_change means churn is outpacing new subscriptions

SELECT
    year_month,
    SUM(mrr_usd)                                            AS mrr_usd,
    SUM(active_subscriptions)                               AS active_subscriptions,
    LAG(SUM(mrr_usd)) OVER (ORDER BY MIN(month_start))     AS prior_month_mrr,
    ROUND(
        (SUM(mrr_usd) - LAG(SUM(mrr_usd)) OVER (ORDER BY MIN(month_start)))
        / NULLIF(LAG(SUM(mrr_usd)) OVER (ORDER BY MIN(month_start)), 0) * 100
    , 2)                                                    AS pct_change
FROM mart_mrr_by_month
GROUP BY year_month
ORDER BY year_month;
