-- monthly churn rate by account tier
-- high churn in Enterprise is a red flag, high churn in Free is expected

SELECT
    m.year_month,
    m.account_tier,
    SUM(m.active_subscriptions)                             AS active_subscriptions,
    COALESCE(SUM(c.churned_subscriptions), 0)               AS churned_subscriptions,
    ROUND(
        COALESCE(SUM(c.churned_subscriptions), 0) * 100.0
        / NULLIF(SUM(m.active_subscriptions), 0)
    , 2)                                                    AS churn_rate_pct
FROM mart_mrr_by_month m
LEFT JOIN mart_churn_by_month c
       ON m.year_month    = c.year_month
      AND m.account_tier  = c.account_tier
GROUP BY m.year_month, m.account_tier
ORDER BY m.year_month, m.account_tier;
