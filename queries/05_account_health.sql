-- top 25 accounts by lifetime revenue with subscription + engagement status
-- accounts with high revenue but a large days_since_active are paying but not using the product
-- those are your highest churn risks

WITH account_revenue AS (
    SELECT
        fp.account_id,
        SUM(fp.revenue_usd)                                 AS lifetime_revenue_usd,
        COUNT(DISTINCT fp.subscription_id)                  AS subscriptions,
        SUM(fp.is_failed::INTEGER)                          AS failed_payments
    FROM fact_payments fp
    GROUP BY fp.account_id
),
account_activity AS (
    SELECT
        account_id,
        COUNT(*)                                            AS total_events,
        COUNT(DISTINCT occurred_at::DATE)                   AS active_days,
        MAX(occurred_at)                                    AS last_seen_at
    FROM fact_events
    GROUP BY account_id
),
account_subscription AS (
    SELECT
        account_id,
        MAX(plan)                                           AS current_plan,
        MAX(status)                                         AS subscription_status,
        SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_subscriptions
    FROM fact_subscriptions
    GROUP BY account_id
)
SELECT
    a.account_id,
    a.account_name,
    a.tier,
    a.industry,
    a.region,
    s.current_plan,
    s.subscription_status,
    s.active_subscriptions,
    ROUND(r.lifetime_revenue_usd, 2)                        AS lifetime_revenue_usd,
    r.failed_payments,
    e.total_events,
    e.active_days,
    e.last_seen_at::DATE                                    AS last_seen_date,
    DATEDIFF('day', e.last_seen_at, CURRENT_TIMESTAMP)      AS days_since_active
FROM dim_account a
JOIN account_revenue     r ON a.account_id = r.account_id
JOIN account_activity    e ON a.account_id = e.account_id
JOIN account_subscription s ON a.account_id = s.account_id
ORDER BY lifetime_revenue_usd DESC
LIMIT 25;
