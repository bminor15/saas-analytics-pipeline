-- mart views on top of the warehouse facts + dims

-- MRR per plan / tier / region by month
-- uses a month spine so each active sub contributes to every month it covers
CREATE OR REPLACE VIEW mart_mrr_by_month AS
WITH month_spine AS (
    SELECT DISTINCT DATE_TRUNC('month', date_day) AS month_start
    FROM dim_date
),
sub_monthly AS (
    SELECT
        ms.month_start,
        fs.subscription_id,
        fs.plan,
        fs.plan_rank,
        fs.account_tier,
        fs.account_region,
        fs.list_price_usd                                           AS mrr_usd
    FROM month_spine ms
    JOIN fact_subscriptions fs
        ON  ms.month_start >= DATE_TRUNC('month', fs.start_at::DATE)
        AND ms.month_start <= DATE_TRUNC('month', COALESCE(fs.end_at, CURRENT_TIMESTAMP)::DATE)
        AND fs.list_price_usd > 0
)
SELECT
    month_start,
    strftime(month_start, '%Y-%m')      AS year_month,
    plan,
    plan_rank,
    account_tier,
    account_region,
    COUNT(subscription_id)              AS active_subscriptions,
    SUM(mrr_usd)                        AS mrr_usd
FROM sub_monthly
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY month_start, plan_rank;


-- churned subscriptions per month - join to mart_mrr_by_month to get churn rate
CREATE OR REPLACE VIEW mart_churn_by_month AS
SELECT
    DATE_TRUNC('month', end_at::DATE)               AS churn_month,
    strftime(DATE_TRUNC('month', end_at::DATE), '%Y-%m') AS year_month,
    plan,
    account_tier,
    account_region,
    COUNT(*)                                        AS churned_subscriptions
FROM fact_subscriptions
WHERE is_churned
  AND end_at IS NOT NULL
GROUP BY 1, 2, 3, 4, 5
ORDER BY churn_month;


-- actual collected revenue from payments (not MRR estimates)
-- includes failed/refunded counts for payment health monitoring
CREATE OR REPLACE VIEW mart_revenue_by_month AS
SELECT
    d.year_month,
    fp.plan,
    fp.account_tier,
    fp.account_region,
    COUNT(DISTINCT fp.subscription_id)              AS paying_subscriptions,
    COUNT(fp.payment_id)                            AS payment_count,
    SUM(fp.revenue_usd)                             AS total_revenue_usd,
    SUM(fp.is_failed::INTEGER)                      AS failed_payments,
    SUM(fp.is_refunded::INTEGER)                    AS refunded_payments,
    ROUND(AVG(CASE WHEN fp.is_paid THEN fp.amount END), 2) AS avg_payment_usd
FROM fact_payments fp
JOIN dim_date d ON fp.paid_date_key = d.date_key
GROUP BY 1, 2, 3, 4
ORDER BY d.year_month, fp.plan;


-- DAU + active accounts per day, broken out by tier and region
-- roll up to WAU/MAU by grouping on year_month
CREATE OR REPLACE VIEW mart_daily_active_users AS
SELECT
    d.date_day,
    d.year_month,
    d.is_weekend,
    fe.account_tier,
    fe.account_region,
    COUNT(DISTINCT fe.user_id)                      AS dau,
    COUNT(DISTINCT fe.account_id)                   AS active_accounts,
    COUNT(*)                                        AS total_events,
    SUM(fe.is_login::INTEGER)                       AS logins,
    SUM(fe.is_error::INTEGER)                       AS errors,
    SUM(fe.is_api_call::INTEGER)                    AS api_calls
FROM fact_events fe
JOIN dim_date d ON fe.occurred_date_key = d.date_key
GROUP BY 1, 2, 3, 4, 5
ORDER BY d.date_day;


-- event volume by type, device, and OS per month
CREATE OR REPLACE VIEW mart_feature_usage AS
SELECT
    d.year_month,
    fe.event_type,
    fe.device,
    fe.os,
    fe.account_tier,
    COUNT(*)                                        AS event_count,
    COUNT(DISTINCT fe.user_id)                      AS unique_users,
    COUNT(DISTINCT fe.account_id)                   AS unique_accounts
FROM fact_events fe
JOIN dim_date d ON fe.occurred_date_key = d.date_key
GROUP BY 1, 2, 3, 4, 5
ORDER BY d.year_month, event_count DESC;
