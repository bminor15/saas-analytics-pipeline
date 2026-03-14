-- sql/30_warehouse_facts.sql
-- Fact tables for the warehouse layer.

-- ---------------------------------------------------------------------------
-- fact_subscriptions
-- Grain: one row per subscription.
-- Metrics: MRR, subscription duration, churn flag, plan metadata.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_subscriptions AS
SELECT
    s.subscription_id,
    s.account_id,
    a.tier                                                      AS account_tier,
    a.industry                                                  AS account_industry,
    a.region                                                    AS account_region,
    s.plan,
    p.plan_rank,
    p.list_price_usd,

    -- Date keys for joining to dim_date
    CAST(strftime(s.start_at::DATE, '%Y%m%d') AS INTEGER)      AS start_date_key,
    CAST(strftime(
        COALESCE(s.end_at, CURRENT_TIMESTAMP)::DATE, '%Y%m%d'
    ) AS INTEGER)                                               AS end_date_key,

    s.start_at,
    s.end_at,
    s.status,

    -- Churn flag
    (s.status = 'canceled')                                     AS is_churned,

    -- Subscription duration in days (open subs measured to today)
    DATEDIFF('day', s.start_at, COALESCE(s.end_at, CURRENT_TIMESTAMP))
                                                                AS duration_days,

    -- Monthly Recurring Revenue proxy: list price / 12 for annual estimate
    -- Free plan contributes 0 MRR
    CASE WHEN s.status = 'active' THEN p.list_price_usd ELSE 0.00 END
                                                                AS mrr_usd

FROM stg_subscriptions s
LEFT JOIN dim_account   a ON s.account_id = a.account_id
LEFT JOIN dim_plan      p ON s.plan       = p.plan_name;


-- ---------------------------------------------------------------------------
-- fact_payments
-- Grain: one row per payment.
-- Metrics: amount, payment outcomes, revenue by plan/account/period.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_payments AS
SELECT
    py.payment_id,
    py.subscription_id,
    s.account_id,
    a.tier                                                      AS account_tier,
    a.industry                                                  AS account_industry,
    a.region                                                    AS account_region,
    s.plan,
    p.plan_rank,

    CAST(strftime(py.paid_at::DATE, '%Y%m%d') AS INTEGER)      AS paid_date_key,
    py.paid_at,
    py.amount,
    py.currency,
    py.status,

    -- Outcome flags for easy aggregation
    (py.status = 'paid')                                        AS is_paid,
    (py.status = 'failed')                                      AS is_failed,
    (py.status = 'refunded')                                    AS is_refunded,

    -- Revenue only on successful payments
    CASE WHEN py.status = 'paid' THEN py.amount ELSE 0.00 END  AS revenue_usd

FROM stg_payments       py
LEFT JOIN stg_subscriptions s  ON py.subscription_id = s.subscription_id
LEFT JOIN dim_account       a  ON s.account_id       = a.account_id
LEFT JOIN dim_plan          p  ON s.plan             = p.plan_name;


-- ---------------------------------------------------------------------------
-- fact_events
-- Grain: one row per product event.
-- Metrics: DAU, WAU, MAU, feature usage, platform breakdown.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_events AS
SELECT
    e.event_id,
    e.user_id,
    e.account_id,
    a.tier                                                      AS account_tier,
    a.industry                                                  AS account_industry,
    a.region                                                    AS account_region,
    u.role                                                      AS user_role,

    e.event_type,
    e.device,
    e.os,

    CAST(strftime(e.occurred_at::DATE, '%Y%m%d') AS INTEGER)   AS occurred_date_key,
    e.occurred_at,

    -- Convenience flags for common event-type filters
    (e.event_type = 'login')                                    AS is_login,
    (e.event_type = 'error')                                    AS is_error,
    (e.event_type = 'api_call')                                 AS is_api_call

FROM stg_events         e
LEFT JOIN dim_account   a ON e.account_id = a.account_id
LEFT JOIN dim_user      u ON e.user_id    = u.user_id;
