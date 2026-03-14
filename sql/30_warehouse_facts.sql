-- fact tables

-- one row per subscription
-- mrr_usd is 0 for canceled subs, list_price_usd for active ones
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

    -- date keys for joining to dim_date
    CAST(strftime(s.start_at::DATE, '%Y%m%d') AS INTEGER)      AS start_date_key,
    CAST(strftime(
        COALESCE(s.end_at, CURRENT_TIMESTAMP)::DATE, '%Y%m%d'
    ) AS INTEGER)                                               AS end_date_key,

    s.start_at,
    s.end_at,
    s.status,

    (s.status = 'canceled')                                     AS is_churned,

    -- open subs measured to today
    DATEDIFF('day', s.start_at, COALESCE(s.end_at, CURRENT_TIMESTAMP))
                                                                AS duration_days,

    -- free plan contributes 0 MRR
    CASE WHEN s.status = 'active' THEN p.list_price_usd ELSE 0.00 END
                                                                AS mrr_usd

FROM stg_subscriptions s
LEFT JOIN dim_account   a ON s.account_id = a.account_id
LEFT JOIN dim_plan      p ON s.plan       = p.plan_name;


-- one row per payment
-- revenue_usd is 0 for failed/refunded rows
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

    -- outcome flags
    (py.status = 'paid')                                        AS is_paid,
    (py.status = 'failed')                                      AS is_failed,
    (py.status = 'refunded')                                    AS is_refunded,

    CASE WHEN py.status = 'paid' THEN py.amount ELSE 0.00 END  AS revenue_usd

FROM stg_payments       py
LEFT JOIN stg_subscriptions s  ON py.subscription_id = s.subscription_id
LEFT JOIN dim_account       a  ON s.account_id       = a.account_id
LEFT JOIN dim_plan          p  ON s.plan             = p.plan_name;


-- one row per product event
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

    -- handy flags for common filters
    (e.event_type = 'login')                                    AS is_login,
    (e.event_type = 'error')                                    AS is_error,
    (e.event_type = 'api_call')                                 AS is_api_call

FROM stg_events         e
LEFT JOIN dim_account   a ON e.account_id = a.account_id
LEFT JOIN dim_user      u ON e.user_id    = u.user_id;
