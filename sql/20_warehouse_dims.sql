-- sql/20_warehouse_dims.sql
-- Dimension tables for the warehouse layer.

-- ---------------------------------------------------------------------------
-- dim_date
-- One row per calendar date covering the full data window plus buffer.
-- Enables time-series slicing on any fact table via a date key (integer YYYYMMDD).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_date AS
WITH dates AS (
    SELECT CAST(UNNEST(range(
        DATE '2024-01-01',
        DATE '2026-01-01',
        INTERVAL '1 day'
    )) AS DATE) AS date_day
)
SELECT
    CAST(strftime(date_day, '%Y%m%d') AS INTEGER)   AS date_key,
    date_day,
    EXTRACT(year  FROM date_day)::INTEGER            AS year,
    EXTRACT(month FROM date_day)::INTEGER            AS month,
    EXTRACT(day   FROM date_day)::INTEGER            AS day,
    EXTRACT(quarter FROM date_day)::INTEGER          AS quarter,
    EXTRACT(dow   FROM date_day)::INTEGER            AS day_of_week,   -- 0=Sun
    strftime(date_day, '%A')                         AS day_name,
    strftime(date_day, '%B')                         AS month_name,
    strftime(date_day, '%Y-%m')                      AS year_month,
    (EXTRACT(dow FROM date_day) IN (0, 6))           AS is_weekend
FROM dates;

-- ---------------------------------------------------------------------------
-- dim_account
-- One row per account. Carries descriptive attributes for slicing facts.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_account AS
SELECT
    account_id,
    account_name,
    industry,
    region,
    tier,
    created_at,
    CAST(strftime(created_at::DATE, '%Y%m%d') AS INTEGER) AS created_date_key
FROM stg_accounts;

-- ---------------------------------------------------------------------------
-- dim_user
-- One row per user. Excludes invalid emails (flagged in staging).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_user AS
SELECT
    user_id,
    account_id,
    email,
    role,
    created_at,
    CAST(strftime(created_at::DATE, '%Y%m%d') AS INTEGER) AS created_date_key
FROM stg_users
WHERE is_valid_email;

-- ---------------------------------------------------------------------------
-- dim_plan
-- Static lookup for subscription plans and their relative tier ordering.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_plan AS
SELECT *
FROM (VALUES
    ('Free',       1, 0.00),
    ('Starter',    2, 49.00),
    ('Pro',        3, 99.00),
    ('Business',   4, 299.00),
    ('Enterprise', 5, 999.00)
) t(plan_name, plan_rank, list_price_usd);
