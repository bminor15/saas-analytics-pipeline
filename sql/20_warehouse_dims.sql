-- dimension tables

-- one row per calendar day, 2024-01-01 through 2025-12-31
-- date_key is YYYYMMDD integer for fast joins to fact tables
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

-- one row per account
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

-- valid users only (bad emails filtered in staging)
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

-- static plan lookup with price and rank order
CREATE OR REPLACE TABLE dim_plan AS
SELECT *
FROM (VALUES
    ('Free',       1, 0.00),
    ('Starter',    2, 49.00),
    ('Pro',        3, 99.00),
    ('Business',   4, 299.00),
    ('Enterprise', 5, 999.00)
) t(plan_name, plan_rank, list_price_usd);
