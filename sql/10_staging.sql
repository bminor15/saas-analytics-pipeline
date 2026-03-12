-- sql/10_staging.sql
-- Staging layer: cast types, flag/drop known dirty rows, fill controlled nulls.
-- Each table maps 1-to-1 with a raw source.

-- ---------------------------------------------------------------------------
-- stg_accounts
-- Dirty data: ~3% null industry -> coalesce to 'Unknown'
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE stg_accounts AS
SELECT
    account_id,
    account_name,
    COALESCE(industry, 'Unknown')   AS industry,
    region,
    tier,
    TRY_CAST(created_at AS TIMESTAMPTZ) AS created_at
FROM raw_accounts
WHERE account_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- stg_users
-- Dirty data: ~1% literal 'bad_email' strings with no @ sign
-- Keep all rows but flag validity; downstream can filter on is_valid_email.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE stg_users AS
SELECT
    user_id,
    account_id,
    email,
    role,
    TRY_CAST(created_at AS TIMESTAMPTZ) AS created_at,
    (email LIKE '%@%')              AS is_valid_email
FROM raw_users
WHERE user_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- stg_subscriptions
-- No generator-injected dirt beyond nullable end_at (expected business data).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE stg_subscriptions AS
SELECT
    subscription_id,
    account_id,
    plan,
    TRY_CAST(start_at AS TIMESTAMPTZ) AS start_at,
    TRY_CAST(end_at   AS TIMESTAMPTZ) AS end_at,
    status
FROM raw_subscriptions
WHERE subscription_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- stg_payments
-- Dirty data: ~0.3% null amount -> drop (no business value for amount-less payments)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE stg_payments AS
SELECT
    payment_id,
    subscription_id,
    TRY_CAST(paid_at AS TIMESTAMPTZ) AS paid_at,
    amount,
    currency,
    status
FROM raw_payments
WHERE payment_id IS NOT NULL
  AND amount IS NOT NULL;

-- ---------------------------------------------------------------------------
-- stg_events
-- Dirty data: ~0.2% null occurred_at, ~0.2% 'unknown_event' type -> drop both
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE stg_events AS
SELECT
    event_id,
    account_id,
    user_id,
    event_type,
    TRY_CAST(occurred_at AS TIMESTAMPTZ) AS occurred_at,
    device,
    os
FROM raw_events
WHERE occurred_at IS NOT NULL
  AND event_type <> 'unknown_event';
