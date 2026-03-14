-- staging: cast types, flag/drop dirty rows injected by the generator

-- ~3% of accounts have null industry, coalesce to 'Unknown'
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

-- ~1% of emails are 'bad_email' (no @ sign) - flag but keep the row
-- downstream filters on is_valid_email
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

-- end_at is legitimately nullable for active subs, nothing to clean here
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

-- ~0.3% null amounts - drop them, a payment without an amount is useless
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

-- ~0.2% null timestamps, ~0.2% 'unknown_event' type - drop both
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
