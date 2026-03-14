-- row counts across every layer - run via: make check
-- compare raw vs staging to verify expected drops (bad emails, null amounts, etc.)

SELECT 'raw_accounts'       AS table_name, COUNT(*) AS row_count FROM raw_accounts
UNION ALL
SELECT 'raw_users',                         COUNT(*) FROM raw_users
UNION ALL
SELECT 'raw_subscriptions',                 COUNT(*) FROM raw_subscriptions
UNION ALL
SELECT 'raw_payments',                      COUNT(*) FROM raw_payments
UNION ALL
SELECT 'raw_events',                        COUNT(*) FROM raw_events
UNION ALL
SELECT 'stg_accounts',                      COUNT(*) FROM stg_accounts
UNION ALL
SELECT 'stg_users',                         COUNT(*) FROM stg_users
UNION ALL
SELECT 'stg_subscriptions',                 COUNT(*) FROM stg_subscriptions
UNION ALL
SELECT 'stg_payments',                      COUNT(*) FROM stg_payments
UNION ALL
SELECT 'stg_events',                        COUNT(*) FROM stg_events
UNION ALL
SELECT 'dim_date',                          COUNT(*) FROM dim_date
UNION ALL
SELECT 'dim_account',                       COUNT(*) FROM dim_account
UNION ALL
SELECT 'dim_user',                          COUNT(*) FROM dim_user
UNION ALL
SELECT 'dim_plan',                          COUNT(*) FROM dim_plan
UNION ALL
SELECT 'fact_subscriptions',               COUNT(*) FROM fact_subscriptions
UNION ALL
SELECT 'fact_payments',                    COUNT(*) FROM fact_payments
UNION ALL
SELECT 'fact_events',                      COUNT(*) FROM fact_events
ORDER BY table_name;
