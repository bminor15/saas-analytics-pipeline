-- views over the raw source files
-- paths resolve relative to project root

CREATE OR REPLACE VIEW raw_accounts AS
SELECT * FROM read_csv('data/raw/accounts.csv', header = true, auto_detect = true);

CREATE OR REPLACE VIEW raw_users AS
SELECT * FROM read_csv('data/raw/users.csv', header = true, auto_detect = true);

CREATE OR REPLACE VIEW raw_subscriptions AS
SELECT * FROM read_csv('data/raw/subscriptions.csv', header = true, auto_detect = true);

CREATE OR REPLACE VIEW raw_payments AS
SELECT * FROM read_csv('data/raw/payments.csv', header = true, auto_detect = true);

CREATE OR REPLACE VIEW raw_events AS
SELECT * FROM read_parquet('data/raw/events.parquet');
