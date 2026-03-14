# SaaS Analytics Pipeline

A local analytics engineering pipeline using DuckDB. Synthetic SaaS data flows through a layered SQL architecture — raw → staging → warehouse → marts — built to mirror how this would work in a real data stack (Snowflake, dbt, etc.).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Generation                          │
│              src/generate_data.py  (Python + NumPy)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              data/raw/  (CSV + Parquet)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  00  RAW          DuckDB views over raw files (no copy)         │
├─────────────────────────────────────────────────────────────────┤
│  10  STAGING      Type casting, null handling, quality flags     │
├─────────────────────────────────────────────────────────────────┤
│  20  DIMS         dim_date, dim_account, dim_user, dim_plan      │
├─────────────────────────────────────────────────────────────────┤
│  30  FACTS        fact_subscriptions, fact_payments, fact_events │
├─────────────────────────────────────────────────────────────────┤
│  40  MARTS        MRR, churn, revenue, DAU, feature usage        │
└─────────────────────────────────────────────────────────────────┘
                            │
                    warehouse/dev.duckdb
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Warehouse | [DuckDB](https://duckdb.org/) |
| Transformations | SQL |
| Orchestration | Python 3.11+ |
| Data generation | Python, NumPy, Faker, PyArrow |
| Build | GNU Make |

---

## Quickstart

```bash
# install dependencies
pip install -r requirements.txt

# generate synthetic raw data (medium ~4M events)
make gen

# run the full pipeline
make run
```

Other dataset sizes:

```bash
make gen SIZE=small   # ~300k events
make gen SIZE=large   # ~20M events
```

Verify row counts at each layer:

```bash
make check
```

---

## Project Structure

```
saas-analytics-pipeline/
├── data/
│   └── raw/                  # generated source files (git-ignored)
│       ├── accounts.csv
│       ├── users.csv
│       ├── subscriptions.csv
│       ├── payments.csv
│       └── events.parquet
├── sql/
│   ├── 00_raw_ddl.sql        # views over raw files
│   ├── 10_staging.sql        # type casting + data quality
│   ├── 20_warehouse_dims.sql # dimension tables
│   ├── 30_warehouse_facts.sql# fact tables
│   ├── 40_marts.sql          # analytical views
│   └── 99_checks.sql         # row count checks
├── queries/                  # standalone analytical queries
├── src/
│   ├── config.py             # dataset sizing + paths
│   ├── generate_data.py      # synthetic data generator
│   └── run_pipeline.py       # pipeline runner
├── warehouse/                # DuckDB files (git-ignored)
├── Makefile
└── requirements.txt
```

---

## Dataset

Deterministic synthetic data (seed=42) across a 2-year window (2024–2025). Dirty data is intentionally injected at the source to exercise the staging layer.

| Entity | Medium rows | Injected issues |
|---|---|---|
| accounts | 5,000 | 3% null industry |
| users | 50,000 | 1% malformed emails |
| subscriptions | 6,000 | 30% churn rate |
| payments | 120,000 | 0.3% null amounts |
| events | 4,000,000 | 0.2% null timestamps, 0.2% unknown event types |

---

## Data Model

### Dimensions

| Table | Key attributes |
|---|---|
| `dim_date` | year, month, quarter, day_of_week, is_weekend |
| `dim_account` | industry, region, tier |
| `dim_user` | role, account_id (valid emails only) |
| `dim_plan` | plan_rank, list_price_usd |

### Facts

| Table | Key metrics |
|---|---|
| `fact_subscriptions` | mrr_usd, duration_days, is_churned |
| `fact_payments` | revenue_usd, is_paid, is_failed, is_refunded |
| `fact_events` | is_login, is_error, is_api_call |

### Marts

| View | What it answers |
|---|---|
| `mart_mrr_by_month` | MRR by plan, tier, region over time |
| `mart_churn_by_month` | Churned subscriptions per month |
| `mart_revenue_by_month` | Collected revenue + payment health |
| `mart_daily_active_users` | DAU + active accounts per day |
| `mart_feature_usage` | Feature usage by platform and tier |

---

## Sample Queries

**MRR by plan (latest month)**
```sql
SELECT plan, SUM(mrr_usd) AS mrr_usd, SUM(active_subscriptions) AS subs
FROM mart_mrr_by_month
WHERE year_month = (SELECT MAX(year_month) FROM mart_mrr_by_month)
GROUP BY plan
ORDER BY mrr_usd DESC;
```

**Monthly churn rate**
```sql
SELECT
    m.year_month,
    c.churned_subscriptions,
    SUM(m.active_subscriptions)                         AS active_subscriptions,
    ROUND(c.churned_subscriptions * 100.0
          / NULLIF(SUM(m.active_subscriptions), 0), 2)  AS churn_rate_pct
FROM mart_mrr_by_month m
LEFT JOIN (
    SELECT year_month, SUM(churned_subscriptions) AS churned_subscriptions
    FROM mart_churn_by_month
    GROUP BY year_month
) c USING (year_month)
GROUP BY m.year_month, c.churned_subscriptions
ORDER BY m.year_month;
```

**Revenue by account tier**
```sql
SELECT account_tier, SUM(total_revenue_usd) AS revenue_usd
FROM mart_revenue_by_month
GROUP BY account_tier
ORDER BY revenue_usd DESC;
```

**30-day rolling DAU**
```sql
SELECT
    date_day,
    SUM(dau) AS dau,
    AVG(SUM(dau)) OVER (ORDER BY date_day ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS dau_30d_avg
FROM mart_daily_active_users
GROUP BY date_day
ORDER BY date_day;
```
