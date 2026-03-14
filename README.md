# SaaS Analytics Pipeline

A local analytics engineering pipeline built with DuckDB, modelled on Snowflake-style layered architecture. Ingests synthetic SaaS data and transforms it through raw → staging → warehouse → marts layers using SQL.

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
| Transformations | SQL (Snowflake-style layered) |
| Orchestration | Python 3.11+ |
| Data generation | Python, NumPy, Faker, PyArrow |
| Build | GNU Make |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic raw data (medium dataset ~4M events)
make gen

# 3. Run the full pipeline
make run
```

Optional: generate a different dataset size

```bash
make gen SIZE=small   # ~300k events
make gen SIZE=large   # ~20M events
```

---

## Project Structure

```
saas-analytics-pipeline/
├── data/
│   └── raw/                  # Generated source files (git-ignored)
│       ├── accounts.csv
│       ├── users.csv
│       ├── subscriptions.csv
│       ├── payments.csv
│       └── events.parquet
├── sql/
│   ├── 00_raw_ddl.sql        # Raw views over source files
│   ├── 10_staging.sql        # Cleaning and type casting
│   ├── 20_warehouse_dims.sql # Dimension tables
│   ├── 30_warehouse_facts.sql# Fact tables
│   └── 40_marts.sql          # Analytical mart views
├── src/
│   ├── config.py             # Dataset size config and paths
│   ├── generate_data.py      # Synthetic data generator
│   └── run_pipeline.py       # Pipeline orchestrator
├── warehouse/                # DuckDB database files (git-ignored)
├── Makefile
└── requirements.txt
```

---

## Dataset

Synthetic SaaS data generated deterministically (seed = 42) across a 2-year window (2024–2025). Intentional data quality issues are injected at the source to exercise the staging layer.

| Entity | Medium rows | Notes |
|---|---|---|
| accounts | 5,000 | 3% null industry |
| users | 50,000 | 1% malformed emails |
| subscriptions | 6,000 | 30% churn rate |
| payments | 120,000 | 0.3% null amounts |
| events | 4,000,000 | 0.2% null timestamps, 0.2% unknown types |

---

## Data Model

### Dimensions

| Table | Grain | Key attributes |
|---|---|---|
| `dim_date` | One row per calendar day | year, month, quarter, day_of_week, is_weekend |
| `dim_account` | One row per account | industry, region, tier |
| `dim_user` | One row per valid user | role, account_id |
| `dim_plan` | One row per plan | plan_rank, list_price_usd |

### Facts

| Table | Grain | Key metrics |
|---|---|---|
| `fact_subscriptions` | One row per subscription | mrr_usd, duration_days, is_churned |
| `fact_payments` | One row per payment | revenue_usd, is_paid, is_failed, is_refunded |
| `fact_events` | One row per product event | is_login, is_error, is_api_call |

### Marts

| View | Business question |
|---|---|
| `mart_mrr_by_month` | What is MRR by plan, tier, and region over time? |
| `mart_churn_by_month` | How many subscriptions churned per month? |
| `mart_revenue_by_month` | What is actual collected revenue and payment health? |
| `mart_daily_active_users` | What is DAU / active account count per day? |
| `mart_feature_usage` | Which features are used most, and on which platforms? |

---

## Sample Queries

**Total MRR by plan (latest month)**
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

**30-day rolling DAU trend**
```sql
SELECT
    date_day,
    SUM(dau) AS dau,
    AVG(SUM(dau)) OVER (ORDER BY date_day ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS dau_30d_avg
FROM mart_daily_active_users
GROUP BY date_day
ORDER BY date_day;
```
