DB=warehouse/dev.duckdb

# Run the full pipeline: raw → staging → dims → facts → marts
run:
	python -m src.run_pipeline

# Generate synthetic raw data. Optional: SIZE=small|medium|large
gen:
	python -m src.generate_data --size $(or $(SIZE),medium)

# Check raw file row counts and schema
check-raw:
	python -c "import pathlib; import pyarrow.parquet as pq; import pandas as pd; p=pathlib.Path('data/raw/events.parquet'); t=pq.ParquetFile(p); print('events.parquet rows:', t.metadata.num_rows); print('events.parquet schema:'); print(t.schema); print('raw csv row counts:'); print(pd.read_csv('data/raw/accounts.csv').shape[0], 'accounts'); print(pd.read_csv('data/raw/users.csv').shape[0], 'users'); print(pd.read_csv('data/raw/subscriptions.csv').shape[0], 'subscriptions'); print(pd.read_csv('data/raw/payments.csv').shape[0], 'payments')"

# Validate row counts at each pipeline layer
check:
	python -c "import duckdb, pandas as pd; con = duckdb.connect('$(DB)'); df = con.execute(open('sql/99_checks.sql').read()).df(); print(df.to_string(index=False))"

# Preview MRR by month from the mart layer
sample:
	python -c "import duckdb, pandas as pd; con = duckdb.connect('$(DB)'); df = con.execute('SELECT year_month, SUM(mrr_usd) AS mrr_usd, SUM(active_subscriptions) AS subs FROM mart_mrr_by_month GROUP BY year_month ORDER BY year_month').df(); print(df.to_string(index=False))"

# Export gold layer dims and facts to exports/ as Parquet for Power BI
export:
	python -c "import duckdb, pathlib; pathlib.Path('exports').mkdir(exist_ok=True); con = duckdb.connect('$(DB)'); tables = ['dim_date','dim_account','dim_user','dim_plan','fact_subscriptions','fact_payments','fact_events']; [con.execute(f\"COPY {t} TO 'exports/{t}.parquet' (FORMAT PARQUET)\") or print(f'  [ok]    {t}.parquet') for t in tables]; print('Done.')"
