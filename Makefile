DB=warehouse/dev.duckdb

run:
	python src/run_pipeline.py

gen:
	python -m src.generate_data --size $(or $(SIZE),medium)

check-raw:
	python -c "import pathlib; import pyarrow.parquet as pq; import pandas as pd; p=pathlib.Path('data/raw/events.parquet'); t=pq.ParquetFile(p); print('events.parquet rows:', t.metadata.num_rows); print('events.parquet schema:'); print(t.schema); print('raw csv row counts:'); print(pd.read_csv('data/raw/accounts.csv').shape[0], 'accounts'); print(pd.read_csv('data/raw/users.csv').shape[0], 'users'); print(pd.read_csv('data/raw/subscriptions.csv').shape[0], 'subscriptions'); print(pd.read_csv('data/raw/payments.csv').shape[0], 'payments')"
