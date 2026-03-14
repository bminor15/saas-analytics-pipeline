# src/run_pipeline.py
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

from src.config import GeneratorConfig

SQL_STEPS = [
    "sql/00_raw_ddl.sql",
    "sql/10_staging.sql",
    "sql/20_warehouse_dims.sql",
    "sql/30_warehouse_facts.sql",
    "sql/40_marts.sql",
]


def _run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8").strip()
    if not sql:
        print(f"  [skip]  {path.name}  (empty)")
        return
    t0 = time.perf_counter()
    con.execute(sql)
    elapsed = time.perf_counter() - t0
    print(f"  [ok]    {path.name}  ({elapsed:.1f}s)")


def main() -> None:
    paths = GeneratorConfig.paths()
    project_root = paths.project_root

    # Ensure DuckDB resolves relative paths (e.g. data/raw/...) from project root
    os.chdir(project_root)

    db_path = project_root / "warehouse" / "dev.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {db_path.relative_to(project_root)}")
    con = duckdb.connect(str(db_path))

    print("Running pipeline steps:")
    for step in SQL_STEPS:
        _run_sql_file(con, project_root / step)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
