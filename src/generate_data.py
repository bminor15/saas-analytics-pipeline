# src/generate_data.py
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker

from src.config import GeneratorConfig


def _date_range_seconds(start: str, end: str) -> tuple[int, int]:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _random_timestamps_iso(rng: np.random.Generator, n: int, start: str, end: str) -> np.ndarray:
    lo, hi = _date_range_seconds(start, end)
    # inclusive-ish, good enough for synthetic data
    secs = rng.integers(lo, hi, size=n, endpoint=False, dtype=np.int64)
    dts = pd.to_datetime(secs, unit="s", utc=True)
    # ISO 8601 strings (Snowflake-friendly)
    return dts.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()


def _weighted_choice(rng: np.random.Generator, values: list[str], weights: list[float], n: int) -> np.ndarray:
    p = np.array(weights, dtype=float)
    p = p / p.sum()
    idx = rng.choice(len(values), size=n, replace=True, p=p)
    return np.array(values, dtype=object)[idx]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_accounts(cfg: GeneratorConfig, fake: Faker, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.rows.accounts

    industries = ["SaaS", "Ecommerce", "Healthcare", "Finance", "Education", "Media", "Retail", "Logistics"]
    regions = ["NA", "EMEA", "APAC", "LATAM"]
    tiers = ["SMB", "Mid-Market", "Enterprise"]

    created_at = _random_timestamps_iso(rng, n, cfg.start_date, cfg.end_date)
    account_id = np.arange(1, n + 1, dtype=np.int64)

    df = pd.DataFrame(
        {
            "account_id": account_id,
            "account_name": [fake.company() for _ in range(n)],
            "industry": _weighted_choice(rng, industries, [40, 15, 10, 10, 10, 5, 5, 5], n),
            "region": _weighted_choice(rng, regions, [55, 20, 15, 10], n),
            "tier": _weighted_choice(rng, tiers, [65, 25, 10], n),
            "created_at": created_at,
        }
    )

    # A touch of “realism”: ~3% missing industry (forces staging cleanup)
    mask = rng.random(n) < 0.03
    df.loc[mask, "industry"] = None

    return df


def generate_users(cfg: GeneratorConfig, fake: Faker, rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    n = cfg.rows.users
    account_ids = accounts["account_id"].to_numpy()

    # Skew: some accounts have many more users
    # Using a lognormal-ish weight distribution for account selection
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=len(account_ids))
    weights = weights / weights.sum()
    user_account = rng.choice(account_ids, size=n, replace=True, p=weights)

    roles = ["admin", "member", "viewer"]
    created_at = _random_timestamps_iso(rng, n, cfg.start_date, cfg.end_date)

    user_id = np.arange(1, n + 1, dtype=np.int64)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "account_id": user_account.astype(np.int64),
            "email": [fake.email() for _ in range(n)],
            "role": _weighted_choice(rng, roles, [8, 82, 10], n),
            "created_at": created_at,
        }
    )

    # ~1% malformed emails (forces staging validation)
    bad_mask = rng.random(n) < 0.01
    df.loc[bad_mask, "email"] = "bad_email"

    return df


def generate_subscriptions(cfg: GeneratorConfig, rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    n = cfg.rows.subscriptions
    account_ids = accounts["account_id"].to_numpy()

    plans = ["Free", "Starter", "Pro", "Business", "Enterprise"]
    plan_weights = [10, 35, 35, 15, 5]

    sub_id = np.arange(1, n + 1, dtype=np.int64)
    account_id = rng.choice(account_ids, size=n, replace=True)

    start_at = _random_timestamps_iso(rng, n, cfg.start_date, cfg.end_date)
    plan = _weighted_choice(rng, plans, plan_weights, n)

    # Some churn: end_at nullable
    churn_mask = rng.random(n) < 0.30
    end_at = np.array([None] * n, dtype=object)
    # For churned subs, end date is after start date by 30–365 days
    add_days = rng.integers(30, 365, size=churn_mask.sum(), dtype=np.int64)

    start_dt = pd.to_datetime(start_at, utc=True)
    churned_end = (start_dt[churn_mask] + pd.to_timedelta(add_days, unit="D")).strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    end_at[churn_mask] = churned_end

    df = pd.DataFrame(
        {
            "subscription_id": sub_id,
            "account_id": account_id.astype(np.int64),
            "plan": plan,
            "start_at": start_at,
            "end_at": end_at,
            "status": np.where(churn_mask, "canceled", "active"),
        }
    )

    return df


def generate_payments(cfg: GeneratorConfig, rng: np.random.Generator, subs: pd.DataFrame) -> pd.DataFrame:
    n = cfg.rows.payments

    # Tie payments to subscriptions
    sub_ids = subs["subscription_id"].to_numpy()
    payment_id = np.arange(1, n + 1, dtype=np.int64)

    subscription_id = rng.choice(sub_ids, size=n, replace=True)
    paid_at = _random_timestamps_iso(rng, n, cfg.start_date, cfg.end_date)

    # Amount roughly aligns with plan (we'll approximate by joining later in staging if desired)
    # For now: lognormal distribution to mimic SaaS billing variability
    amounts = rng.lognormal(mean=4.2, sigma=0.6, size=n)  # ~ 30–300-ish typical
    amounts = np.round(amounts, 2)

    currency = _weighted_choice(rng, ["USD", "EUR", "GBP"], [85, 10, 5], n)

    # Payment outcomes (forces lifecycle logic later)
    status = _weighted_choice(rng, ["paid", "failed", "refunded"], [92, 6, 2], n)

    df = pd.DataFrame(
        {
            "payment_id": payment_id,
            "subscription_id": subscription_id.astype(np.int64),
            "paid_at": paid_at,
            "amount": amounts.astype(float),
            "currency": currency,
            "status": status,
        }
    )

    # Sprinkle some null amounts for staging cleanup (rare)
    null_mask = rng.random(n) < 0.003
    df.loc[null_mask, "amount"] = None

    return df


def _event_batches(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    batches = math.ceil(total / batch_size)
    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        yield start, end


def generate_events_parquet(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    users: pd.DataFrame,
    out_path: Path,
) -> None:
    total = cfg.rows.events
    chunk = cfg.events_chunk_rows

    user_ids = users["user_id"].to_numpy()
    account_ids_by_user = users.set_index("user_id")["account_id"]

    event_types = ["login", "page_view", "click", "export", "api_call", "error"]
    event_weights = [10, 45, 30, 5, 8, 2]

    # We’ll write a single parquet file via a ParquetWriter for chunked scalability.
    writer: pq.ParquetWriter | None = None

    for (start, end) in _event_batches(total, chunk):
        n = end - start
        event_id = np.arange(start + 1, end + 1, dtype=np.int64)

        # User selection skew: active users generate more events
        # Use weights based on a lognormal draw each chunk for simplicity
        w = rng.lognormal(mean=0.0, sigma=1.2, size=len(user_ids))
        w = w / w.sum()
        chosen_users = rng.choice(user_ids, size=n, replace=True, p=w).astype(np.int64)

        # Map to account_id (denormalized in raw to mimic event logs)
        chosen_accounts = account_ids_by_user.loc[chosen_users].to_numpy(dtype=np.int64)

        occurred_at = _random_timestamps_iso(rng, n, cfg.start_date, cfg.end_date)
        event_type = _weighted_choice(rng, event_types, event_weights, n)

        # Simple device/os fields for realism
        device = _weighted_choice(rng, ["web", "ios", "android"], [78, 12, 10], n)
        os = _weighted_choice(rng, ["windows", "mac", "linux", "ios", "android"], [55, 20, 8, 9, 8], n)

        # A few bad rows to justify staging (null timestamps / unknown types)
        bad_ts = rng.random(n) < 0.002
        bad_type = rng.random(n) < 0.002
        occurred_at = occurred_at.astype(object)
        event_type = event_type.astype(object)
        occurred_at[bad_ts] = None
        event_type[bad_type] = "unknown_event"

        df = pd.DataFrame(
            {
                "event_id": event_id,
                "account_id": chosen_accounts,
                "user_id": chosen_users,
                "event_type": event_type,
                "occurred_at": occurred_at,
                "device": device,
                "os": os,
            }
        )

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path.as_posix(), table.schema, compression="zstd")

        writer.write_table(table)

    if writer is not None:
        writer.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic SaaS data.")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--events-chunk-rows", type=int, default=250_000)
    return parser


def _build_config(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        dataset_size=args.size,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
        events_chunk_rows=args.events_chunk_rows,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _build_config(args)
    paths = cfg.paths()
    _ensure_dir(paths.raw_dir)

    fake = Faker()
    fake.seed_instance(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Generate and write raw files
    accounts = generate_accounts(cfg, fake, rng)
    accounts_path = paths.raw_dir / "accounts.csv"
    accounts.to_csv(accounts_path, index=False)

    users = generate_users(cfg, fake, rng, accounts)
    users_path = paths.raw_dir / "users.csv"
    users.to_csv(users_path, index=False)

    subs = generate_subscriptions(cfg, rng, accounts)
    subs_path = paths.raw_dir / "subscriptions.csv"
    subs.to_csv(subs_path, index=False)

    payments = generate_payments(cfg, rng, subs)
    payments_path = paths.raw_dir / "payments.csv"
    payments.to_csv(payments_path, index=False)

    events_path = paths.raw_dir / "events.parquet"
    generate_events_parquet(cfg, rng, users, events_path)

    print("OK - Synthetic raw data generated:")
    print(f"  - {accounts_path}")
    print(f"  - {users_path}")
    print(f"  - {subs_path}")
    print(f"  - {payments_path}")
    print(f"  - {events_path}")
    print("")
    print("Row counts:")
    print(f"  accounts:       {len(accounts):,}")
    print(f"  users:          {len(users):,}")
    print(f"  subscriptions:  {len(subs):,}")
    print(f"  payments:       {len(payments):,}")
    print(f"  events:         {cfg.rows.events:,} (written in chunks)")


if __name__ == "__main__":
    main()
