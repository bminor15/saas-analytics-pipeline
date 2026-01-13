# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dir: Path


@dataclass(frozen=True)
class MediumRowCounts:
    accounts: int = 5_000
    users: int = 50_000
    subscriptions: int = 6_000
    payments: int = 120_000
    events: int = 4_000_000  # scalable via config


@dataclass(frozen=True)
class SmallRowCounts:
    accounts: int = 500
    users: int = 5_000
    subscriptions: int = 650
    payments: int = 12_000
    events: int = 300_000


@dataclass(frozen=True)
class LargeRowCounts:
    accounts: int = 25_000
    users: int = 250_000
    subscriptions: int = 30_000
    payments: int = 600_000
    events: int = 20_000_000


@dataclass(frozen=True)
class GeneratorConfig:
    dataset_size: str = "medium"  # "small" | "medium" | "large"
    seed: int = 42

    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

    # Events are written in chunks to keep memory stable
    events_chunk_rows: int = 250_000

    @property
    def rows(self):
        if self.dataset_size.lower() == "small":
            return SmallRowCounts()
        if self.dataset_size.lower() == "large":
            return LargeRowCounts()
        return MediumRowCounts()  # default

    @staticmethod
    def paths() -> Paths:
        # repo-root/src/config.py -> repo-root
        root = Path(__file__).resolve().parents[1]
        return Paths(project_root=root, raw_dir=root / "data" / "raw")
