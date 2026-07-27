"""Load trading universe symbol lists from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.paths import CONFIG_DIR


@dataclass
class Universe:
    """Trading universe metadata."""

    name: str
    description: str
    tickers: list[str]


def load_universe(universe_file: str) -> Universe:
    """Load ticker universe from a YAML file in configs/.

    Args:
        universe_file: Filename under configs/ (e.g. sp100_tickers.yaml).

    Returns:
        Universe with deduplicated, uppercased tickers.
    """
    path = CONFIG_DIR / universe_file
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    tickers = sorted({str(t).upper().strip() for t in raw.get("tickers", []) if t})
    return Universe(
        name=str(raw.get("name", "universe")),
        description=str(raw.get("description", "")),
        tickers=tickers,
    )
