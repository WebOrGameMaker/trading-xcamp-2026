"""Parquet storage helpers for raw and processed market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def raw_symbol_path(symbol: str) -> Path:
    """Return path for a symbol's raw parquet file.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Path to data/raw/{symbol}.parquet
    """
    return RAW_DATA_DIR / f"{symbol.upper()}.parquet"


def save_raw_bars(symbol: str, df: pd.DataFrame) -> Path:
    """Persist OHLCV bars for a symbol to parquet.

    Args:
        symbol: Stock ticker symbol.
        df: DataFrame with datetime index and OHLCV columns.

    Returns:
        Path where data was written.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = raw_symbol_path(symbol)
    df.to_parquet(path, index=True)
    return path


def load_raw_bars(symbol: str) -> pd.DataFrame | None:
    """Load cached OHLCV bars for a symbol.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        DataFrame if cache exists, else None.
    """
    path = raw_symbol_path(symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_all_raw_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load cached bars for multiple symbols.

    Args:
        symbols: List of ticker symbols.

    Returns:
        Mapping of symbol to DataFrame for symbols with cached data.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_raw_bars(symbol)
        if df is not None and not df.empty:
            result[symbol] = df
    return result


def save_processed_dataset(df: pd.DataFrame, name: str = "features") -> Path:
    """Save processed feature dataset to parquet.

    Args:
        df: Processed feature DataFrame.
        name: Base filename without extension.

    Returns:
        Path where data was written.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed_dataset(name: str = "features") -> pd.DataFrame:
    """Load processed feature dataset from parquet.

    Args:
        name: Base filename without extension.

    Returns:
        Processed feature DataFrame.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_parquet(path)
