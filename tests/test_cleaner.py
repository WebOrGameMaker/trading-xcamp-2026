"""Tests for OHLCV data cleaning."""

import pandas as pd
import pytest

from src.data.cleaner import clean_ohlcv, detect_large_gaps


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), index=dates, dtype=float)
    return pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1_000_000,
    }, index=dates)


def test_clean_removes_duplicates() -> None:
    """Duplicate dates are removed keeping last."""
    df = _make_ohlcv(260)
    df = pd.concat([df, df.iloc[[0]]])
    cleaned = clean_ohlcv(df, "TEST", min_bars=252)
    assert cleaned is not None
    assert not cleaned.index.duplicated().any()


def test_clean_sorts_index() -> None:
    """Output index is sorted ascending."""
    df = _make_ohlcv(260)
    df = df.iloc[::-1]
    cleaned = clean_ohlcv(df, "TEST", min_bars=252)
    assert cleaned is not None
    assert cleaned.index.is_monotonic_increasing


def test_clean_rejects_insufficient_bars() -> None:
    """Returns None when bar count below minimum."""
    df = _make_ohlcv(100)
    assert clean_ohlcv(df, "TEST", min_bars=252) is None


def test_detect_large_gaps() -> None:
    """Detects calendar gaps exceeding threshold."""
    df = _make_ohlcv(10)
    assert not detect_large_gaps(df, max_gap_days=10)

    gap_df = df.drop(df.index[3:7])
    assert detect_large_gaps(gap_df, max_gap_days=5)
