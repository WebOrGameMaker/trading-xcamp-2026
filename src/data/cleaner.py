"""Data cleaning and validation for OHLCV bars."""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(df: pd.DataFrame, symbol: str, min_bars: int = 252) -> pd.DataFrame | None:
    """Clean and validate OHLCV data for a single symbol.

    Args:
        df: Raw OHLCV DataFrame with datetime index.
        symbol: Ticker symbol for logging.
        min_bars: Minimum required bar count after cleaning.

    Returns:
        Cleaned DataFrame or None if validation fails.
    """
    if df is None or df.empty:
        logger.warning("No data for %s", symbol)
        return None

    cleaned = df.copy()
    cleaned.columns = [str(c).lower() for c in cleaned.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in cleaned.columns]
    if missing:
        logger.warning("Missing columns %s for %s", missing, symbol)
        return None

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)

    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna()

    cleaned = cleaned[cleaned["volume"] >= 0]
    cleaned = cleaned[(cleaned["high"] >= cleaned["low"]) & (cleaned["close"] > 0)]

    if len(cleaned) < min_bars:
        logger.warning(
            "Insufficient bars for %s: %d < %d",
            symbol,
            len(cleaned),
            min_bars,
        )
        return None

    return cleaned


def detect_large_gaps(df: pd.DataFrame, max_gap_days: int = 10) -> bool:
    """Detect abnormally large gaps between consecutive trading days.

    Args:
        df: OHLCV DataFrame with datetime index.
        max_gap_days: Maximum allowed gap in calendar days.

    Returns:
        True if any gap exceeds threshold.
    """
    if len(df) < 2:
        return False
    gaps = df.index.to_series().diff().dt.days.dropna()
    return bool((gaps > max_gap_days).any())
