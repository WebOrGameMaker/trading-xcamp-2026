"""Technical indicator feature engineering using pandas-ta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.logging import get_logger

logger = get_logger(__name__)

METADATA_COLUMNS = {
    "symbol",
    "date",
    "label",
    "forward_return_5d",
    "predicted_rank",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "probability",
    "prediction",
}


def _is_forward_return_column(name: str) -> bool:
    """Return True for forward-return / label metadata columns."""
    return name.startswith("forward_return_") or name in METADATA_COLUMNS


def _safe_indicator(value: pd.Series | None, index: pd.Index) -> pd.Series:
    """Normalize a pandas-ta indicator result to a NaN-filled Series.

    pandas-ta returns ``None`` (instead of an all-NaN Series) when the input
    is shorter than the indicator's lookback window, which would otherwise
    crash any downstream arithmetic (e.g. ``close / sma_50`` on a symbol
    with under 50 bars of history). Treating "not enough data yet" as NaN
    keeps behavior identical once enough history has accumulated while
    staying safe for short histories (new listings, small test fixtures).

    Args:
        value: Indicator output, or None if pandas-ta rejected the input.
        index: Index to align a fallback all-NaN Series to.

    Returns:
        The original Series, or an all-NaN Series matching ``index``.
    """
    if value is None:
        return pd.Series(np.nan, index=index)
    return value


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple return features.

    Args:
        df: OHLCV DataFrame with 'close' column.

    Returns:
        DataFrame with return columns.
    """
    features = pd.DataFrame(index=df.index)
    features["return_1d"] = df["close"].pct_change(1)
    features["return_5d"] = df["close"].pct_change(5)
    features["return_20d"] = df["close"].pct_change(20)
    return features


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend-based technical indicators.

    All outputs are scale-free (ratios or price-normalized values) rather
    than raw price levels. Raw SMA/EMA/MACD levels grow with a stock's price
    over a multi-year sample, so a classifier trained on 2012-2022 prices
    sees out-of-distribution inputs on 2023-2026 test prices. Normalizing by
    the current close keeps every feature stationary across the full history
    and comparable across symbols.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with trend indicator columns.
    """
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    sma_10 = _safe_indicator(ta.sma(close, length=10), df.index)
    sma_20 = _safe_indicator(ta.sma(close, length=20), df.index)
    sma_50 = _safe_indicator(ta.sma(close, length=50), df.index)
    ema_12 = _safe_indicator(ta.ema(close, length=12), df.index)
    ema_26 = _safe_indicator(ta.ema(close, length=26), df.index)

    features["price_sma10_ratio"] = close / sma_10
    features["price_sma20_ratio"] = close / sma_20
    features["price_sma50_ratio"] = close / sma_50
    features["price_ema12_ratio"] = close / ema_12
    features["price_ema26_ratio"] = close / ema_26

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # Express MACD components as a fraction of price instead of raw
        # dollar values so they stay on a comparable, stationary scale.
        features["macd_pct"] = macd.iloc[:, 0] / close
        features["macd_signal_pct"] = macd.iloc[:, 1] / close
        features["macd_hist_pct"] = macd.iloc[:, 2] / close

    return features


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum-based technical indicators.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with momentum indicator columns.
    """
    features = pd.DataFrame(index=df.index)
    features["rsi_14"] = _safe_indicator(ta.rsi(df["close"], length=14), df.index)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch is not None and not stoch.empty:
        features["stoch_k"] = stoch.iloc[:, 0]
        features["stoch_d"] = stoch.iloc[:, 1]

    return features


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility-based technical indicators.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with volatility indicator columns.
    """
    features = pd.DataFrame(index=df.index)
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        upper = bbands.iloc[:, 0]
        lower = bbands.iloc[:, 2]
        mid = bbands.iloc[:, 1]
        features["bb_width"] = (upper - lower) / mid.replace(0, pd.NA)

    atr = _safe_indicator(ta.atr(df["high"], df["low"], df["close"], length=14), df.index)
    # ATR as a fraction of price ("ATR%") instead of a raw dollar amount, so
    # it is stationary as price drifts higher across the sample.
    features["atr_pct"] = atr / df["close"].replace(0, pd.NA)
    features["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    return features


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based technical indicators.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with volume indicator columns.
    """
    features = pd.DataFrame(index=df.index)

    # Raw OBV is an unbounded cumulative sum that grows over a symbol's
    # entire history, making it non-stationary (test-period values are far
    # outside the range seen in training). A rolling z-score turns it into a
    # bounded oscillator that measures whether accumulation/distribution is
    # currently running hot or cold relative to its own recent history.
    obv = _safe_indicator(ta.obv(df["close"], df["volume"]), df.index)
    obv_mean = obv.rolling(60, min_periods=20).mean()
    obv_std = obv.rolling(60, min_periods=20).std()
    features["obv_zscore_60"] = (obv - obv_mean) / obv_std.replace(0, pd.NA)

    vol_sma = _safe_indicator(ta.sma(df["volume"], length=20), df.index)
    features["volume_sma_ratio"] = df["volume"] / vol_sma.replace(0, pd.NA)
    return features


def engineer_features(df: pd.DataFrame, indicator_set: str = "standard") -> pd.DataFrame:
    """Engineer all technical features for a single symbol.

    Args:
        df: OHLCV DataFrame with datetime index.
        indicator_set: Feature preset name ('standard' supported).

    Returns:
        DataFrame combining OHLCV close and all feature columns.
    """
    if indicator_set != "standard":
        raise ValueError(f"Unknown indicator set: {indicator_set}")

    parts = [
        compute_returns(df),
        compute_trend_features(df),
        compute_momentum_features(df),
        compute_volatility_features(df),
        compute_volume_features(df),
    ]
    features = pd.concat(parts, axis=1)
    features["close"] = df["close"]
    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature column names from a feature DataFrame.

    Excludes metadata, OHLCV, labels, forward returns, and prediction columns.
    Ticker identity is never included as a feature.

    Args:
        df: Feature DataFrame.

    Returns:
        List of numeric feature column names.
    """
    return [
        col
        for col in df.columns
        if not _is_forward_return_column(col)
        and not col.startswith("forward_return_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
