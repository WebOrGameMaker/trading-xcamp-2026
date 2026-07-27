"""Download historical market data from Yahoo Finance."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data.cleaner import clean_ohlcv
from src.data.storage import load_raw_bars, save_raw_bars
from src.data.universe import load_universe
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Yahoo Finance uses different ticker formats for some symbols.
_YFINANCE_SYMBOL_MAP = {
    "BRK.B": "BRK-B",
}


def _to_yfinance_symbol(symbol: str) -> str:
    """Map internal ticker symbols to Yahoo Finance format.

    Args:
        symbol: Internal ticker symbol.

    Returns:
        Yahoo Finance compatible ticker.
    """
    return _YFINANCE_SYMBOL_MAP.get(symbol.upper(), symbol.upper())


def _history_to_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance history output to standard OHLCV schema.

    Args:
        raw: Raw DataFrame returned by Ticker.history().

    Returns:
        DataFrame indexed by timestamp with lowercase OHLCV columns.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index()


def download_symbol_bars(
    symbol: str,
    start: datetime,
    end: datetime | None,
) -> pd.DataFrame:
    """Download OHLCV bars for a single symbol from Yahoo Finance.

    Args:
        symbol: Ticker symbol.
        start: Start datetime (inclusive).
        end: End datetime (inclusive), or None for latest available.

    Returns:
        DataFrame of OHLCV bars.
    """
    yf_symbol = _to_yfinance_symbol(symbol)
    yf_end = None
    if end is not None:
        # yfinance end date is exclusive.
        yf_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    ticker = yf.Ticker(yf_symbol)
    raw = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=yf_end,
        # Split- and dividend-adjust OHLC so a stock split (e.g. NVDA 10:1,
        # AVGO 10:1) does not appear as a fake ~90% single-day price move in
        # returns, indicators, forward-return labels, or backtest P&L.
        auto_adjust=True,
    )
    return _history_to_dataframe(raw)


def download_universe(config: AppConfig, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download and cache OHLCV data for all symbols in the configured universe.

    Args:
        config: Application configuration.
        force: If True, re-download even when cache exists.

    Returns:
        Mapping of symbol to cleaned DataFrame.
    """
    if config.data.timeframe != "1Day":
        raise ValueError("yfinance downloader supports daily bars only (timeframe='1Day')")

    universe = load_universe(config.data.universe_file)
    logger.info("Downloading %d symbols from universe '%s'", len(universe.tickers), universe.name)

    start = datetime.fromisoformat(config.data.start_date)
    end = None
    if config.data.end_date:
        end = datetime.fromisoformat(config.data.end_date)

    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, symbol in enumerate(universe.tickers):
        if not force:
            cached = load_raw_bars(symbol)
            if cached is not None and len(cached) >= config.data.min_bars:
                cleaned = clean_ohlcv(cached, symbol, config.data.min_bars)
                if cleaned is not None:
                    logger.debug("Using cached data for %s (%d bars)", symbol, len(cleaned))
                    results[symbol] = cleaned
                    continue

        try:
            raw = download_symbol_bars(symbol, start, end)
            cleaned = clean_ohlcv(raw, symbol, config.data.min_bars)
            if cleaned is not None:
                save_raw_bars(symbol, cleaned)
                results[symbol] = cleaned
                logger.info("Downloaded %s: %d bars", symbol, len(cleaned))
            else:
                failed.append(symbol)
        except Exception as exc:
            logger.error("Failed to download %s: %s", symbol, exc)
            failed.append(symbol)

        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    logger.info(
        "Download complete: %d succeeded, %d failed",
        len(results),
        len(failed),
    )
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))

    return results
