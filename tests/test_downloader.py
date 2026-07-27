"""Tests for yfinance market data downloader."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.downloader import _history_to_dataframe, _to_yfinance_symbol, download_symbol_bars
from src.utils.config import AppConfig, DataConfig


def test_to_yfinance_symbol_maps_class_shares() -> None:
    """BRK.B is mapped to Yahoo's BRK-B ticker format."""
    assert _to_yfinance_symbol("BRK.B") == "BRK-B"
    assert _to_yfinance_symbol("aapl") == "AAPL"


def test_history_to_dataframe_normalizes_columns() -> None:
    """yfinance history output is normalized to lowercase OHLCV."""
    raw = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000],
        "Dividends": [0.0],
        "Stock Splits": [0.0],
    }, index=pd.to_datetime(["2020-01-02"], utc=True))

    result = _history_to_dataframe(raw)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.tz is None
    assert result.iloc[0]["close"] == 100.5


@patch("src.data.downloader.yf.Ticker")
def test_download_symbol_bars_uses_yfinance_mapping(mock_ticker_cls: MagicMock) -> None:
    """Downloader queries Yahoo with mapped ticker symbols."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000],
    }, index=pd.to_datetime(["2020-01-02"]))
    mock_ticker_cls.return_value = mock_ticker

    result = download_symbol_bars("BRK.B", datetime(2012, 1, 1), datetime(2012, 12, 31))

    mock_ticker_cls.assert_called_once_with("BRK-B")
    mock_ticker.history.assert_called_once()
    assert not result.empty
    assert "close" in result.columns


@patch("src.data.downloader.yf.Ticker")
def test_download_symbol_bars_requests_split_dividend_adjusted_prices(
    mock_ticker_cls: MagicMock,
) -> None:
    """Prices must be adjusted so splits/dividends don't corrupt returns and labels."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000],
    }, index=pd.to_datetime(["2020-01-02"]))
    mock_ticker_cls.return_value = mock_ticker

    download_symbol_bars("AAPL", datetime(2012, 1, 1), datetime(2012, 12, 31))

    _, call_kwargs = mock_ticker.history.call_args
    assert call_kwargs["auto_adjust"] is True


@patch("src.data.downloader.download_symbol_bars")
@patch("src.data.downloader.load_universe")
def test_download_universe_does_not_require_alpaca_keys(
    mock_load_universe: MagicMock,
    mock_download_symbol: MagicMock,
) -> None:
    """Universe download works without Alpaca credentials."""
    from src.data.downloader import download_universe

    universe = MagicMock()
    universe.name = "test"
    universe.tickers = ["AAPL"]
    mock_load_universe.return_value = universe

    ohlcv = pd.DataFrame({
        "open": [100.0] * 300,
        "high": [101.0] * 300,
        "low": [99.0] * 300,
        "close": [100.5] * 300,
        "volume": [1_000_000] * 300,
    }, index=pd.date_range("2012-01-01", periods=300, freq="B"))
    mock_download_symbol.return_value = ohlcv

    config = AppConfig(data=DataConfig(min_bars=252))
    with patch("src.data.downloader.save_raw_bars", return_value=MagicMock()):
        with patch("src.data.downloader.load_raw_bars", return_value=None):
            results = download_universe(config, force=True)

    assert "AAPL" in results


def test_download_universe_rejects_non_daily_timeframe() -> None:
    """Only daily bars are supported for yfinance downloads."""
    from src.data.downloader import download_universe

    config = AppConfig(data=DataConfig(timeframe="1Hour"))
    with pytest.raises(ValueError, match="daily bars only"):
        download_universe(config)
