"""Tests for universe loading."""

from src.data.universe import load_universe


def test_load_sp100_universe() -> None:
    """S&P 100 universe loads with expected tickers."""
    universe = load_universe("sp100_tickers.yaml")
    assert len(universe.tickers) >= 90
    assert "AAPL" in universe.tickers
    assert "MSFT" in universe.tickers
    assert universe.tickers == sorted(universe.tickers)


def test_tickers_uppercased() -> None:
    """Tickers are normalized to uppercase."""
    universe = load_universe("sp100_tickers.yaml")
    assert all(t == t.upper() for t in universe.tickers)
