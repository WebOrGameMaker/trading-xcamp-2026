"""Tests for label generation — no lookahead leakage."""

import pandas as pd
import pytest

from src.features.engineer import engineer_features
from src.features.labels import (
    assign_cross_sectional_labels,
    compute_forward_returns,
    drop_unlabeled_rows,
    generate_labels,
)


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create synthetic OHLCV for testing."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series([100 + i * 0.5 for i in range(n)], index=dates)
    return pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1_000_000,
    }, index=dates)


def test_forward_return_uses_future_close_only() -> None:
    """Forward return at t uses close at t+5, not past data."""
    ohlcv = _make_ohlcv(20)
    features = engineer_features(ohlcv)
    labeled = generate_labels(features, horizon_days=5, threshold=0.0)

    t = 0
    expected = ohlcv["close"].iloc[t + 5] / ohlcv["close"].iloc[t] - 1
    assert abs(labeled["forward_return_5d"].iloc[t] - expected) < 1e-10


def test_features_use_only_past_data() -> None:
    """Return features at t depend only on data up to t."""
    ohlcv = _make_ohlcv(50)
    features = engineer_features(ohlcv)

    t = 30
    truncated = ohlcv.iloc[: t + 1]
    trunc_features = engineer_features(truncated)

    for col in ["return_1d", "return_5d", "price_sma10_ratio", "atr_pct"]:
        if col in features.columns and not pd.isna(features[col].iloc[t]):
            assert features[col].iloc[t] == pytest.approx(trunc_features[col].iloc[t], rel=1e-6)


def test_labels_drop_last_horizon_rows() -> None:
    """Last horizon_days rows have NaN labels before drop."""
    ohlcv = _make_ohlcv(30)
    features = engineer_features(ohlcv)
    labeled = generate_labels(features, horizon_days=5)
    assert labeled["forward_return_5d"].iloc[-5:].isna().all()

    cleaned = drop_unlabeled_rows(
        labeled.reset_index().rename(columns={"index": "date"}),
        horizon_days=5,
    )
    assert cleaned["label"].notna().all()
    assert len(cleaned) == len(labeled) - 5 - labeled.iloc[:-5].isna().any(axis=1).sum()


def test_binary_label_threshold() -> None:
    """Labels respect configurable threshold."""
    ohlcv = _make_ohlcv(30)
    features = engineer_features(ohlcv)
    labeled = generate_labels(features, horizon_days=5, threshold=0.01)
    valid = labeled.dropna(subset=["forward_return_5d"])
    for _, row in valid.iterrows():
        if row["forward_return_5d"] > 0.01:
            assert row["label"] == 1
        else:
            assert row["label"] == 0


def test_cross_sectional_labels_mark_top_quantile() -> None:
    """Within each date, only the top positive_quantile of returns are labeled 1."""
    dates = pd.to_datetime(["2024-01-02"] * 10 + ["2024-01-03"] * 10)
    symbols = [f"S{i}" for i in range(10)] * 2
    # First date: returns 0..0.09; second date: returns 0.09..0.00 descending.
    returns = list(range(10)) + list(range(9, -1, -1))
    frame = pd.DataFrame({
        "date": dates,
        "symbol": symbols,
        "forward_return_5d": [r / 100 for r in returns],
        "close": 100.0,
    })

    labeled = assign_cross_sectional_labels(
        frame,
        horizon_days=5,
        positive_quantile=0.20,
        min_names=10,
    )

    day1 = labeled[labeled["date"] == "2024-01-02"]
    # Top 20% of 10 = 2 names (ranks 9 and 10 → pct 0.9, 1.0).
    assert int(day1["label"].sum()) == 2
    assert set(day1.loc[day1["label"] == 1, "symbol"]) == {"S8", "S9"}

    day2 = labeled[labeled["date"] == "2024-01-03"]
    assert int(day2["label"].sum()) == 2
    assert set(day2.loc[day2["label"] == 1, "symbol"]) == {"S0", "S1"}


def test_cross_sectional_labels_skip_thin_cross_section() -> None:
    """Dates with fewer than min_names get NaN labels."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 3),
        "symbol": ["A", "B", "C"],
        "forward_return_5d": [0.01, 0.02, 0.03],
        "close": 100.0,
    })
    labeled = assign_cross_sectional_labels(
        frame,
        horizon_days=5,
        positive_quantile=0.20,
        min_names=10,
    )
    assert labeled["label"].isna().all()


def test_compute_forward_returns_column_name_matches_horizon() -> None:
    """Forward-return column name tracks horizon_days."""
    ohlcv = _make_ohlcv(20)
    features = engineer_features(ohlcv)
    labeled = compute_forward_returns(features, horizon_days=3)
    assert "forward_return_3d" in labeled.columns
    assert labeled["forward_return_3d"].iloc[-3:].isna().all()
