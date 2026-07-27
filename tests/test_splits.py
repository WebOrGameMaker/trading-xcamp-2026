"""Tests for calendar-based dataset splits."""

import pandas as pd
import pytest

from src.data.splits import calendar_split


def _make_frame(start: str, periods: int) -> pd.DataFrame:
    """Build a dated feature frame for split tests."""
    return pd.DataFrame({
        "date": pd.date_range(start, periods=periods, freq="B"),
        "symbol": "AAPL",
        "close": range(periods),
        "label": 0,
    })


def test_calendar_split_respects_date_boundaries() -> None:
    """Train, val, and test pools stay within configured calendar windows."""
    frame = _make_frame("2010-01-01", 4500)

    train, val, test = calendar_split(
        frame,
        train_end_date="2022-12-31",
        val_start_date="2023-01-01",
        val_end_date="2024-12-31",
        test_start_date="2025-01-01",
    )

    assert train["date"].max() <= pd.Timestamp("2022-12-31")
    assert val["date"].min() >= pd.Timestamp("2023-01-01")
    assert val["date"].max() <= pd.Timestamp("2024-12-31")
    assert test["date"].min() >= pd.Timestamp("2025-01-01")
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_calendar_split_validation_uses_explicit_window() -> None:
    """Validation rows come from the configured 2023-2024 calendar window."""
    frame = _make_frame("2010-01-01", 4500)

    train, val, test = calendar_split(
        frame,
        train_end_date="2022-12-31",
        val_start_date="2023-01-01",
        val_end_date="2024-12-31",
        test_start_date="2025-01-01",
    )

    expected_val = frame[
        (frame["date"] >= "2023-01-01") & (frame["date"] <= "2024-12-31")
    ]
    assert val["date"].min() == expected_val["date"].min()
    assert val["date"].max() == expected_val["date"].max()
    assert len(val) == len(expected_val)
    assert len(train) == len(frame[frame["date"] <= "2022-12-31"])


def test_calendar_split_purges_label_horizon() -> None:
    """Unique trading dates near split boundaries are purged to prevent label leakage."""
    frame = _make_frame("2010-01-01", 4500)

    train, val, test = calendar_split(
        frame,
        train_end_date="2022-12-31",
        val_start_date="2023-01-01",
        val_end_date="2024-12-31",
        test_start_date="2025-01-01",
        purge_rows=5,
    )

    unpurged_train = frame[frame["date"] <= "2022-12-31"]
    unpurged_val = frame[
        (frame["date"] >= "2023-01-01") & (frame["date"] <= "2024-12-31")
    ]
    assert len(train) == len(unpurged_train) - 5
    assert len(val) == len(unpurged_val) - 5
    assert (val["date"].min() - train["date"].max()).days >= 1


def test_calendar_split_purges_entire_cross_section_on_panel() -> None:
    """On a multi-symbol panel, purge removes all rows for the last N dates."""
    dates = pd.date_range("2022-12-01", periods=20, freq="B")
    rows = []
    for date in dates:
        for symbol in ("AAPL", "MSFT", "GOOG"):
            rows.append({"date": date, "symbol": symbol, "close": 100.0, "label": 0})
    frame = pd.DataFrame(rows)

    train, val, test = calendar_split(
        frame,
        train_end_date="2022-12-16",
        val_start_date="2022-12-19",
        val_end_date="2022-12-23",
        test_start_date="2022-12-26",
        purge_rows=2,
    )

    # Train dates through 2022-12-16, minus last 2 unique dates.
    train_dates = sorted(train["date"].unique())
    assert len(train_dates) == len(sorted(frame[frame["date"] <= "2022-12-16"]["date"].unique())) - 2
    # Every remaining train date still has the full 3-symbol cross-section.
    assert train.groupby("date").size().min() == 3
    assert train.groupby("date").size().max() == 3


def test_calendar_split_rejects_overlapping_windows() -> None:
    """Overlapping calendar boundaries raise a clear configuration error."""
    frame = _make_frame("2020-01-01", 800)

    with pytest.raises(ValueError, match="train_end_date must be before val_start_date"):
        calendar_split(
            frame,
            train_end_date="2023-06-01",
            val_start_date="2023-01-01",
            val_end_date="2024-12-31",
            test_start_date="2025-01-01",
        )

    with pytest.raises(ValueError, match="val_end_date must be before test_start_date"):
        calendar_split(
            frame,
            train_end_date="2022-12-31",
            val_start_date="2023-01-01",
            val_end_date="2025-06-01",
            test_start_date="2025-01-01",
        )
