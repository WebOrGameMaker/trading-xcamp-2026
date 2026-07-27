"""Rebalance cadence utilities for scheduling weekly portfolio turnover."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def weekly_rebalance_dates(
    dates: Iterable[pd.Timestamp],
    freq: str = "W-FRI",
) -> pd.DatetimeIndex:
    """Pick the last available trading date in each calendar week.

    Given the set of dates on which predictions/prices actually exist, returns
    one date per calendar week (the latest trading day on or before the week
    boundary), so the strategy rebalances weekly instead of daily.

    Args:
        dates: Iterable of available trading dates.
        freq: Pandas offset alias marking the end of a week (default Friday).

    Returns:
        Sorted DatetimeIndex of one rebalance date per calendar week.
    """
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))).unique()).sort_values()
    if index.empty:
        return index

    series = pd.Series(index, index=index)
    weekly_last = series.resample(freq).last().dropna()
    return pd.DatetimeIndex(sorted(weekly_last.values))
