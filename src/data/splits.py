"""Calendar-based train/validation/test splits for time-series datasets."""

from __future__ import annotations

import pandas as pd


def _purge_tail_dates(df: pd.DataFrame, purge_days: int) -> pd.DataFrame:
    """Drop rows belonging to the last ``purge_days`` unique trading dates.

    For a pooled multi-symbol panel this removes entire cross-sections near a
    split boundary so forward labels cannot leak into the next period. For a
    single-symbol series (one row per date) this is equivalent to dropping the
    last ``purge_days`` rows.

    Args:
        df: Split frame with a ``date`` column.
        purge_days: Number of unique trading dates to remove from the tail.

    Returns:
        Filtered DataFrame with the tail dates removed.
    """
    if purge_days <= 0 or df.empty:
        return df.reset_index(drop=True)

    dates = pd.Series(sorted(df["date"].unique()))
    if len(dates) <= purge_days:
        return df.iloc[0:0].reset_index(drop=True)

    cutoff = dates.iloc[-purge_days]
    return df[df["date"] < cutoff].reset_index(drop=True)


def calendar_split(
    df: pd.DataFrame,
    train_end_date: str,
    val_start_date: str,
    val_end_date: str,
    test_start_date: str,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into non-overlapping calendar train, val, and test pools.

    Training rows are on or before ``train_end_date``. Validation rows fall
    between ``val_start_date`` and ``val_end_date`` (inclusive). Test rows are
    on or after ``test_start_date`` (out-of-sample holdout).

    When ``purge_rows`` > 0, the last ``purge_rows`` *unique trading dates*
    are removed from the train and validation tails so forward-looking labels
    cannot cross temporal boundaries. (The parameter name is retained for
    compatibility; the unit is trading days, not panel rows.)

    Args:
        df: Feature dataset with a ``date`` column.
        train_end_date: Last inclusive date for the training pool.
        val_start_date: First inclusive date for the validation pool.
        val_end_date: Last inclusive date for the validation pool.
        test_start_date: First inclusive date for the test pool.
        purge_rows: Unique trading dates removed from the tail of train and
            val to prevent forward labels from crossing a temporal boundary.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    if purge_rows < 0:
        raise ValueError("purge_rows cannot be negative")

    train_end = pd.Timestamp(train_end_date)
    val_start = pd.Timestamp(val_start_date)
    val_end = pd.Timestamp(val_end_date)
    test_start = pd.Timestamp(test_start_date)

    if train_end >= val_start:
        raise ValueError("train_end_date must be before val_start_date")
    if val_end >= test_start:
        raise ValueError("val_end_date must be before test_start_date")
    if val_start > val_end:
        raise ValueError("val_start_date must be on or before val_end_date")

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "symbol"] if "symbol" in frame.columns else ["date"])
    frame = frame.reset_index(drop=True)

    train_df = frame[frame["date"] <= train_end].reset_index(drop=True)
    val_df = frame[
        (frame["date"] >= val_start) & (frame["date"] <= val_end)
    ].reset_index(drop=True)
    test_df = frame[frame["date"] >= test_start].reset_index(drop=True)

    if purge_rows > 0:
        train_df = _purge_tail_dates(train_df, purge_rows)
        val_df = _purge_tail_dates(val_df, purge_rows)

    return train_df, val_df, test_df
