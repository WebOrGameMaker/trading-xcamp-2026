"""Tests for cross-sectional ranking metrics."""

from __future__ import annotations

import pandas as pd

from src.models.cross_sectional import (
    evaluate_cross_sectional,
    information_coefficient,
    top_decile_hit_rate,
)


def _panel() -> pd.DataFrame:
    """Build a small prediction panel with known ranks."""
    rows = []
    for day, offset in (("2024-01-02", 0.0), ("2024-01-03", 0.01)):
        for i, symbol in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
            rows.append({
                "date": pd.Timestamp(day),
                "symbol": symbol,
                "probability": (i + 1) / 10 + offset,
                "forward_return_5d": (i + 1) / 100,
                "label": int(i >= 8),
            })
    return pd.DataFrame(rows)


def test_information_coefficient_positive_when_aligned() -> None:
    """Scores aligned with returns produce a positive Spearman IC."""
    ic = information_coefficient(_panel())
    assert ic["ic_overall"] > 0.9
    assert ic["ic_mean_daily"] > 0.9


def test_top_decile_hit_rate_counts_positive_labels() -> None:
    """Top 10% predictions that are labeled positive contribute to hit rate."""
    rate = top_decile_hit_rate(_panel(), top_frac=0.10)
    # Top name each day is J with label 1.
    assert rate == 1.0


def test_evaluate_cross_sectional_bundle() -> None:
    """Full CS evaluation returns IC, hit rate, and decile returns."""
    result = evaluate_cross_sectional(_panel())
    assert "ic_overall" in result
    assert "top_decile_hit_rate" in result
    assert len(result["mean_return_by_prediction_decile"]) >= 2
