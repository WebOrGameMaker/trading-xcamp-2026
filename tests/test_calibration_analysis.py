"""Tests for calibration trading analysis helpers."""

from __future__ import annotations

from src.models.calibration_analysis import _build_recommendation, _select_best_threshold


def test_select_best_threshold_prefers_sharpe_with_trade_floor() -> None:
    rows = [
        {"sharpe_ratio": 2.0, "total_trades": 2, "long_entry_threshold": 0.7},
        {"sharpe_ratio": 1.0, "total_trades": 50, "long_entry_threshold": 0.6},
        {"sharpe_ratio": 0.8, "total_trades": 40, "long_entry_threshold": 0.65},
    ]
    best = _select_best_threshold(rows)
    assert best is not None
    # High Sharpe with only 2 trades is ignored because of the trade floor.
    assert best["long_entry_threshold"] == 0.6


def test_recommendation_prefers_ranking_when_confidence_does_not_help() -> None:
    classification = {
        "test": {
            "probability": {"brier_score": 0.25, "ece": 0.12},
            "probability_isotonic": {"brier_score": 0.18, "ece": 0.05},
        }
    }
    strategy_a = {"sharpe_ratio": 1.2, "total_trades": 100}
    strategy_b = {
        "sharpe_ratio": 0.5,
        "total_trades": 80,
        "probability_column": "probability_isotonic",
    }
    rec = _build_recommendation(
        classification, strategy_a, strategy_b, "probability_isotonic"
    )
    assert rec["calibrated_probabilities_materially_better"] is True
    assert rec["confidence_filtering_improves_oos"] is False
    assert rec["production_mode"] == "long_short"


def test_recommendation_uses_confidence_when_test_sharpe_wins() -> None:
    classification = {
        "test": {
            "probability": {"brier_score": 0.25, "ece": 0.12},
            "probability_platt": {"brier_score": 0.20, "ece": 0.08},
        }
    }
    strategy_a = {"sharpe_ratio": 0.4, "total_trades": 100}
    strategy_b = {
        "sharpe_ratio": 0.9,
        "total_trades": 60,
        "probability_column": "probability_platt",
    }
    rec = _build_recommendation(
        classification, strategy_a, strategy_b, "probability_platt"
    )
    assert rec["confidence_filtering_improves_oos"] is True
    assert rec["production_mode"] == "long_short_confidence"
    assert rec["production_probability_column"] == "probability_platt"
