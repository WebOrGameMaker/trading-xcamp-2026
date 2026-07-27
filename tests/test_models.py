"""Tests for per-symbol model training and persistence."""

from pathlib import Path

import pandas as pd
import pytest

import src.models.registry as registry
from src.models.trainer import _aggregate_metrics, _diagnose_generalization_gap, time_based_split


def test_time_based_split_preserves_chronology() -> None:
    """A symbol's train, validation, and test periods do not overlap."""
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=20, freq="D"),
        "symbol": "AAPL",
    })

    train, val, test = time_based_split(frame, 0.5, 0.25, 0.25)

    assert len(train) == 10
    assert len(val) == 5
    assert len(test) == 5
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_time_based_split_purges_label_horizon() -> None:
    """Rows whose forward labels cross split boundaries are excluded."""
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=20, freq="D"),
        "symbol": "AAPL",
    })

    train, val, test = time_based_split(
        frame,
        0.5,
        0.25,
        0.25,
        purge_rows=2,
    )

    assert len(train) == 8
    assert len(val) == 3
    assert len(test) == 5
    assert (val["date"].min() - train["date"].max()).days == 3
    assert (test["date"].min() - val["date"].max()).days == 3


def _metrics(accuracy: float, roc_auc: float, support: int) -> dict:
    """Build a minimal per-split metrics dict for aggregation tests."""
    return {
        "accuracy": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
        "roc_auc": roc_auc,
        "brier_score": 0.25,
        "support": support,
    }


def test_aggregate_metrics_is_support_weighted() -> None:
    """Symbols with more test rows contribute proportionally more to the average."""
    per_symbol = {
        "AAPL": {"test": _metrics(accuracy=0.8, roc_auc=0.6, support=100)},
        "MSFT": {"test": _metrics(accuracy=0.4, roc_auc=0.5, support=300)},
    }

    aggregate = _aggregate_metrics(per_symbol, "test")

    assert aggregate["support"] == 400
    assert aggregate["accuracy"] == pytest.approx((0.8 * 100 + 0.4 * 300) / 400)


def test_aggregate_metrics_handles_no_support() -> None:
    """An empty or all-zero-support split returns zeroed metrics, not an error."""
    aggregate = _aggregate_metrics({}, "test")

    assert aggregate["support"] == 0
    assert aggregate["accuracy"] == 0.0


def test_diagnose_generalization_gap_detects_overfitting() -> None:
    """A large train-over-test accuracy gap is flagged as overfitting."""
    train = _metrics(accuracy=0.85, roc_auc=0.9, support=1000)
    test = _metrics(accuracy=0.55, roc_auc=0.6, support=200)

    assert _diagnose_generalization_gap(train, test) == "overfitting"


def test_diagnose_generalization_gap_detects_no_signal() -> None:
    """Near-random AUC on both splits (this project's actual symptom) is flagged as such."""
    train = _metrics(accuracy=0.50, roc_auc=0.51, support=1000)
    test = _metrics(accuracy=0.49, roc_auc=0.51, support=200)

    assert _diagnose_generalization_gap(train, test) == "underfitting_or_no_signal"


def test_diagnose_generalization_gap_ok_when_close_and_skillful() -> None:
    """Close train/test accuracy with above-baseline AUC is reported as ok."""
    train = _metrics(accuracy=0.60, roc_auc=0.62, support=1000)
    test = _metrics(accuracy=0.58, roc_auc=0.60, support=200)

    assert _diagnose_generalization_gap(train, test) == "ok"


def test_registry_loads_models_by_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The latest manifest resolves each ticker to its own artifact."""
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    artifacts = [
        registry.save_model(
            pipeline={"ticker": symbol},
            symbol=symbol,
            model_type="random_forest",
            feature_columns=["return_5d"],
            train_rows=10,
            val_rows=3,
            test_rows=3,
            metrics={},
        )
        for symbol in ("AAPL", "MSFT")
    ]
    registry.save_model_manifest(
        artifacts,
        model_type="random_forest",
        run_id="test_run",
        metrics={},
        scope="per_symbol",
    )

    loaded = registry.load_latest_models()

    assert set(loaded) == {"AAPL", "MSFT"}
    assert loaded["AAPL"][0]["ticker"] == "AAPL"
    assert loaded["MSFT"][1].symbol == "MSFT"


def test_registry_loads_pooled_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pooled manifests expose a single global model via load_latest_model."""
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    artifact = registry.save_model(
        pipeline={"scope": "pooled"},
        symbol=registry.POOLED_SYMBOL,
        model_type="xgboost",
        feature_columns=["return_5d", "rsi_14"],
        train_rows=1000,
        val_rows=200,
        test_rows=200,
        metrics={"test": {"roc_auc": 0.55}},
    )
    registry.save_model_manifest(
        [artifact],
        model_type="xgboost",
        run_id="pooled_run",
        metrics={"test": {"roc_auc": 0.55}},
        scope="pooled",
    )

    pipeline, loaded = registry.load_latest_model()
    assert pipeline["scope"] == "pooled"
    assert loaded.symbol == "pooled"
    assert loaded.feature_columns == ["return_5d", "rsi_14"]
