"""Tests for per-symbol model training and persistence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.models.registry as registry
from src.models.cross_sectional import evaluate_cross_sectional
from src.models.trainer import (
    _aggregate_metrics,
    _build_regressor,
    _diagnose_generalization_gap,
    _prepare_xy,
    time_based_split,
)


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


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "random_forest", "catboost"])
def test_build_regressor_fits_and_predicts(model_type: str) -> None:
    """Each tree regressor builds, fits, and returns continuous scores."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] * 0.1 + rng.normal(scale=0.01, size=80)
    model = _build_regressor(model_type, {"n_estimators": 10, "max_depth": 3, "random_state": 0})
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


def test_prepare_xy_regression_uses_forward_return() -> None:
    """Regression task pulls continuous forward returns, not binary labels."""
    frame = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "label": [0, 1, 0],
        "forward_return_5d": [0.01, -0.02, 0.03],
    })
    X, y = _prepare_xy(frame, ["f1"], task="regression", target_col="forward_return_5d")
    assert X.shape == (3, 1)
    assert list(y) == pytest.approx([0.01, -0.02, 0.03])


def test_resolve_feature_columns_override_preserves_order() -> None:
    """Explicit feature_columns override is validated and de-duplicated."""
    from src.models.trainer import _resolve_feature_columns

    frame = pd.DataFrame({
        "return_1d": [0.1],
        "rsi_14": [50.0],
        "atr_pct": [0.02],
        "label": [1],
        "forward_return_5d": [0.01],
        "close": [100.0],
    })
    cols = _resolve_feature_columns(frame, ["rsi_14", "return_1d", "rsi_14"])
    assert cols == ["rsi_14", "return_1d"]


def test_resolve_feature_columns_missing_raises() -> None:
    from src.models.trainer import _resolve_feature_columns

    frame = pd.DataFrame({"return_1d": [0.1], "close": [100.0]})
    with pytest.raises(ValueError, match="missing from dataset"):
        _resolve_feature_columns(frame, ["not_a_feature"])


def test_prepare_xy_classification_uses_label() -> None:
    """Classification task still uses the binary label column."""
    frame = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "label": [0, 1, 0],
        "forward_return_5d": [0.01, -0.02, 0.03],
    })
    X, y = _prepare_xy(frame, ["f1"], task="classification")
    assert list(y) == [0, 1, 0]


def test_ranking_on_continuous_scores_via_cross_sectional() -> None:
    """Predicted returns as the score column drive IC / hit-rate evaluation."""
    rows = []
    for day in (0, 1):
        for i, symbol in enumerate(list("ABCDEFGHIJ")):
            rows.append({
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                "symbol": symbol,
                "probability": float(i),
                "forward_return_5d": float(i) / 100.0,
                "label": int(i >= 8),
            })
    result = evaluate_cross_sectional(pd.DataFrame(rows))
    assert result["ic_overall"] > 0.9
    assert result["top_decile_hit_rate"] > 0.5


def test_diagnose_generalization_gap_regression_overfitting() -> None:
    """Large train-over-test R² gap is flagged for regression task."""
    train = {"r2": 0.20, "support": 1000}
    test = {"r2": 0.01, "support": 200}
    assert _diagnose_generalization_gap(train, test, task="regression") == "overfitting"
