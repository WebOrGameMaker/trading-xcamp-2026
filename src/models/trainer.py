"""Model training pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import calendar_split
from src.data.storage import load_processed_dataset, save_processed_dataset
from src.features.engineer import get_feature_columns
from src.features.labels import forward_return_column
from src.models.calibration import (
    PROBABILITY_COLUMNS,
    fit_calibrators,
    save_calibrators,
)
from src.models.cross_sectional import evaluate_cross_sectional
from src.models.evaluator import (
    compute_shap_importance,
    evaluate_classifier,
    extract_feature_importance,
    save_evaluation_report,
)
from src.models.registry import POOLED_SYMBOL, save_model, save_model_manifest
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import LOG_DIR, MODEL_DIR

logger = get_logger(__name__)


def _build_classifier(model_type: str, hyperparams: dict[str, Any]) -> Any:
    """Instantiate a classifier by type name.

    Args:
        model_type: One of xgboost, lightgbm, random_forest.
        hyperparams: Hyperparameter dictionary for the model.

    Returns:
        Unfitted sklearn-compatible classifier.
    """
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=hyperparams.get("random_state", 42),
            n_estimators=hyperparams.get("n_estimators", 200),
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.05),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            scale_pos_weight=hyperparams.get("scale_pos_weight", 1.0),
        )
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            random_state=hyperparams.get("random_state", 42),
            n_estimators=hyperparams.get("n_estimators", 200),
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.05),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            scale_pos_weight=hyperparams.get("scale_pos_weight", 1.0),
            verbose=-1,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=hyperparams.get("random_state", 42),
            n_estimators=hyperparams.get("n_estimators", 200),
            max_depth=hyperparams.get("max_depth", 10),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 5),
            class_weight=hyperparams.get("class_weight", "balanced"),
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically and optionally purge split boundaries.

    Args:
        df: Feature dataset sorted by date.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        purge_rows: Unique trading dates removed from train/val tails.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if purge_rows < 0:
        raise ValueError("purge_rows cannot be negative")

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.Series(sorted(df["date"].unique()))
    n = len(dates)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_dates = dates.iloc[:train_end_idx]
    val_dates = dates.iloc[train_end_idx:val_end_idx]
    test_dates = dates.iloc[val_end_idx:]

    if purge_rows > 0:
        if len(train_dates) > purge_rows:
            train_dates = train_dates.iloc[:-purge_rows]
        else:
            train_dates = train_dates.iloc[0:0]
        if len(val_dates) > purge_rows:
            val_dates = val_dates.iloc[:-purge_rows]
        else:
            val_dates = val_dates.iloc[0:0]

    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)
    return (
        df[df["date"].isin(train_set)].reset_index(drop=True),
        df[df["date"].isin(val_set)].reset_index(drop=True),
        df[df["date"].isin(test_set)].reset_index(drop=True),
    )


def _prepare_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and labels from DataFrame."""
    X = df[feature_columns].values
    y = df["label"].values.astype(int)
    return X, y


def _predict_proba_positive(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from pipeline."""
    proba = pipeline.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    """Convert classification metrics to manifest-friendly values."""
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "brier_score": metrics.brier_score,
        "ece": metrics.ece,
        "support": metrics.support,
        "calibration_bins": metrics.calibration_bins,
        "reliability_bins": metrics.reliability_bins,
    }


def _evaluate_probability_sources(
    y_true: np.ndarray,
    probability_map: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Evaluate classification metrics for each probability column.

    Hard labels use a 0.5 threshold on that probability source.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, proba in probability_map.items():
        y_pred = (np.asarray(proba) >= 0.5).astype(int)
        metrics = evaluate_classifier(y_true, y_pred, proba)
        results[name] = _metrics_dict(metrics)
    return results


def _diagnose_generalization_gap(
    train_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> str:
    """Classify the train-vs-test accuracy gap for a quick overfit signal."""
    if not train_metrics.get("support") or not test_metrics.get("support"):
        return "no_data"

    gap = float(train_metrics["accuracy"]) - float(test_metrics["accuracy"])
    if gap > 0.05:
        return "overfitting"
    if float(train_metrics["roc_auc"]) < 0.53 and float(test_metrics["roc_auc"]) < 0.53:
        return "underfitting_or_no_signal"
    return "ok"


def _assign_predicted_rank(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Add within-date predicted rank (1 = highest probability)."""
    ranked = pred_df.copy()
    ranked["predicted_rank"] = (
        ranked.groupby("date")["probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ranked


def _top_n_features(importance: dict[str, float], n: int = 20) -> dict[str, float]:
    """Return the top-N importance entries preserving sort order."""
    return dict(list(importance.items())[:n])


def _run_walk_forward(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    base_hp: dict[str, Any],
    horizon_days: int,
    n_folds: int = 4,
) -> list[dict[str, Any]]:
    """Expanding-window walk-forward evaluation (does not replace the main model)."""
    dates = pd.Series(sorted(dataset["date"].unique()))
    if len(dates) < n_folds + 1:
        logger.warning("Too few dates for walk-forward evaluation")
        return []

    fold_size = len(dates) // (n_folds + 1)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(1, n_folds + 1):
        train_end = dates.iloc[fold * fold_size]
        test_start_idx = fold * fold_size + horizon_days
        test_end_idx = min((fold + 1) * fold_size + horizon_days, len(dates) - 1)
        if test_start_idx >= len(dates) or test_start_idx > test_end_idx:
            continue

        test_start = dates.iloc[test_start_idx]
        test_end = dates.iloc[test_end_idx]
        train_df = dataset[dataset["date"] <= train_end]
        test_df = dataset[(dataset["date"] >= test_start) & (dataset["date"] <= test_end)]
        if len(train_df) < 100 or len(test_df) < 20:
            continue

        X_train, y_train = _prepare_xy(train_df, feature_columns)
        if len(np.unique(y_train)) < 2:
            continue

        hp = dict(base_hp)
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        hp["scale_pos_weight"] = n_neg / n_pos

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", _build_classifier(model_type, hp)),
        ])
        pipeline.fit(X_train, y_train)

        X_test, y_test = _prepare_xy(test_df, feature_columns)
        y_pred = pipeline.predict(X_test)
        y_proba = _predict_proba_positive(pipeline, X_test)
        metrics = evaluate_classifier(y_test, y_pred, y_proba)

        pred_df = test_df[["date", "symbol", "close", forward_return_column(horizon_days), "label"]].copy()
        pred_df["probability"] = y_proba
        pred_df["prediction"] = y_pred
        cs = evaluate_cross_sectional(pred_df)

        fold_reports.append({
            "fold": fold,
            "train_end": str(pd.Timestamp(train_end).date()),
            "test_start": str(pd.Timestamp(test_start).date()),
            "test_end": str(pd.Timestamp(test_end).date()),
            "metrics": _metrics_dict(metrics),
            "cross_sectional": cs,
        })
        logger.info(
            "Walk-forward fold %d — AUC: %.3f, IC: %.4f",
            fold,
            metrics.roc_auc,
            cs.get("ic_mean_daily", 0.0),
        )
    return fold_reports


def train_model(config: AppConfig) -> None:
    """Train, evaluate, and persist a single pooled cross-sectional model.

    Args:
        config: Application configuration.
    """
    if config.model.scope != "pooled":
        raise ValueError(
            "This trainer supports model.scope='pooled' only. "
            "Per-symbol training has been replaced by the pooled cross-sectional model."
        )
    if config.model.include_ticker:
        logger.warning(
            "model.include_ticker=true is ignored; ticker identity is omitted from features"
        )

    dataset = load_processed_dataset("features")
    feature_columns = get_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("No feature columns found in dataset")

    horizon = config.labels.horizon_days
    ret_col = forward_return_column(horizon)
    if ret_col not in dataset.columns:
        raise ValueError(f"Dataset missing forward-return column {ret_col}")

    model_type = config.model.type
    hp = dict(config.model.hyperparams.get(model_type, {}))
    hp["random_state"] = config.model.random_state
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    train_df, val_df, test_df = calendar_split(
        dataset,
        train_end_date=config.data.train_end_date,
        val_start_date=config.data.val_start_date,
        val_end_date=config.data.val_end_date,
        test_start_date=config.data.test_start_date,
        purge_rows=horizon,
    )
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("One or more calendar splits are empty")

    X_train, y_train = _prepare_xy(train_df, feature_columns)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels have only one class")

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    hp["scale_pos_weight"] = n_neg / n_pos

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", _build_classifier(model_type, hp)),
    ])
    logger.info(
        "Training pooled %s model on %d samples, %d features, %d symbols "
        "(pos=%.1f%%, scale_pos_weight=%.2f)",
        model_type,
        len(X_train),
        len(feature_columns),
        train_df["symbol"].nunique(),
        100.0 * n_pos / (n_pos + n_neg),
        hp["scale_pos_weight"],
    )
    pipeline.fit(X_train, y_train)

    # Fit calibrators on validation raw scores only (never on test).
    X_val, y_val = _prepare_xy(val_df, feature_columns)
    raw_val_proba = _predict_proba_positive(pipeline, X_val)
    calibrators = fit_calibrators(y_val, raw_val_proba)
    calibrator_path = MODEL_DIR / "pooled" / f"calibrators_{run_id}.joblib"
    save_calibrators(calibrators, calibrator_path)

    split_metrics: dict[str, Any] = {}
    calibration_comparison: dict[str, Any] = {"run_id": run_id, "splits": {}}
    predictions: dict[str, pd.DataFrame] = {}
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        X_split, y_split = _prepare_xy(split_df, feature_columns)
        y_proba_raw = _predict_proba_positive(pipeline, X_split)
        calibrated = calibrators.transform(y_proba_raw)
        probability_map = {
            "probability": y_proba_raw,
            "probability_platt": calibrated["platt"],
            "probability_isotonic": calibrated["isotonic"],
        }
        y_pred = (y_proba_raw >= 0.5).astype(int)
        metrics = evaluate_classifier(y_split, y_pred, y_proba_raw)
        save_evaluation_report(
            metrics,
            f"pooled_{split_name}",
            run_id,
            extra={"scope": "pooled", "data_split": split_name},
        )
        split_metrics[split_name] = _metrics_dict(metrics)
        calibration_comparison["splits"][split_name] = _evaluate_probability_sources(
            y_split, probability_map
        )

        pred_df = split_df[["date", "symbol", "close", ret_col, "label"]].copy()
        pred_df["probability"] = y_proba_raw
        pred_df["probability_platt"] = calibrated["platt"]
        pred_df["probability_isotonic"] = calibrated["isotonic"]
        pred_df["prediction"] = y_pred
        pred_df = _assign_predicted_rank(pred_df)
        pred_df = pred_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        predictions[split_name] = pred_df
        save_processed_dataset(pred_df, name=f"predictions_{split_name}")

        cs = evaluate_cross_sectional(pred_df)
        split_metrics[split_name]["cross_sectional"] = cs
        logger.info(
            "%s CS metrics — IC daily: %.4f, top-decile hit: %.3f",
            split_name,
            cs.get("ic_mean_daily", 0.0),
            cs.get("top_decile_hit_rate", 0.0),
        )
        for col in PROBABILITY_COLUMNS:
            col_metrics = calibration_comparison["splits"][split_name][col]
            logger.info(
                "%s %s — brier: %.4f, ece: %.4f, roc_auc: %.4f, pr_auc: %.4f",
                split_name,
                col,
                col_metrics["brier_score"],
                col_metrics["ece"],
                col_metrics["roc_auc"],
                col_metrics["pr_auc"],
            )

    # Prefer the calibrator with lower validation Brier (tie-break: lower ECE).
    val_compare = calibration_comparison["splits"]["val"]
    best_calibrator = min(
        ("probability_platt", "probability_isotonic"),
        key=lambda col: (
            val_compare[col]["brier_score"],
            val_compare[col]["ece"],
        ),
    )
    calibration_comparison["best_by_val_brier"] = best_calibrator
    calibration_comparison["calibrator_path"] = str(calibrator_path)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    calib_path = LOG_DIR / f"calibration_comparison_{run_id}.json"
    with calib_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration_comparison, handle, indent=2)
    logger.info("Wrote calibration comparison to %s", calib_path)

    feature_importance = extract_feature_importance(pipeline, feature_columns)
    shap_importance = compute_shap_importance(
        pipeline,
        val_df[feature_columns],
        feature_columns,
        random_state=config.model.random_state,
    )

    generalization_gap = _diagnose_generalization_gap(
        split_metrics["train"],
        split_metrics["test"],
    )

    all_metrics: dict[str, Any] = {
        "train": split_metrics["train"],
        "val": split_metrics["val"],
        "test": split_metrics["test"],
        "generalization_gap": generalization_gap,
        "feature_importance": _top_n_features(feature_importance, 20),
        "shap_importance": _top_n_features(shap_importance, 20),
        "calibration": {
            "best_by_val_brier": best_calibrator,
            "calibrator_path": str(calibrator_path),
            "comparison": calibration_comparison["splits"],
        },
    }

    if config.model.walk_forward:
        walk_reports = _run_walk_forward(
            dataset,
            feature_columns,
            model_type,
            {**dict(config.model.hyperparams.get(model_type, {})), "random_state": config.model.random_state},
            horizon_days=horizon,
        )
        all_metrics["walk_forward"] = walk_reports
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        walk_path = LOG_DIR / f"walk_forward_{run_id}.json"
        with walk_path.open("w", encoding="utf-8") as handle:
            json.dump(walk_reports, handle, indent=2)
        logger.info("Wrote walk-forward report to %s", walk_path)

    artifact = save_model(
        pipeline=pipeline,
        symbol=POOLED_SYMBOL,
        model_type=model_type,
        feature_columns=feature_columns,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        metrics=all_metrics,
    )
    manifest_path = save_model_manifest(
        artifacts=[artifact],
        model_type=model_type,
        run_id=run_id,
        metrics=all_metrics,
        scope="pooled",
    )
    logger.info(
        "Pooled accuracy — train: %.3f, val: %.3f, test: %.3f | "
        "AUC — train: %.3f, val: %.3f, test: %.3f | diagnosis: %s",
        split_metrics["train"]["accuracy"],
        split_metrics["val"]["accuracy"],
        split_metrics["test"]["accuracy"],
        split_metrics["train"]["roc_auc"],
        split_metrics["val"]["roc_auc"],
        split_metrics["test"]["roc_auc"],
        generalization_gap,
    )
    logger.info("Saved pooled model and manifest to %s", manifest_path)


# Retained for unit tests that still reference per-symbol aggregation helpers.
def _aggregate_metrics(
    per_symbol: dict[str, dict[str, Any]],
    split_name: str,
) -> dict[str, float | int]:
    """Compute support-weighted classification metrics across symbols."""
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier_score")
    split_metrics = [
        values[split_name]
        for values in per_symbol.values()
        if split_name in values and values[split_name]["support"] > 0
    ]
    total_support = sum(int(values["support"]) for values in split_metrics)
    if total_support == 0:
        return {**dict.fromkeys(metric_names, 0.0), "support": 0}

    aggregate: dict[str, float | int] = {
        name: sum(
            float(values[name]) * int(values["support"])
            for values in split_metrics
        ) / total_support
        for name in metric_names
    }
    aggregate["support"] = total_support
    return aggregate


def _aggregate_feature_importance(
    per_symbol: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Average feature importance values across trained symbols."""
    importance_sets = [
        values["feature_importance"]
        for values in per_symbol.values()
        if values.get("feature_importance")
    ]
    if not importance_sets:
        return {}

    features = set().union(*(values.keys() for values in importance_sets))
    averaged = {
        feature: float(np.mean([values.get(feature, 0.0) for values in importance_sets]))
        for feature in features
    }
    return dict(sorted(averaged.items(), key=lambda item: item[1], reverse=True))
