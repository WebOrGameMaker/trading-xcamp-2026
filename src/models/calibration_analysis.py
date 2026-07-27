"""Calibration quality + confidence-filtered trading evaluation.

Thresholds for Strategy B are selected on validation only; the test period is
used once for the final ranking-vs-confidence comparison.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.backtesting.engine import run_backtest_on_predictions
from src.backtesting.metrics import metrics_to_dict
from src.models.calibration import PROBABILITY_COLUMNS
from src.models.evaluator import evaluate_classifier
from src.utils.config import AppConfig, StrategyConfig
from src.utils.logging import get_logger
from src.utils.paths import LOG_DIR, PROCESSED_DATA_DIR
from src.visualization.loaders import latest_run_id

logger = get_logger(__name__)

DEFAULT_THRESHOLD_PAIRS: tuple[tuple[float, float], ...] = (
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
)

# Prefer this many trades on the val window before trusting a Sharpe maximum.
_MIN_VAL_TRADES = 10


def _load_predictions(split: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found for split={split!r}: {path}. "
            "Run 'python main.py train' first."
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _available_probability_columns(predictions: pd.DataFrame) -> list[str]:
    return [col for col in PROBABILITY_COLUMNS if col in predictions.columns]


def _classification_for_column(
    predictions: pd.DataFrame,
    proba_col: str,
) -> dict[str, Any]:
    y_true = predictions["label"].to_numpy(dtype=int)
    y_proba = predictions[proba_col].to_numpy(dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "brier_score": metrics.brier_score,
        "ece": metrics.ece,
        "reliability_bins": metrics.reliability_bins,
        "calibration_bins": metrics.calibration_bins,
        "support": metrics.support,
    }


def _strategy_a(config: AppConfig, proba_col: str) -> StrategyConfig:
    return replace(
        config.strategy,
        mode="long_short",
        probability_column=proba_col,
    )


def _strategy_b(
    config: AppConfig,
    proba_col: str,
    long_threshold: float,
    short_threshold: float,
) -> StrategyConfig:
    return replace(
        config.strategy,
        mode="long_short_confidence",
        probability_column=proba_col,
        long_entry_threshold=long_threshold,
        short_entry_threshold=short_threshold,
    )


def _metrics_summary(metrics: Any) -> dict[str, float | int]:
    payload = metrics_to_dict(metrics)
    return {
        "total_trades": int(payload["total_trades"]),
        "win_rate": float(payload["win_rate"]),
        "annualized_return": float(payload["annualized_return"]),
        "sharpe_ratio": float(payload["sharpe_ratio"]),
        "max_drawdown": float(payload["max_drawdown"]),
        "profit_factor": float(payload["profit_factor"]),
        "turnover": float(payload.get("turnover", 0.0)),
        "total_return": float(payload["total_return"]),
    }


def _select_best_threshold(
    sweep_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the val threshold pair with the highest Sharpe (trade-count floor)."""
    if not sweep_rows:
        return None
    eligible = [row for row in sweep_rows if int(row["total_trades"]) >= _MIN_VAL_TRADES]
    pool = eligible or sweep_rows
    return max(pool, key=lambda row: float(row["sharpe_ratio"]))


def _build_recommendation(
    classification: dict[str, Any],
    strategy_a_test: dict[str, Any],
    strategy_b_test: dict[str, Any] | None,
    best_calibrator: str,
) -> dict[str, Any]:
    """Encode the production decision from test-period evidence."""
    raw_test = classification.get("test", {}).get("probability", {})
    best_test = classification.get("test", {}).get(best_calibrator, {})
    raw_brier = float(raw_test.get("brier_score", 1.0))
    best_brier = float(best_test.get("brier_score", raw_brier))
    raw_ece = float(raw_test.get("ece", 1.0))
    best_ece = float(best_test.get("ece", raw_ece))

    calibration_helps = (best_brier < raw_brier - 0.005) or (best_ece < raw_ece - 0.01)

    a_sharpe = float(strategy_a_test.get("sharpe_ratio", 0.0))
    b_sharpe = float(strategy_b_test.get("sharpe_ratio", float("-inf"))) if strategy_b_test else float("-inf")
    b_trades = int(strategy_b_test.get("total_trades", 0)) if strategy_b_test else 0
    confidence_helps = (
        strategy_b_test is not None
        and b_trades >= _MIN_VAL_TRADES
        and b_sharpe > a_sharpe
    )

    if confidence_helps:
        production_mode = "long_short_confidence"
        production_probability_column = strategy_b_test.get(
            "probability_column", best_calibrator
        )
        rationale = (
            "Val-selected confidence thresholds beat pure ranking on test Sharpe "
            "with a meaningful trade count; use calibrated confidence gating."
        )
    else:
        production_mode = "long_short"
        production_probability_column = "probability"
        rationale = (
            "Confidence filtering does not improve out-of-sample Sharpe versus "
            "pure ranking (or trade count is too thin). Continue with ranking only."
        )

    return {
        "calibrated_probabilities_materially_better": calibration_helps,
        "confidence_filtering_improves_oos": confidence_helps,
        "production_mode": production_mode,
        "production_probability_column": production_probability_column,
        "best_calibrator_by_val": best_calibrator,
        "test_delta_brier": raw_brier - best_brier,
        "test_delta_ece": raw_ece - best_ece,
        "strategy_a_test_sharpe": a_sharpe,
        "strategy_b_test_sharpe": None if strategy_b_test is None else b_sharpe,
        "rationale": rationale,
    }


def run_calibration_analysis(
    config: AppConfig,
    *,
    threshold_pairs: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLD_PAIRS,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare raw/Platt/isotonic probs and ranking vs confidence-filtered books.

    Args:
        config: Application configuration.
        threshold_pairs: Strategy B (long, short) thresholds to sweep on val.
        run_id: Optional run id for artifact naming; defaults to latest train run.

    Returns:
        Full report dictionary (also written under ``logs/``).
    """
    resolved_run_id = run_id or latest_run_id() or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    pred_val = _load_predictions("val")
    pred_test = _load_predictions("test")
    proba_cols = _available_probability_columns(pred_val)
    if "probability" not in proba_cols:
        raise ValueError("predictions_val.parquet is missing the raw 'probability' column")

    classification: dict[str, Any] = {"val": {}, "test": {}}
    for split_name, frame in (("val", pred_val), ("test", pred_test)):
        for col in proba_cols:
            classification[split_name][col] = _classification_for_column(frame, col)

    # Best calibrator by validation Brier (tie-break ECE); raw is fallback only.
    calibrated_cols = [c for c in ("probability_platt", "probability_isotonic") if c in proba_cols]
    if calibrated_cols:
        best_calibrator = min(
            calibrated_cols,
            key=lambda col: (
                classification["val"][col]["brier_score"],
                classification["val"][col]["ece"],
            ),
        )
    else:
        best_calibrator = "probability"

    # Strategy A: pure ranking on test (monotone calibration preserves ranks, so
    # raw probability is sufficient and keeps the comparison clean).
    strategy_a = _strategy_a(config, "probability")
    strategy_a_test_metrics = _metrics_summary(
        run_backtest_on_predictions(config, pred_test, strategy=strategy_a, persist=False)
    )
    strategy_a_test_metrics["probability_column"] = "probability"
    strategy_a_test_metrics["mode"] = "long_short"

    # Strategy B: sweep thresholds on VAL for each probability column.
    val_sweeps: dict[str, list[dict[str, Any]]] = {}
    selected_thresholds: dict[str, dict[str, Any]] = {}
    for col in proba_cols:
        rows: list[dict[str, Any]] = []
        for long_thr, short_thr in threshold_pairs:
            strategy = _strategy_b(config, col, long_thr, short_thr)
            metrics = _metrics_summary(
                run_backtest_on_predictions(config, pred_val, strategy=strategy, persist=False)
            )
            rows.append({
                **metrics,
                "probability_column": col,
                "long_entry_threshold": long_thr,
                "short_entry_threshold": short_thr,
                "mode": "long_short_confidence",
            })
        val_sweeps[col] = rows
        best = _select_best_threshold(rows)
        if best is not None:
            selected_thresholds[col] = best
            logger.info(
                "Val-selected thresholds for %s: long>=%.2f / short<=%.2f "
                "(sharpe=%.3f, trades=%d)",
                col,
                best["long_entry_threshold"],
                best["short_entry_threshold"],
                best["sharpe_ratio"],
                best["total_trades"],
            )

    # Single test evaluation per probability column using frozen val thresholds.
    strategy_b_test: dict[str, dict[str, Any]] = {}
    for col, chosen in selected_thresholds.items():
        strategy = _strategy_b(
            config,
            col,
            float(chosen["long_entry_threshold"]),
            float(chosen["short_entry_threshold"]),
        )
        metrics = _metrics_summary(
            run_backtest_on_predictions(config, pred_test, strategy=strategy, persist=False)
        )
        strategy_b_test[col] = {
            **metrics,
            "probability_column": col,
            "long_entry_threshold": chosen["long_entry_threshold"],
            "short_entry_threshold": chosen["short_entry_threshold"],
            "mode": "long_short_confidence",
            "selected_on": "val",
        }

    # Recommendation compares Strategy A vs the val-best calibrator's Strategy B
    # (thresholds frozen on val). Do not re-pick the calibrator on test.
    preferred_b = strategy_b_test.get(best_calibrator) or strategy_b_test.get("probability")

    recommendation = _build_recommendation(
        classification,
        strategy_a_test_metrics,
        preferred_b,
        best_calibrator,
    )

    report: dict[str, Any] = {
        "run_id": resolved_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold_pairs": [list(pair) for pair in threshold_pairs],
        "probability_columns": proba_cols,
        "best_calibrator_by_val_brier": best_calibrator,
        "classification": {
            split: {
                col: {
                    k: v
                    for k, v in metrics.items()
                    if k not in {"reliability_bins", "calibration_bins"}
                }
                for col, metrics in cols.items()
            }
            for split, cols in classification.items()
        },
        "classification_reliability_bins": {
            split: {col: metrics["reliability_bins"] for col, metrics in cols.items()}
            for split, cols in classification.items()
        },
        "strategy_a_test": strategy_a_test_metrics,
        "strategy_b_val_sweep": val_sweeps,
        "strategy_b_selected_thresholds": selected_thresholds,
        "strategy_b_test": strategy_b_test,
        "recommendation": recommendation,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / f"calibration_trading_report_{resolved_run_id}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote calibration trading report to %s", report_path)

    # Persist a short latest pointer for dashboards / visualize.
    latest_path = LOG_DIR / "calibration_trading_report_latest.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "Recommendation — calibrated_better=%s, confidence_helps=%s, "
        "production_mode=%s (%s)",
        recommendation["calibrated_probabilities_materially_better"],
        recommendation["confidence_filtering_improves_oos"],
        recommendation["production_mode"],
        recommendation["rationale"],
    )
    return report
