"""Experiment 2 orchestrator — feature family ablation and selection (H2).

Trains pooled XGBoost on continuous 5-day forward returns for Stage A family
arms and Stage B importance-pruned subsets, runs each arm through the identical
weekly top-10 / bottom-10 long-short trading pipeline, and writes comparison
tables + figures under results/experiment_2/. Does not touch calibration or
confidence-gated trading.

Usage:
    python scripts/run_experiment2.py
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.backtesting.engine import run_backtest_on_predictions
from src.backtesting.metrics import metrics_to_dict
from src.data.downloader import download_universe
from src.features.families import (
    STAGE_A_ARMS,
    STAGE_B_ARMS,
    resolve_feature_arm,
    resolve_prune_arm,
)
from src.features.pipeline import build_feature_dataset
from src.models.trainer import train_model
from src.utils.config import StrategyConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, ensure_directories
from src.visualization.experiment2_plots import generate_experiment2_figures

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_2"
FROZEN_MODEL = "xgboost"
SPLITS = ("train", "val", "test")
BACKTEST_SPLITS = ("val", "test")
METRIC_COLS = (
    "rmse",
    "mae",
    "r2",
    "roc_auc",
    "pr_auc",
    "support",
)
TRADING_METRIC_COLS = (
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "turnover",
    "total_return",
    "total_trades",
)


@dataclass
class ArmRunResult:
    """Everything captured from one feature-arm training run."""

    arm: str
    stage: str
    feature_columns: list[str]
    run_id: str
    manifest: dict[str, Any]
    out_dir: Path
    trading: dict[str, dict[str, float]]


def _read_latest_manifest() -> dict[str, Any]:
    path = MODEL_DIR / "latest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _frozen_long_short_strategy(config) -> StrategyConfig:
    """Copy strategy settings but force pure ranking (no confidence gating)."""
    base = config.strategy
    return StrategyConfig(
        mode="long_short",
        long_positions=base.long_positions,
        short_positions=base.short_positions,
        long_gross_exposure=base.long_gross_exposure,
        short_gross_exposure=base.short_gross_exposure,
        max_weight_per_symbol=base.max_weight_per_symbol,
        rebalance_frequency=base.rebalance_frequency,
        entry_threshold=base.entry_threshold,
        max_positions=base.max_positions,
        long_entry_threshold=base.long_entry_threshold,
        short_entry_threshold=base.short_entry_threshold,
        probability_column=base.probability_column,
    )


def _run_trading_pipeline(
    config,
    out_dir: Path,
    strategy: StrategyConfig,
) -> dict[str, dict[str, float]]:
    """Run the identical long_short backtest on archived val/test predictions."""
    trading: dict[str, dict[str, float]] = {}
    for split in BACKTEST_SPLITS:
        pred_path = out_dir / f"predictions_{split}.parquet"
        if not pred_path.exists():
            logger.warning("Missing %s — skipping %s backtest", pred_path, split)
            continue
        predictions = pd.read_parquet(pred_path)
        metrics = run_backtest_on_predictions(
            config,
            predictions,
            strategy=strategy,
            persist=False,
        )
        trading[split] = {
            key: float(metrics_to_dict(metrics)[key])
            for key in TRADING_METRIC_COLS
        }
        logger.info(
            "%s/%s backtest — Sharpe: %.3f, ann: %.2f%%, max DD: %.2f%%",
            out_dir.name,
            split,
            trading[split]["sharpe_ratio"],
            trading[split]["annualized_return"] * 100,
            trading[split]["max_drawdown"] * 100,
        )
    return trading


def _run_one_arm(
    config,
    arm: str,
    stage: str,
    feature_columns: list[str],
    strategy: StrategyConfig,
) -> ArmRunResult:
    """Train XGBoost on one feature subset, archive artifacts, and backtest."""
    logger.info(
        "=== Arm %s (stage %s) — %d features ===",
        arm,
        stage,
        len(feature_columns),
    )
    config.model.type = FROZEN_MODEL
    config.model.task = "regression"
    train_model(config, feature_columns=feature_columns)

    manifest = _read_latest_manifest()
    artifact = manifest["artifacts"][0]
    run_id = manifest["run_id"]
    assert artifact["model_type"] == FROZEN_MODEL, (
        f"Manifest model_type mismatch: expected {FROZEN_MODEL}, "
        f"got {artifact['model_type']}"
    )

    out_dir = RESULTS_DIR / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    with (out_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump({"arm": arm, "stage": stage, "feature_columns": feature_columns}, handle, indent=2)

    trading = _run_trading_pipeline(config, out_dir, strategy)
    with (out_dir / "trading_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(trading, handle, indent=2)

    logger.info("Archived arm %s -> %s", arm, out_dir)
    return ArmRunResult(
        arm=arm,
        stage=stage,
        feature_columns=list(feature_columns),
        run_id=run_id,
        manifest=manifest,
        out_dir=out_dir,
        trading=trading,
    )


def _metrics_table(results: list[ArmRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        for split in SPLITS:
            split_metrics = metrics[split]
            row = {
                "arm": r.arm,
                "stage": r.stage,
                "split": split,
                "run_id": r.run_id,
                "n_features": len(r.feature_columns),
            }
            row.update({col: split_metrics.get(col) for col in METRIC_COLS})
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[["arm", "stage", "split", "run_id", "n_features", *METRIC_COLS]]


def _cross_sectional_table(results: list[ArmRunResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        cs = metrics[split].get("cross_sectional", {})
        ic_mean = cs.get("ic_mean_daily")
        ic_std = cs.get("ic_std_daily")
        ic_ir = None
        if ic_mean is not None and ic_std is not None and float(ic_std) > 0:
            ic_ir = float(ic_mean) / float(ic_std)
        rows.append({
            "arm": r.arm,
            "stage": r.stage,
            "split": split,
            "ic_overall": cs.get("ic_overall"),
            "ic_mean_daily": ic_mean,
            "ic_std_daily": ic_std,
            "ic_ir": ic_ir,
            "top_decile_hit_rate": cs.get("top_decile_hit_rate"),
        })
    return pd.DataFrame(rows)


def _returns_table(results: list[ArmRunResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        cs = metrics[split].get("cross_sectional", {}) or {}
        deciles = cs.get("mean_return_by_prediction_decile") or []
        if not deciles:
            continue
        ordered = sorted(deciles, key=lambda d: int(d["decile"]))
        bottom = float(ordered[0]["mean_forward_return"])
        top = float(ordered[-1]["mean_forward_return"])
        rows.append({
            "arm": r.arm,
            "stage": r.stage,
            "split": split,
            "top_decile_mean_return": top,
            "bottom_decile_mean_return": bottom,
            "top_minus_bottom": top - bottom,
        })
    return pd.DataFrame(rows)


def _trading_table(results: list[ArmRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for split, metrics in r.trading.items():
            row = {
                "arm": r.arm,
                "stage": r.stage,
                "split": split,
                "run_id": r.run_id,
                "n_features": len(r.feature_columns),
            }
            row.update({col: metrics.get(col) for col in TRADING_METRIC_COLS})
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["arm", "stage", "split", "run_id", "n_features", *TRADING_METRIC_COLS]
        )
    return pd.DataFrame(rows)[
        ["arm", "stage", "split", "run_id", "n_features", *TRADING_METRIC_COLS]
    ]


def _pick_winner(trading_df: pd.DataFrame, cs_df: pd.DataFrame) -> str:
    """Rank by test Sharpe, tie-break mean-daily IC, then top-decile hit rate."""
    test_trading = trading_df[trading_df["split"] == "test"].set_index("arm")
    cs = cs_df.set_index("arm")
    ranking = pd.DataFrame({
        "sharpe_ratio": test_trading["sharpe_ratio"],
        "ic_mean_daily": cs["ic_mean_daily"],
        "top_decile_hit_rate": cs["top_decile_hit_rate"],
    })
    ranking = ranking.sort_values(
        ["sharpe_ratio", "ic_mean_daily", "top_decile_hit_rate"], ascending=False
    )
    logger.info("Arm ranking (test Sharpe / IC / hit-rate):\n%s", ranking.to_string())
    return str(ranking.index[0])


def _pick_best_stage_b_by_val(trading_df: pd.DataFrame) -> str | None:
    """Select the best Stage B prune arm on validation Sharpe."""
    stage_b = trading_df[
        (trading_df["stage"] == "B") & (trading_df["split"] == "val")
    ]
    if stage_b.empty:
        return None
    ranked = stage_b.sort_values("sharpe_ratio", ascending=False)
    best = str(ranked.iloc[0]["arm"])
    logger.info(
        "Best Stage B arm by val Sharpe: %s (%.3f)",
        best,
        float(ranked.iloc[0]["sharpe_ratio"]),
    )
    return best


def _full_importance(full_result: ArmRunResult) -> dict[str, float]:
    metrics = full_result.manifest["artifacts"][0]["metrics"]
    importance = metrics.get("feature_importance") or {}
    return {str(k): float(v) for k, v in importance.items()}


def main() -> None:
    setup_logging(level="INFO")
    ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/default.yaml")
    config.model.type = FROZEN_MODEL
    config.model.task = "regression"
    strategy = _frozen_long_short_strategy(config)

    logger.info("Ensuring raw data is downloaded (cache-aware)...")
    download_universe(config, force=False)

    logger.info("Rebuilding feature dataset from current code...")
    build_feature_dataset(config)

    results: list[ArmRunResult] = []

    # Stage A — family ablation (includes Full baseline).
    for arm in STAGE_A_ARMS:
        feature_columns = resolve_feature_arm(arm)
        results.append(
            _run_one_arm(config, arm, stage="A", feature_columns=feature_columns, strategy=strategy)
        )

    full_result = next(r for r in results if r.arm == "full")
    importance = _full_importance(full_result)
    if not importance:
        raise RuntimeError("Full arm produced no feature_importance for Stage B pruning")

    with (RESULTS_DIR / "full_feature_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(importance, handle, indent=2)

    # Stage B — importance prune arms derived from Full only.
    for arm in STAGE_B_ARMS:
        feature_columns = resolve_prune_arm(arm, importance)
        results.append(
            _run_one_arm(config, arm, stage="B", feature_columns=feature_columns, strategy=strategy)
        )

    metrics_df = _metrics_table(results)
    cs_test = _cross_sectional_table(results, split="test")
    cs_val = _cross_sectional_table(results, split="val")
    cs_df = pd.concat([cs_val, cs_test], ignore_index=True)
    returns_df = _returns_table(results, split="test")
    trading_df = _trading_table(results)

    metrics_path = RESULTS_DIR / "metrics_by_arm_split.csv"
    cs_path = RESULTS_DIR / "cross_sectional_by_arm.csv"
    returns_path = RESULTS_DIR / "returns_by_arm.csv"
    trading_path = RESULTS_DIR / "trading_by_arm.csv"
    metrics_df.to_csv(metrics_path, index=False)
    cs_df.to_csv(cs_path, index=False)
    returns_df.to_csv(returns_path, index=False)
    trading_df.to_csv(trading_path, index=False)
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", cs_path)
    logger.info("Wrote %s", returns_path)
    logger.info("Wrote %s", trading_path)

    winner = _pick_winner(trading_df, cs_test)
    best_stage_b = _pick_best_stage_b_by_val(trading_df)
    logger.info("Winning arm (test Sharpe / IC / hit-rate): %s", winner)

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": "configs/default.yaml",
        "task": "regression",
        "target": f"forward_return_{config.labels.horizon_days}d",
        "frozen_model": FROZEN_MODEL,
        "winner": winner,
        "best_stage_b_by_val_sharpe": best_stage_b,
        "winner_rule": "test_sharpe -> ic_mean_daily -> top_decile_hit_rate",
        "strategy": {
            "mode": strategy.mode,
            "long_positions": strategy.long_positions,
            "short_positions": strategy.short_positions,
            "rebalance_frequency": strategy.rebalance_frequency,
        },
        "runs": [
            {
                "arm": r.arm,
                "stage": r.stage,
                "run_id": r.run_id,
                "n_features": len(r.feature_columns),
                "feature_columns": r.feature_columns,
                "train_rows": r.manifest["artifacts"][0]["train_rows"],
                "val_rows": r.manifest["artifacts"][0]["val_rows"],
                "test_rows": r.manifest["artifacts"][0]["test_rows"],
                "generalization_gap": r.manifest["artifacts"][0]["metrics"].get(
                    "generalization_gap"
                ),
                "trading": r.trading,
            }
            for r in results
        ],
        "data": {
            "universe_file": config.data.universe_file,
            "train_end_date": config.data.train_end_date,
            "val_start_date": config.data.val_start_date,
            "val_end_date": config.data.val_end_date,
            "test_start_date": config.data.test_start_date,
        },
        "labels": {
            "horizon_days": config.labels.horizon_days,
            "train_target": "forward_return",
            "hit_rate_label_mode": config.labels.mode,
            "positive_quantile": config.labels.positive_quantile,
        },
    }
    with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    written = generate_experiment2_figures(RESULTS_DIR)
    logger.info("Wrote %d comparison figure(s)", len(written))

    logger.info("Experiment 2 complete. Results in %s", RESULTS_DIR)
    logger.info("Winner: %s", winner)
    if best_stage_b is not None:
        logger.info("Best Stage B (val Sharpe): %s", best_stage_b)


if __name__ == "__main__":
    main()
