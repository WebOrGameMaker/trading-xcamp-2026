"""Experiment 3 orchestrator — target (label) engineering (H3).

Trains the frozen pooled XGBoost model on the frozen Experiment 2 top-5
feature set for four prediction targets:

    A. Absolute 5-day forward return   (baseline, unchanged from Exp 1/2)
    B. Absolute 3-day forward return
    C. Absolute 10-day forward return
    D. Cross-sectional relative 5-day return (5d return minus the within-date
       median 5d return)

Each target is run through the identical weekly top-10 / bottom-10 long-short
trading pipeline. Ranking metrics (IC, IC IR, hit rate, decile spread) are
always scored against a *common yardstick* -- the absolute 5-day forward
return and a label fixed at the 5-day horizon -- regardless of which column a
given model was trained to predict, so B/C are not rewarded merely for having
an easier-to-predict own-horizon target. Own-target RMSE/MAE/R2 are kept only
as secondary fit diagnostics. The prediction target is the only independent
variable: model family, feature set, universe, splits, rebalance, costs, and
evaluation code are all frozen from Experiment 2.

Usage:
    python scripts/run_experiment3.py
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
from src.features.experiment3_targets import (
    DATASET_NAME,
    EVAL_HORIZON_DAYS,
    build_experiment3_feature_dataset,
)
from src.models.cross_sectional import (
    information_coefficient,
    mean_return_by_prediction_decile,
    top_decile_hit_rate,
)
from src.models.trainer import train_model
from src.utils.config import StrategyConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, ensure_directories
from src.visualization.experiment3_plots import generate_experiment3_figures

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_3"
FROZEN_MODEL = "xgboost"
EXP2_TOP5_PATH = PROJECT_ROOT / "results" / "experiment_2" / "top5" / "feature_columns.json"
SPLITS = ("train", "val", "test")
BACKTEST_SPLITS = ("val", "test")

# Common ranking yardstick: every target is scored on the same absolute
# 5-day return and the same fixed-horizon binary label, regardless of what
# it was trained to predict.
COMMON_RETURN_COL = f"forward_return_{EVAL_HORIZON_DAYS}d"
COMMON_LABEL_COL = "label"
BASELINE_TARGET = "A_5d_absolute"

TARGETS: dict[str, dict[str, Any]] = {
    "A_5d_absolute": {
        "target_col": "forward_return_5d",
        "purge_horizon": 5,
        "description": "Absolute 5-day forward return (baseline)",
    },
    "B_3d_absolute": {
        "target_col": "forward_return_3d",
        "purge_horizon": 3,
        "description": "Absolute 3-day forward return",
    },
    "C_10d_absolute": {
        "target_col": "forward_return_10d",
        "purge_horizon": 10,
        "description": "Absolute 10-day forward return",
    },
    "D_5d_relative": {
        "target_col": "forward_return_5d_rel",
        "purge_horizon": 5,
        "description": (
            "Cross-sectional relative 5-day return "
            "(5d return minus within-date median 5d return)"
        ),
    },
}
TARGET_ORDER = tuple(TARGETS.keys())

METRIC_COLS = ("rmse", "mae", "r2", "roc_auc", "pr_auc", "support")
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

# Same practical-materiality thresholds used in Experiment 1 / 2
# (docs/research_presentation.md).
MATERIALITY = {
    "delta_sharpe": 0.10,
    "delta_annual_return": 0.02,
    "delta_mean_daily_ic": 0.005,
}


@dataclass
class TargetRunResult:
    """Everything captured from one target-arm training run."""

    target: str
    target_col: str
    purge_horizon: int
    feature_columns: list[str]
    run_id: str
    manifest: dict[str, Any]
    out_dir: Path
    trading: dict[str, dict[str, float]]
    predictions: dict[str, pd.DataFrame]


def _read_latest_manifest() -> dict[str, Any]:
    path = MODEL_DIR / "latest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_frozen_top5_features() -> list[str]:
    """Load the Experiment 2 winning top-5 feature set (frozen for Exp 3)."""
    if not EXP2_TOP5_PATH.exists():
        raise FileNotFoundError(
            f"Experiment 2 top5 feature set not found: {EXP2_TOP5_PATH}. "
            "Run scripts/run_experiment2.py first."
        )
    with EXP2_TOP5_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = list(payload["feature_columns"])
    if not features:
        raise ValueError(f"Empty feature_columns in {EXP2_TOP5_PATH}")
    return features


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
            key: float(metrics_to_dict(metrics)[key]) for key in TRADING_METRIC_COLS
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


def _run_one_target(
    config,
    target: str,
    spec: dict[str, Any],
    feature_columns: list[str],
    strategy: StrategyConfig,
) -> TargetRunResult:
    """Train XGBoost on one prediction target, archive artifacts, and backtest."""
    logger.info(
        "=== Target %s (%s) — target_col=%s, purge=%d ===",
        target,
        spec["description"],
        spec["target_col"],
        spec["purge_horizon"],
    )
    config.model.type = FROZEN_MODEL
    config.model.task = "regression"
    train_model(
        config,
        feature_columns=feature_columns,
        dataset_name=DATASET_NAME,
        target_col_override=spec["target_col"],
        purge_horizon_override=spec["purge_horizon"],
    )

    manifest = _read_latest_manifest()
    artifact = manifest["artifacts"][0]
    run_id = manifest["run_id"]
    assert artifact["model_type"] == FROZEN_MODEL, (
        f"Manifest model_type mismatch: expected {FROZEN_MODEL}, "
        f"got {artifact['model_type']}"
    )

    out_dir = RESULTS_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        src = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            predictions[split] = pd.read_parquet(src)

    with (out_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "target": target,
                "target_col": spec["target_col"],
                "purge_horizon": spec["purge_horizon"],
                "feature_columns": feature_columns,
            },
            handle,
            indent=2,
        )

    importance = artifact["metrics"].get("feature_importance") or {}
    with (out_dir / "feature_importance.json").open("w", encoding="utf-8") as handle:
        json.dump({str(k): float(v) for k, v in importance.items()}, handle, indent=2)

    trading = _run_trading_pipeline(config, out_dir, strategy)
    with (out_dir / "trading_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(trading, handle, indent=2)

    logger.info("Archived target %s -> %s", target, out_dir)
    return TargetRunResult(
        target=target,
        target_col=spec["target_col"],
        purge_horizon=spec["purge_horizon"],
        feature_columns=list(feature_columns),
        run_id=run_id,
        manifest=manifest,
        out_dir=out_dir,
        trading=trading,
        predictions=predictions,
    )


def _metrics_table(results: list[TargetRunResult]) -> pd.DataFrame:
    """Own-target RMSE/MAE/R2 (fit diagnostics) + ROC/PR-AUC by target x split."""
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        for split in SPLITS:
            split_metrics = metrics[split]
            row = {
                "target": r.target,
                "target_col": r.target_col,
                "purge_horizon": r.purge_horizon,
                "split": split,
                "run_id": r.run_id,
            }
            row.update({col: split_metrics.get(col) for col in METRIC_COLS})
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[["target", "target_col", "purge_horizon", "split", "run_id", *METRIC_COLS]]


def _cross_sectional_table(results: list[TargetRunResult], split: str) -> pd.DataFrame:
    """Common-yardstick ranking metrics: always vs absolute 5d return + fixed label."""
    rows = []
    for r in results:
        pred_df = r.predictions.get(split)
        if pred_df is None or pred_df.empty:
            continue
        ic = information_coefficient(
            pred_df, score_col="probability", return_col=COMMON_RETURN_COL
        )
        hit_rate = top_decile_hit_rate(
            pred_df, score_col="probability", label_col=COMMON_LABEL_COL
        )
        ic_mean = ic["ic_mean_daily"]
        ic_std = ic["ic_std_daily"]
        ic_ir = ic_mean / ic_std if ic_std > 0 else None
        rows.append({
            "target": r.target,
            "split": split,
            "ic_overall": ic["ic_overall"],
            "ic_mean_daily": ic_mean,
            "ic_std_daily": ic_std,
            "ic_ir": ic_ir,
            "top_decile_hit_rate": hit_rate,
        })
    return pd.DataFrame(rows)


def _returns_table(results: list[TargetRunResult], split: str) -> pd.DataFrame:
    """Top/bottom decile mean return and spread, vs the common 5d return."""
    rows = []
    for r in results:
        pred_df = r.predictions.get(split)
        if pred_df is None or pred_df.empty:
            continue
        deciles = mean_return_by_prediction_decile(
            pred_df, score_col="probability", return_col=COMMON_RETURN_COL
        )
        if not deciles:
            continue
        ordered = sorted(deciles, key=lambda d: int(d["decile"]))
        bottom = float(ordered[0]["mean_forward_return"])
        top = float(ordered[-1]["mean_forward_return"])
        rows.append({
            "target": r.target,
            "split": split,
            "top_decile_mean_return": top,
            "bottom_decile_mean_return": bottom,
            "top_minus_bottom": top - bottom,
        })
    return pd.DataFrame(rows)


def _trading_table(results: list[TargetRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for split, metrics in r.trading.items():
            row = {
                "target": r.target,
                "target_col": r.target_col,
                "split": split,
                "run_id": r.run_id,
            }
            row.update({col: metrics.get(col) for col in TRADING_METRIC_COLS})
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["target", "target_col", "split", "run_id", *TRADING_METRIC_COLS]
        )
    return pd.DataFrame(rows)[["target", "target_col", "split", "run_id", *TRADING_METRIC_COLS]]


def _pick_winner(trading_df: pd.DataFrame, cs_df: pd.DataFrame) -> str:
    """Rank by test Sharpe, tie-break common-5d mean-daily IC, then hit rate."""
    test_trading = trading_df[trading_df["split"] == "test"].set_index("target")
    cs = cs_df[cs_df["split"] == "test"].set_index("target")
    ranking = pd.DataFrame({
        "sharpe_ratio": test_trading["sharpe_ratio"],
        "ic_mean_daily": cs["ic_mean_daily"],
        "top_decile_hit_rate": cs["top_decile_hit_rate"],
    })
    ranking = ranking.sort_values(
        ["sharpe_ratio", "ic_mean_daily", "top_decile_hit_rate"], ascending=False
    )
    logger.info("Target ranking (test Sharpe / IC / hit-rate):\n%s", ranking.to_string())
    return str(ranking.index[0])


def _pick_best_by_val(trading_df: pd.DataFrame) -> str | None:
    """Select the best target by validation Sharpe."""
    val = trading_df[trading_df["split"] == "val"]
    if val.empty:
        return None
    ranked = val.sort_values("sharpe_ratio", ascending=False)
    best = str(ranked.iloc[0]["target"])
    logger.info(
        "Best target by val Sharpe: %s (%.3f)", best, float(ranked.iloc[0]["sharpe_ratio"])
    )
    return best


def _deltas_vs_baseline(
    trading_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    baseline: str = BASELINE_TARGET,
) -> pd.DataFrame:
    """Delta Sharpe / Delta annualized return / Delta mean-daily IC vs baseline (test)."""
    test_trading = trading_df[trading_df["split"] == "test"].set_index("target")
    test_cs = cs_df[cs_df["split"] == "test"].set_index("target")
    if baseline not in test_trading.index:
        return pd.DataFrame()

    base_sharpe = float(test_trading.loc[baseline, "sharpe_ratio"])
    base_ann = float(test_trading.loc[baseline, "annualized_return"])
    base_ic = float(test_cs.loc[baseline, "ic_mean_daily"]) if baseline in test_cs.index else None

    rows = []
    for target in test_trading.index:
        if target == baseline:
            continue
        d_sharpe = float(test_trading.loc[target, "sharpe_ratio"]) - base_sharpe
        d_ann = float(test_trading.loc[target, "annualized_return"]) - base_ann
        d_ic = None
        if target in test_cs.index and base_ic is not None:
            d_ic = float(test_cs.loc[target, "ic_mean_daily"]) - base_ic
        material = (
            abs(d_sharpe) >= MATERIALITY["delta_sharpe"]
            or abs(d_ann) >= MATERIALITY["delta_annual_return"]
            or (d_ic is not None and abs(d_ic) >= MATERIALITY["delta_mean_daily_ic"])
        )
        rows.append({
            "target": target,
            "baseline": baseline,
            "delta_sharpe": d_sharpe,
            "delta_annual_return": d_ann,
            "delta_mean_daily_ic": d_ic,
            "material": material,
        })
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging(level="INFO")
    ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/default.yaml")
    config.model.type = FROZEN_MODEL
    config.model.task = "regression"
    strategy = _frozen_long_short_strategy(config)
    feature_columns = _load_frozen_top5_features()

    logger.info("Ensuring raw data is downloaded (cache-aware)...")
    download_universe(config, force=False)

    logger.info("Building Experiment 3 multi-target feature dataset...")
    build_experiment3_feature_dataset(config)

    results: list[TargetRunResult] = []
    for target in TARGET_ORDER:
        spec = TARGETS[target]
        results.append(_run_one_target(config, target, spec, feature_columns, strategy))

    metrics_df = _metrics_table(results)
    cs_val = _cross_sectional_table(results, split="val")
    cs_test = _cross_sectional_table(results, split="test")
    cs_df = pd.concat([cs_val, cs_test], ignore_index=True)
    returns_df = _returns_table(results, split="test")
    trading_df = _trading_table(results)
    deltas_df = _deltas_vs_baseline(trading_df, cs_df)

    metrics_path = RESULTS_DIR / "metrics_by_target_split.csv"
    cs_path = RESULTS_DIR / "cross_sectional_by_target.csv"
    returns_path = RESULTS_DIR / "returns_by_target.csv"
    trading_path = RESULTS_DIR / "trading_by_target.csv"
    deltas_path = RESULTS_DIR / "deltas_vs_baseline.csv"
    metrics_df.to_csv(metrics_path, index=False)
    cs_df.to_csv(cs_path, index=False)
    returns_df.to_csv(returns_path, index=False)
    trading_df.to_csv(trading_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)
    for path in (metrics_path, cs_path, returns_path, trading_path, deltas_path):
        logger.info("Wrote %s", path)

    winner = _pick_winner(trading_df, cs_df)
    best_by_val = _pick_best_by_val(trading_df)
    logger.info("Winning target (test Sharpe / IC / hit-rate): %s", winner)

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": "configs/default.yaml",
        "task": "regression",
        "frozen_model": FROZEN_MODEL,
        "frozen_features": feature_columns,
        "eval_horizon_days": EVAL_HORIZON_DAYS,
        "eval_return_col": COMMON_RETURN_COL,
        "eval_label_col": COMMON_LABEL_COL,
        "baseline_target": BASELINE_TARGET,
        "targets": {t: TARGETS[t] for t in TARGET_ORDER},
        "winner": winner,
        "winner_rule": "test_sharpe -> ic_mean_daily(common_5d) -> top_decile_hit_rate(common_5d)",
        "best_by_val_sharpe": best_by_val,
        "materiality_thresholds": MATERIALITY,
        "strategy": {
            "mode": strategy.mode,
            "long_positions": strategy.long_positions,
            "short_positions": strategy.short_positions,
            "rebalance_frequency": strategy.rebalance_frequency,
        },
        "runs": [
            {
                "target": r.target,
                "target_col": r.target_col,
                "purge_horizon": r.purge_horizon,
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
            "eval_horizon_days": EVAL_HORIZON_DAYS,
            "positive_quantile": config.labels.positive_quantile,
            "min_cross_section": config.labels.min_cross_section,
        },
    }
    with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    written = generate_experiment3_figures(RESULTS_DIR)
    logger.info("Wrote %d comparison figure(s)", len(written))

    logger.info("Experiment 3 complete. Results in %s", RESULTS_DIR)
    logger.info("Winner: %s", winner)
    if best_by_val is not None:
        logger.info("Best by val Sharpe: %s", best_by_val)


if __name__ == "__main__":
    main()
