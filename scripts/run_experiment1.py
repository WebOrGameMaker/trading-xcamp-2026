"""Experiment 1 orchestrator — pooled model family comparison (H1).

Trains pooled XGBoost, LightGBM, and Random Forest classifiers on identical
features/labels/splits (per configs/default.yaml), archives every artifact
needed for the presentation into results/experiment_1/, and writes
comparison tables + figures. Does not touch calibration or confidence-gated
trading (Experiment 2/3) — those remain out of scope until reviewed.

Usage:
    python scripts/run_experiment1.py
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

import matplotlib.pyplot as plt
import pandas as pd

from src.data.downloader import download_universe
from src.features.pipeline import build_feature_dataset
from src.models.trainer import train_model
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, ensure_directories
from src.visualization import loaders
from src.visualization.report import generate_report
from src.visualization.style import apply_style, save_figure

logger = get_logger(__name__)

MODEL_TYPES = ["xgboost", "lightgbm", "random_forest"]
RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_1"
SPLITS = ("train", "val", "test")
METRIC_COLS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece",
    "support",
)
FIGURE_FILES = (
    "01_classification_metric_overview.png",
    "02_roc_auc_distribution.png",
    "03_confusion_matrices.png",
    "04_probability_separation.png",
    "05_feature_importance.png",
    "09_calibration_curve.png",
)
MODEL_COLORS = {
    "xgboost": "#1f77b4",
    "lightgbm": "#2ca02c",
    "random_forest": "#d62728",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
}


@dataclass
class ModelRunResult:
    """Everything captured from one pooled training run."""

    model_type: str
    run_id: str
    manifest: dict[str, Any]
    out_dir: Path


def _read_latest_manifest() -> dict[str, Any]:
    path = MODEL_DIR / "latest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_one_model(config, model_type: str) -> ModelRunResult:
    """Train one pooled model and archive its artifacts."""
    logger.info("=== Training pooled %s ===", model_type)
    config.model.type = model_type
    train_model(config)

    manifest = _read_latest_manifest()
    artifact = manifest["artifacts"][0]
    run_id = manifest["run_id"]
    assert artifact["model_type"] == model_type, (
        f"Manifest model_type mismatch: expected {model_type}, got {artifact['model_type']}"
    )

    out_dir = RESULTS_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    with (out_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    try:
        report = generate_report(config, run_id=run_id)
        fig_out = out_dir / "figures"
        fig_out.mkdir(parents=True, exist_ok=True)
        for fname in FIGURE_FILES:
            src_fig = report.output_dir / fname
            if src_fig.exists():
                shutil.copy2(src_fig, fig_out / fname)
        if report.skipped:
            logger.warning("%s: figures skipped: %s", model_type, report.skipped)
    except Exception:
        logger.exception("Figure generation failed for %s (continuing)", model_type)

    logger.info("Archived %s results -> %s", model_type, out_dir)
    return ModelRunResult(model_type=model_type, run_id=run_id, manifest=manifest, out_dir=out_dir)


def _metrics_table(results: list[ModelRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        for split in SPLITS:
            split_metrics = metrics[split]
            row = {"model_type": r.model_type, "split": split, "run_id": r.run_id}
            row.update({col: split_metrics.get(col) for col in METRIC_COLS})
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[["model_type", "split", "run_id", *METRIC_COLS]]


def _cross_sectional_table(results: list[ModelRunResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.manifest["artifacts"][0]["metrics"]
        cs = metrics[split].get("cross_sectional", {})
        rows.append({
            "model_type": r.model_type,
            "split": split,
            "ic_overall": cs.get("ic_overall"),
            "ic_mean_daily": cs.get("ic_mean_daily"),
            "ic_std_daily": cs.get("ic_std_daily"),
            "top_decile_hit_rate": cs.get("top_decile_hit_rate"),
        })
    return pd.DataFrame(rows)


def _pick_winner(metrics_df: pd.DataFrame, cs_df: pd.DataFrame) -> str:
    """Rank by test ROC-AUC, tie-break PR-AUC, then IC mean daily."""
    test = metrics_df[metrics_df["split"] == "test"].set_index("model_type")
    cs = cs_df.set_index("model_type")
    ranking = pd.DataFrame({
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
        "ic_mean_daily": cs["ic_mean_daily"],
    })
    ranking = ranking.sort_values(
        ["roc_auc", "pr_auc", "ic_mean_daily"], ascending=False
    )
    logger.info("Model ranking (test):\n%s", ranking.to_string())
    return str(ranking.index[0])


def _plot_model_comparison(metrics_df: pd.DataFrame, cs_df: pd.DataFrame, out_path: Path) -> Path:
    """Bonus grouped bar chart: test ROC-AUC / PR-AUC / IC-mean-daily by model."""
    apply_style()
    test = metrics_df[metrics_df["split"] == "test"].set_index("model_type")
    cs = cs_df.set_index("model_type")
    model_order = [m for m in MODEL_TYPES if m in test.index]

    metrics_to_plot = [
        ("roc_auc", "Test ROC-AUC", test["roc_auc"]),
        ("pr_auc", "Test PR-AUC", test["pr_auc"]),
        ("ic_mean_daily", "Test IC (mean daily)", cs["ic_mean_daily"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (key, title, series) in zip(axes, metrics_to_plot, strict=True):
        values = [series.get(m, 0.0) for m in model_order]
        colors = [MODEL_COLORS.get(m, "#7f7f7f") for m in model_order]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in model_order], values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        if key in ("roc_auc", "pr_auc"):
            ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
        else:
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 1 — Model Comparison (Test Split)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return out_path


def main() -> None:
    setup_logging(level="INFO")
    ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/default.yaml")

    logger.info("Ensuring raw data is downloaded (cache-aware)...")
    download_universe(config, force=False)

    logger.info("Rebuilding feature dataset from current code...")
    build_feature_dataset(config)

    results: list[ModelRunResult] = []
    for model_type in MODEL_TYPES:
        results.append(_run_one_model(config, model_type))

    metrics_df = _metrics_table(results)
    cs_df = _cross_sectional_table(results, split="test")

    metrics_path = RESULTS_DIR / "metrics_by_model_split.csv"
    cs_path = RESULTS_DIR / "cross_sectional_by_model.csv"
    metrics_df.to_csv(metrics_path, index=False)
    cs_df.to_csv(cs_path, index=False)
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", cs_path)

    winner = _pick_winner(metrics_df, cs_df)
    logger.info("Winning model (test ROC-AUC / PR-AUC / IC tie-break): %s", winner)

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    winner_result = next(r for r in results if r.model_type == winner)
    winner_importance = winner_result.manifest["artifacts"][0]["metrics"].get("feature_importance", {})
    if winner_importance:
        from src.visualization.classification_plots import plot_feature_importance
        plot_feature_importance(
            winner_importance, figures_dir / "feature_importance_winner.png"
        )

    _plot_model_comparison(metrics_df, cs_df, figures_dir / "model_comparison_test_metrics.png")

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": "configs/default.yaml",
        "winner": winner,
        "runs": [
            {
                "model_type": r.model_type,
                "run_id": r.run_id,
                "train_rows": r.manifest["artifacts"][0]["train_rows"],
                "val_rows": r.manifest["artifacts"][0]["val_rows"],
                "test_rows": r.manifest["artifacts"][0]["test_rows"],
                "feature_columns": r.manifest["artifacts"][0]["feature_columns"],
                "generalization_gap": r.manifest["artifacts"][0]["metrics"].get("generalization_gap"),
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
            "mode": config.labels.mode,
            "positive_quantile": config.labels.positive_quantile,
        },
    }
    with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    logger.info("Experiment 1 complete. Results in %s", RESULTS_DIR)
    logger.info("Winner: %s", winner)


if __name__ == "__main__":
    main()
