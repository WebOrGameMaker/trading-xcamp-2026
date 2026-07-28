"""Multi-model comparison figures for Experiment 1 (H1).

Reads archived CSVs and model manifests under ``results/experiment_1/`` and
writes slide-ready comparison PNGs into ``results/experiment_1/figures/``.
Does not retrain; suitable for regenerating presentation assets after a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.classification_plots import plot_feature_importance
from src.visualization.style import (
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    REFERENCE_LINE_COLOR,
    apply_style,
    save_figure,
)

MODEL_TYPES = ("xgboost", "lightgbm", "random_forest", "catboost")
MODEL_COLORS = {
    "xgboost": "#1f77b4",
    "lightgbm": "#2ca02c",
    "random_forest": "#d62728",
    "catboost": "#ff7f0e",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "catboost": "CatBoost",
}
SPLITS = ("train", "val", "test")

COMPARISON_FIGURES = (
    "model_comparison_test_metrics.png",
    "model_comparison_metrics_by_split.png",
    "model_comparison_cross_sectional.png",
    "model_comparison_feature_importance.png",
    "model_comparison_prediction_deciles.png",
    "model_comparison_returns.png",
    "model_comparison_trading.png",
    "feature_importance_winner.png",
)


def load_experiment1_metrics(results_dir: str | Path) -> pd.DataFrame:
    """Load ``metrics_by_model_split.csv`` from an Experiment 1 results directory."""
    path = Path(results_dir) / "metrics_by_model_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics table: {path}")
    return pd.read_csv(path)


def load_experiment1_cross_sectional(results_dir: str | Path) -> pd.DataFrame:
    """Load ``cross_sectional_by_model.csv`` from an Experiment 1 results directory."""
    path = Path(results_dir) / "cross_sectional_by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-sectional table: {path}")
    return pd.read_csv(path)


def load_experiment1_trading(results_dir: str | Path) -> pd.DataFrame:
    """Load ``trading_by_model.csv`` from an Experiment 1 results directory."""
    path = Path(results_dir) / "trading_by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trading table: {path}")
    return pd.read_csv(path)


def load_experiment1_run_manifest(results_dir: str | Path) -> dict[str, Any]:
    """Load ``run_manifest.json`` (winner + run metadata)."""
    path = Path(results_dir) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment1_manifests(results_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load per-model ``model_manifest.json`` files keyed by model type."""
    results_dir = Path(results_dir)
    manifests: dict[str, dict[str, Any]] = {}
    for model_type in MODEL_TYPES:
        path = results_dir / model_type / "model_manifest.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            manifests[model_type] = json.load(handle)
    if not manifests:
        raise FileNotFoundError(
            f"No model manifests found under {results_dir}/{{model}}/model_manifest.json"
        )
    return manifests


def _artifact_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["artifacts"][0]["metrics"]


def load_experiment1_feature_importances(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Extract feature_importance dicts keyed by model type."""
    out: dict[str, dict[str, float]] = {}
    for model_type, manifest in manifests.items():
        importance = _artifact_metrics(manifest).get("feature_importance") or {}
        if importance:
            out[model_type] = {str(k): float(v) for k, v in importance.items()}
    return out


def load_experiment1_deciles(
    manifests: dict[str, dict[str, Any]],
    split: str = "test",
) -> pd.DataFrame:
    """Tidy frame of mean forward return by prediction decile × model."""
    rows: list[dict[str, Any]] = []
    for model_type, manifest in manifests.items():
        split_metrics = _artifact_metrics(manifest).get(split, {})
        cs = split_metrics.get("cross_sectional") or {}
        for entry in cs.get("mean_return_by_prediction_decile") or []:
            rows.append({
                "model_type": model_type,
                "split": split,
                "decile": int(entry["decile"]),
                "mean_forward_return": float(entry["mean_forward_return"]),
                "count": int(entry.get("count", 0)),
            })
    if not rows:
        raise ValueError(
            f"No mean_return_by_prediction_decile entries found for split={split!r}"
        )
    return pd.DataFrame(rows)


def _model_order(present: list[str] | pd.Index | set[str]) -> list[str]:
    present_set = set(present)
    return [m for m in MODEL_TYPES if m in present_set]


def plot_model_comparison_test_metrics(
    metrics_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Grouped bar chart: test ranking / fit metrics by model.

    Prefers IC + R² when regression columns are present; falls back to
    ROC-AUC / PR-AUC for older classification Experiment 1 archives.
    """
    apply_style()
    test = metrics_df[metrics_df["split"] == "test"].set_index("model_type")
    cs = cs_df.set_index("model_type")
    model_order = _model_order(test.index)

    has_r2 = "r2" in test.columns and test["r2"].notna().any()
    if has_r2:
        metrics_to_plot = [
            ("ic_mean_daily", "Test IC (mean daily)", cs["ic_mean_daily"], 0.0),
            ("top_decile_hit_rate", "Test top-decile hit rate", cs["top_decile_hit_rate"], 0.2),
            ("r2", "Test R²", test["r2"], 0.0),
        ]
    else:
        metrics_to_plot = [
            ("roc_auc", "Test ROC-AUC", test["roc_auc"], 0.5),
            ("pr_auc", "Test PR-AUC", test["pr_auc"], 0.2),
            ("ic_mean_daily", "Test IC (mean daily)", cs["ic_mean_daily"], 0.0),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (key, title, series, baseline) in zip(axes, metrics_to_plot, strict=True):
        values = [float(series.get(m, 0.0)) for m in model_order]
        colors = [MODEL_COLORS.get(m, "#7f7f7f") for m in model_order]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in model_order], values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 1 — Model Comparison (Test Split)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_model_comparison_metrics_by_split(
    metrics_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Grouped bars of primary fit metrics across train/val/test × model."""
    apply_style()
    model_order = _model_order(metrics_df["model_type"].unique())
    splits = [s for s in SPLITS if s in set(metrics_df["split"])]

    has_r2 = "r2" in metrics_df.columns and metrics_df["r2"].notna().any()
    if has_r2:
        metric_specs = (("r2", "R²"), ("rmse", "RMSE"))
        baselines = (0.0, None)
        suptitle = "Experiment 1 — Regression Metrics by Split"
    else:
        metric_specs = (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"))
        baselines = (0.5, 0.2)
        suptitle = "Experiment 1 — Classification Metrics by Split"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    x = np.arange(len(splits))
    width = 0.8 / max(len(model_order), 1)

    for ax, (metric, title), baseline in zip(axes, metric_specs, baselines, strict=True):
        for i, model_type in enumerate(model_order):
            subset = metrics_df[metrics_df["model_type"] == model_type].set_index("split")
            values = [
                float(subset.loc[s, metric]) if s in subset.index and metric in subset.columns else 0.0
                for s in splits
            ]
            offset = (i - (len(model_order) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=MODEL_LABELS.get(model_type, model_type),
                color=MODEL_COLORS.get(model_type, "#7f7f7f"),
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        if baseline is not None:
            ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in splits])
        ax.set_title(title)
        ax.set_ylabel(title)

    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_model_comparison_cross_sectional(
    cs_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Bar chart: IC overall, IC mean daily, and top-decile hit rate by model."""
    apply_style()
    cs = cs_df.set_index("model_type")
    model_order = _model_order(cs.index)

    metrics_to_plot = [
        ("ic_overall", "IC (overall)", 0.0),
        ("ic_mean_daily", "IC (mean daily)", 0.0),
        ("top_decile_hit_rate", "Top-decile hit rate", 0.2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (key, title, baseline) in zip(axes, metrics_to_plot, strict=True):
        values = [float(cs.loc[m, key]) for m in model_order]
        colors = [MODEL_COLORS.get(m, "#7f7f7f") for m in model_order]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in model_order], values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 1 — Cross-Sectional Ranking (Test)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def _normalize_importances(importances: dict[str, float]) -> dict[str, float]:
    """Scale importances to sum to 1 so split-count and gain scales are comparable."""
    total = float(sum(importances.values()))
    if total <= 0:
        return {k: 0.0 for k in importances}
    return {k: float(v) / total for k, v in importances.items()}


def plot_model_comparison_feature_importance(
    importances_by_model: dict[str, dict[str, float]],
    out_path: str | Path,
    top_n: int = 10,
) -> Path:
    """Grouped horizontal bars of top features across all models.

    Each model's importances are L1-normalized before plotting so LightGBM
    split counts are comparable to XGBoost gain / Random Forest Gini.
    """
    apply_style()
    model_order = _model_order(importances_by_model.keys())
    if not model_order:
        raise ValueError("No feature importances provided")

    normalized = {
        model_type: _normalize_importances(importances_by_model[model_type])
        for model_type in model_order
    }

    # Union of top-N features per model, ranked by mean normalized importance.
    candidate_scores: dict[str, float] = {}
    for model_type in model_order:
        ranked = sorted(
            normalized[model_type].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:top_n]
        for name, value in ranked:
            candidate_scores[name] = candidate_scores.get(name, 0.0) + float(value)
    for name in candidate_scores:
        candidate_scores[name] /= len(model_order)

    features = [
        name
        for name, _ in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    ][::-1]

    y = np.arange(len(features))
    height = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    for i, model_type in enumerate(model_order):
        values = [float(normalized[model_type].get(f, 0.0)) for f in features]
        offset = (i - (len(model_order) - 1) / 2) * height
        ax.barh(
            y + offset,
            values,
            height=height,
            label=MODEL_LABELS.get(model_type, model_type),
            color=MODEL_COLORS.get(model_type, "#7f7f7f"),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.set_xlabel("Normalized importance (sums to 1)")
    ax.set_title(f"Top {len(features)} Feature Importances — All Models")
    ax.legend(loc="lower right", framealpha=0.9)

    save_figure(fig, out_path)
    return Path(out_path)


def plot_model_comparison_prediction_deciles(
    deciles_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Line chart: mean forward return by prediction decile for each model."""
    apply_style()
    model_order = _model_order(deciles_df["model_type"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for model_type in model_order:
        subset = (
            deciles_df[deciles_df["model_type"] == model_type]
            .sort_values("decile")
        )
        ax.plot(
            subset["decile"],
            subset["mean_forward_return"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model_type, model_type),
            color=MODEL_COLORS.get(model_type, "#7f7f7f"),
        )

    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_xticks(sorted(deciles_df["decile"].unique()))
    ax.set_xlabel("Prediction decile (1 = lowest predicted return)")
    ax.set_ylabel("Mean forward return")
    ax.set_title("Experiment 1 — Mean Forward Return by Prediction Decile (Test)")
    ax.legend(loc="upper left", framealpha=0.9)

    save_figure(fig, out_path)
    return Path(out_path)


def _decile_extremes(deciles_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model top-decile return and top−bottom spread from decile table."""
    rows: list[dict[str, Any]] = []
    for model_type, subset in deciles_df.groupby("model_type"):
        ordered = subset.sort_values("decile")
        if ordered.empty:
            continue
        bottom = float(ordered.iloc[0]["mean_forward_return"])
        top = float(ordered.iloc[-1]["mean_forward_return"])
        rows.append({
            "model_type": model_type,
            "top_decile_mean_return": top,
            "bottom_decile_mean_return": bottom,
            "top_minus_bottom": top - bottom,
        })
    return pd.DataFrame(rows)


def plot_model_comparison_returns(
    deciles_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Bar chart: top-decile mean return and top−bottom spread by model."""
    apply_style()
    extremes = _decile_extremes(deciles_df)
    if extremes.empty:
        raise ValueError("No decile returns available for returns comparison plot")
    extremes = extremes.set_index("model_type")
    model_order = _model_order(extremes.index)

    metrics_to_plot = [
        ("top_decile_mean_return", "Top-decile mean forward return"),
        ("top_minus_bottom", "Top − bottom decile spread"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key, title) in zip(axes, metrics_to_plot, strict=True):
        values = [float(extremes.loc[m, key]) for m in model_order]
        colors = [MODEL_COLORS.get(m, "#7f7f7f") for m in model_order]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in model_order], values, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Mean forward return")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Experiment 1 — Realized Returns by Model (Test)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_model_comparison_trading(
    trading_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Bar chart: Sharpe and annualized return from the shared long_short backtest."""
    apply_style()
    subset = trading_df[trading_df["split"] == split].set_index("model_type")
    if subset.empty:
        raise ValueError(f"No trading metrics found for split={split!r}")
    model_order = _model_order(subset.index)

    metrics_to_plot = [
        ("sharpe_ratio", "Sharpe ratio", "{:.2f}"),
        ("annualized_return", "Annualized return", "{:.1%}"),
        ("max_drawdown", "Max drawdown", "{:.1%}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, title, fmt) in zip(axes, metrics_to_plot, strict=True):
        values = [float(subset.loc[m, key]) for m in model_order]
        colors = [MODEL_COLORS.get(m, "#7f7f7f") for m in model_order]
        bars = ax.bar([MODEL_LABELS.get(m, m) for m in model_order], values, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3)
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"Experiment 1 — Long/Short Trading Performance ({split.title()})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def generate_experiment1_figures(
    results_dir: str | Path,
    *,
    regenerate_per_model: bool = True,
) -> list[Path]:
    """Write all Experiment 1 comparison figures into ``results_dir/figures``.

    Args:
        results_dir: Path to ``results/experiment_1`` (or a test fixture).
        regenerate_per_model: When True, also attempt per-model figure regen
            if prediction parquets are present under each model directory.

    Returns:
        Paths of figures successfully written (comparison + optional per-model).
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_experiment1_metrics(results_dir)
    cs_df = load_experiment1_cross_sectional(results_dir)
    manifests = load_experiment1_manifests(results_dir)
    run_manifest = load_experiment1_run_manifest(results_dir)
    importances = load_experiment1_feature_importances(manifests)
    deciles_df = load_experiment1_deciles(manifests, split="test")

    written: list[Path] = []
    written.append(
        plot_model_comparison_test_metrics(
            metrics_df, cs_df, figures_dir / "model_comparison_test_metrics.png"
        )
    )
    written.append(
        plot_model_comparison_metrics_by_split(
            metrics_df, figures_dir / "model_comparison_metrics_by_split.png"
        )
    )
    written.append(
        plot_model_comparison_cross_sectional(
            cs_df, figures_dir / "model_comparison_cross_sectional.png"
        )
    )
    if importances:
        written.append(
            plot_model_comparison_feature_importance(
                importances, figures_dir / "model_comparison_feature_importance.png"
            )
        )
    written.append(
        plot_model_comparison_prediction_deciles(
            deciles_df, figures_dir / "model_comparison_prediction_deciles.png"
        )
    )
    written.append(
        plot_model_comparison_returns(
            deciles_df, figures_dir / "model_comparison_returns.png"
        )
    )

    trading_path = results_dir / "trading_by_model.csv"
    if trading_path.exists():
        trading_df = load_experiment1_trading(results_dir)
        written.append(
            plot_model_comparison_trading(
                trading_df, figures_dir / "model_comparison_trading.png", split="test"
            )
        )

    winner = str(run_manifest.get("winner", ""))
    winner_importance = importances.get(winner) or {}
    if winner_importance:
        written.append(
            plot_feature_importance(
                winner_importance, figures_dir / "feature_importance_winner.png"
            )
        )

    if regenerate_per_model:
        written.extend(_regenerate_per_model_figures(results_dir, manifests))

    return written


def _eval_frame_from_manifest(manifest: dict[str, Any]) -> pd.DataFrame:
    """Build a minimal tidy eval frame for classification plotters 01/02."""
    metrics = _artifact_metrics(manifest)
    run_id = str(manifest.get("run_id", "pooled"))
    rows = []
    for split in SPLITS:
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict):
            continue
        if "roc_auc" not in split_metrics:
            continue
        # Classification overview needs hard-label metrics; skip regression-only.
        if "accuracy" not in split_metrics and "r2" in split_metrics:
            continue
        rows.append({
            "run_id": run_id,
            "symbol": "pooled",
            "split": split,
            "accuracy": split_metrics.get("accuracy"),
            "precision": split_metrics.get("precision"),
            "recall": split_metrics.get("recall"),
            "f1": split_metrics.get("f1"),
            "roc_auc": split_metrics.get("roc_auc"),
            "pr_auc": split_metrics.get("pr_auc"),
            "brier_score": split_metrics.get("brier_score"),
            "support": split_metrics.get("support", 0),
        })
    return pd.DataFrame(rows)


def _regenerate_per_model_figures(
    results_dir: Path,
    manifests: dict[str, dict[str, Any]],
) -> list[Path]:
    """Optionally regenerate per-model figures when prediction parquets exist."""
    from src.utils.logging import get_logger
    from src.visualization import classification_plots

    logger = get_logger(__name__)
    written: list[Path] = []

    for model_type, manifest in manifests.items():
        model_dir = results_dir / model_type
        pred_val = model_dir / "predictions_val.parquet"
        pred_test = model_dir / "predictions_test.parquet"
        fig_out = model_dir / "figures"
        fig_out.mkdir(parents=True, exist_ok=True)

        eval_df = _eval_frame_from_manifest(manifest)
        if not eval_df.empty:
            written.append(
                classification_plots.plot_metric_overview(
                    eval_df, fig_out / "01_classification_metric_overview.png"
                )
            )
            written.append(
                classification_plots.plot_roc_auc_distribution(
                    eval_df, fig_out / "02_roc_auc_distribution.png"
                )
            )

        importance = load_experiment1_feature_importances({model_type: manifest}).get(model_type)
        if importance:
            written.append(
                classification_plots.plot_feature_importance(
                    importance, fig_out / "05_feature_importance.png"
                )
            )

        # Probability separation / calibration assume [0,1] probabilities.
        top_metrics = _artifact_metrics(manifest)
        is_regression = bool(top_metrics.get("task") == "regression" or (
            isinstance(top_metrics.get("test"), dict) and "r2" in top_metrics["test"]
        ))
        if is_regression:
            logger.info("%s: skipping 04/09 (regression scores are not probabilities)", model_type)
            continue

        if not (pred_val.exists() and pred_test.exists()):
            logger.info(
                "%s: skipping 04/09 (missing predictions_val/test.parquet)",
                model_type,
            )
            continue

        val_df = pd.read_parquet(pred_val)
        test_df = pd.read_parquet(pred_test)
        written.append(
            classification_plots.plot_probability_separation(
                val_df, test_df, fig_out / "04_probability_separation.png"
            )
        )
        written.append(
            classification_plots.plot_calibration_curves(
                {"val": val_df, "test": test_df},
                fig_out / "09_calibration_curve.png",
            )
        )
        logger.info("%s: regenerated per-model figures under %s", model_type, fig_out)

    return written
