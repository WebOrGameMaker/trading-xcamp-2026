"""Multi-arm comparison figures for Experiment 2 (H2 feature selection).

Reads archived CSVs and manifests under ``results/experiment_2/`` and writes
slide-ready comparison PNGs into ``results/experiment_2/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.features.families import STAGE_A_ARMS, STAGE_B_ARMS
from src.visualization.classification_plots import plot_feature_importance
from src.visualization.style import (
    REFERENCE_LINE_COLOR,
    apply_style,
    metric_title,
    save_figure,
)

ARM_ORDER = tuple(STAGE_A_ARMS) + tuple(STAGE_B_ARMS)
ARM_LABELS = {
    "full": "Full",
    "returns": "Returns",
    "trend": "Trend",
    "momentum": "Momentum",
    "volatility": "Volatility",
    "volume": "Volume",
    "returns_volatility": "Returns+Vol",
    "top5": "Top-5",
    "top10": "Top-10",
    "cum80": "Cum 80%",
}
STAGE_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
}

# Default talk-context tick labels are ~11pt; reduce ~30% for dense arm labels.
X_TICK_LABELSIZE = 11


def _style_arm_x_axis(ax: plt.Axes) -> None:
    """Rotate and shrink x-axis arm labels for readability."""
    ax.tick_params(axis="x", rotation=35, labelsize=X_TICK_LABELSIZE)

COMPARISON_FIGURES = (
    "arm_comparison_test_metrics.png",
    "arm_comparison_val_metrics.png",
    "arm_comparison_trading.png",
    "arm_comparison_trading_val.png",
    "arm_comparison_cross_sectional.png",
    "arm_comparison_cross_sectional_val.png",
    "full_feature_importance.png",
)


def load_experiment2_metrics(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "metrics_by_arm_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics table: {path}")
    return pd.read_csv(path)


def load_experiment2_cross_sectional(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "cross_sectional_by_arm.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-sectional table: {path}")
    return pd.read_csv(path)


def load_experiment2_cross_sectional_from_manifests(
    results_dir: str | Path,
    split: str = "val",
) -> pd.DataFrame:
    """Build a cross-sectional table for ``split`` from per-arm manifests.

    Useful when ``cross_sectional_by_arm.csv`` only archived test metrics.
    """
    results_dir = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        manifest_path = results_dir / arm / "model_manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metrics = manifest["artifacts"][0]["metrics"]
        split_metrics = metrics.get(split, {})
        cs = split_metrics.get("cross_sectional") or {}
        if not cs:
            continue
        stage = "A" if arm in STAGE_A_ARMS else "B"
        feature_path = results_dir / arm / "feature_columns.json"
        if feature_path.exists():
            with feature_path.open("r", encoding="utf-8") as handle:
                stage = str(json.load(handle).get("stage", stage))
        ic_mean = cs.get("ic_mean_daily")
        ic_std = cs.get("ic_std_daily")
        ic_ir = None
        if ic_mean is not None and ic_std is not None and float(ic_std) > 0:
            ic_ir = float(ic_mean) / float(ic_std)
        rows.append({
            "arm": arm,
            "stage": stage,
            "split": split,
            "ic_overall": cs.get("ic_overall"),
            "ic_mean_daily": ic_mean,
            "ic_std_daily": ic_std,
            "ic_ir": ic_ir,
            "top_decile_hit_rate": cs.get("top_decile_hit_rate"),
        })
    if not rows:
        raise FileNotFoundError(
            f"No cross-sectional metrics for split={split!r} under {results_dir}"
        )
    return pd.DataFrame(rows)


def resolve_cross_sectional(
    results_dir: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    """Prefer CSV rows for ``split``; fall back to per-arm manifests."""
    results_dir = Path(results_dir)
    csv_path = results_dir / "cross_sectional_by_arm.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        if "split" in frame.columns:
            subset = frame[frame["split"] == split]
            if not subset.empty:
                return subset.reset_index(drop=True)
        elif split == "test" and not frame.empty:
            # Legacy CSV without a split column is test-only.
            return frame
    return load_experiment2_cross_sectional_from_manifests(results_dir, split=split)


def load_experiment2_trading(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "trading_by_arm.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trading table: {path}")
    return pd.read_csv(path)


def load_experiment2_run_manifest(results_dir: str | Path) -> dict[str, Any]:
    path = Path(results_dir) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_full_feature_importance(results_dir: str | Path) -> dict[str, float]:
    path = Path(results_dir) / "full_feature_importance.json"
    if not path.exists():
        # Fall back to full arm manifest if the sidecar is missing.
        manifest_path = Path(results_dir) / "full" / "model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing Full importance: {path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        importance = (
            manifest["artifacts"][0]["metrics"].get("feature_importance") or {}
        )
        return {str(k): float(v) for k, v in importance.items()}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k): float(v) for k, v in raw.items()}


def _arm_order(present: list[str] | pd.Index | set[str]) -> list[str]:
    present_set = set(present)
    ordered = [a for a in ARM_ORDER if a in present_set]
    extras = sorted(present_set - set(ordered))
    return ordered + extras


def _arm_colors(arms: list[str], stage_by_arm: dict[str, str] | None = None) -> list[str]:
    colors = []
    for arm in arms:
        stage = (stage_by_arm or {}).get(arm, "A" if arm in STAGE_A_ARMS else "B")
        colors.append(STAGE_COLORS.get(stage, "#7f7f7f"))
    return colors


def _stage_lookup(frame: pd.DataFrame) -> dict[str, str]:
    if "stage" not in frame.columns:
        return {}
    return {
        str(row.arm): str(row.stage)
        for row in frame[["arm", "stage"]].drop_duplicates().itertuples(index=False)
    }


def plot_arm_comparison_metrics(
    metrics_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Grouped bars: IC, hit rate, and R² by feature arm for one split."""
    apply_style()
    split_metrics = metrics_df[metrics_df["split"] == split].set_index("arm")
    if split_metrics.empty:
        raise ValueError(f"No ranking/fit metrics found for split={split!r}")
    cs_frame = cs_df
    if "split" in cs_frame.columns:
        cs_frame = cs_frame[cs_frame["split"] == split]
    if cs_frame.empty:
        raise ValueError(f"No cross-sectional metrics found for split={split!r}")
    cs = cs_frame.drop_duplicates(subset=["arm"]).set_index("arm")
    arm_order = _arm_order(split_metrics.index.intersection(cs.index))
    if not arm_order:
        arm_order = _arm_order(split_metrics.index)
    stage_by_arm = _stage_lookup(metrics_df)
    label = split.title()

    metrics_to_plot = [
        ("ic_mean_daily", f"{label} IC (mean daily)", cs["ic_mean_daily"], 0.0, True),
        (
            "top_decile_hit_rate",
            f"{label} top-decile hit rate",
            cs["top_decile_hit_rate"],
            0.2,
            True,
        ),
        (
            "r2",
            f"{label} R²",
            split_metrics["r2"] if "r2" in split_metrics.columns else pd.Series(dtype=float),
            0.0,
            True,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, series, baseline, higher_better) in zip(
        axes, metrics_to_plot, strict=True
    ):
        values = [float(series.loc[a]) if a in series.index else 0.0 for a in arm_order]
        colors = _arm_colors(arm_order, stage_by_arm)
        bars = ax.bar([ARM_LABELS.get(a, a) for a in arm_order], values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(metric_title(title, higher_better=higher_better))
        _style_arm_x_axis(ax)

    fig.suptitle(
        f"Experiment 2 — Feature Arms ({label} Ranking / Fit)",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_arm_comparison_test_metrics(
    metrics_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Backward-compatible wrapper for test-split ranking/fit bars."""
    return plot_arm_comparison_metrics(metrics_df, cs_df, out_path, split="test")


def plot_arm_comparison_cross_sectional(
    cs_df: pd.DataFrame,
    out_path: str | Path,
    split: str | None = None,
) -> Path:
    """Bars: IC overall, mean daily IC, and top-decile hit rate by arm."""
    apply_style()
    frame = cs_df
    if split is not None and "split" in frame.columns:
        frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"No cross-sectional metrics found for split={split!r}")
    resolved_split = split or (
        str(frame["split"].iloc[0]) if "split" in frame.columns else "test"
    )
    cs = frame.set_index("arm")
    arm_order = _arm_order(cs.index)
    stage_by_arm = _stage_lookup(frame)

    metrics_to_plot = [
        ("ic_overall", "IC (overall)", 0.0, True),
        ("ic_mean_daily", "IC (mean daily)", 0.0, True),
        ("top_decile_hit_rate", "Top-decile hit rate", 0.2, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, baseline, higher_better) in zip(axes, metrics_to_plot, strict=True):
        values = [float(cs.loc[a, key]) for a in arm_order]
        colors = _arm_colors(arm_order, stage_by_arm)
        bars = ax.bar([ARM_LABELS.get(a, a) for a in arm_order], values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(metric_title(title, higher_better=higher_better))
        _style_arm_x_axis(ax)

    fig.suptitle(
        f"Experiment 2 — Cross-Sectional Ranking by Feature Arm "
        f"({resolved_split.title()})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_arm_comparison_trading(
    trading_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Bars: Sharpe, annualized return, and max drawdown by feature arm."""
    apply_style()
    subset = trading_df[trading_df["split"] == split].set_index("arm")
    if subset.empty:
        raise ValueError(f"No trading metrics found for split={split!r}")
    arm_order = _arm_order(subset.index)
    stage_by_arm = _stage_lookup(trading_df)

    metrics_to_plot = [
        ("sharpe_ratio", "Sharpe ratio", "{:.2f}", True),
        ("annualized_return", "Annualized return", "{:.1%}", True),
        ("max_drawdown", "Max drawdown", "{:.1%}", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, fmt, higher_better) in zip(axes, metrics_to_plot, strict=True):
        values = [float(subset.loc[a, key]) for a in arm_order]
        colors = _arm_colors(arm_order, stage_by_arm)
        bars = ax.bar([ARM_LABELS.get(a, a) for a in arm_order], values, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=8)
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(metric_title(title, higher_better=higher_better))
        _style_arm_x_axis(ax)

    fig.suptitle(
        f"Experiment 2 — Long/Short Trading by Feature Arm ({split.title()})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def generate_experiment2_figures(results_dir: str | Path) -> list[Path]:
    """Write Experiment 2 comparison figures into ``results_dir/figures``.

    Generates both validation and test comparison charts when data are available.
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_experiment2_metrics(results_dir)
    trading_df = load_experiment2_trading(results_dir)
    importance = load_full_feature_importance(results_dir)

    written: list[Path] = []

    # Test figures (presentation primary).
    cs_test = resolve_cross_sectional(results_dir, split="test")
    written.append(
        plot_arm_comparison_metrics(
            metrics_df,
            cs_test,
            figures_dir / "arm_comparison_test_metrics.png",
            split="test",
        )
    )
    written.append(
        plot_arm_comparison_cross_sectional(
            cs_test,
            figures_dir / "arm_comparison_cross_sectional.png",
            split="test",
        )
    )
    written.append(
        plot_arm_comparison_trading(
            trading_df, figures_dir / "arm_comparison_trading.png", split="test"
        )
    )

    # Validation figures (selection / overfitting diagnostics).
    try:
        cs_val = resolve_cross_sectional(results_dir, split="val")
        written.append(
            plot_arm_comparison_metrics(
                metrics_df,
                cs_val,
                figures_dir / "arm_comparison_val_metrics.png",
                split="val",
            )
        )
        written.append(
            plot_arm_comparison_cross_sectional(
                cs_val,
                figures_dir / "arm_comparison_cross_sectional_val.png",
                split="val",
            )
        )
    except (FileNotFoundError, ValueError):
        pass

    if not trading_df[trading_df["split"] == "val"].empty:
        written.append(
            plot_arm_comparison_trading(
                trading_df,
                figures_dir / "arm_comparison_trading_val.png",
                split="val",
            )
        )

    if importance:
        written.append(
            plot_feature_importance(
                importance, figures_dir / "full_feature_importance.png"
            )
        )
    return written
