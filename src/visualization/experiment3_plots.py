"""Multi-target comparison figures for Experiment 3 (H3 target engineering).

Reads archived CSVs under ``results/experiment_3/`` and writes slide-ready
comparison PNGs into ``results/experiment_3/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    REFERENCE_LINE_COLOR,
    apply_style,
    metric_title,
    save_figure,
)

TARGET_ORDER = (
    "A_5d_absolute",
    "B_3d_absolute",
    "C_10d_absolute",
    "D_5d_relative",
)
TARGET_LABELS = {
    "A_5d_absolute": "A: 5d abs\n(baseline)",
    "B_3d_absolute": "B: 3d abs",
    "C_10d_absolute": "C: 10d abs",
    "D_5d_relative": "D: 5d rel",
}
TARGET_COLORS = {
    "A_5d_absolute": "#1f77b4",
    "B_3d_absolute": "#ff7f0e",
    "C_10d_absolute": "#2ca02c",
    "D_5d_relative": "#d62728",
}

X_TICK_LABELSIZE = 11


def _style_target_x_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", rotation=0, labelsize=X_TICK_LABELSIZE)


def _target_order(present: list[str] | pd.Index | set[str]) -> list[str]:
    present_set = set(present)
    ordered = [t for t in TARGET_ORDER if t in present_set]
    extras = sorted(present_set - set(ordered))
    return ordered + extras


def _target_colors(targets: list[str]) -> list[str]:
    return [TARGET_COLORS.get(t, "#7f7f7f") for t in targets]


def _target_labels(targets: list[str]) -> list[str]:
    return [TARGET_LABELS.get(t, t) for t in targets]


# --- Loaders -----------------------------------------------------------


def load_target_metrics(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "metrics_by_target_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics table: {path}")
    return pd.read_csv(path)


def load_target_cross_sectional(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "cross_sectional_by_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-sectional table: {path}")
    return pd.read_csv(path)


def load_target_trading(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "trading_by_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trading table: {path}")
    return pd.read_csv(path)


def load_target_returns(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "returns_by_target.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing returns table: {path}")
    return pd.read_csv(path)


def load_target_deltas(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "deltas_vs_baseline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing deltas table: {path}")
    return pd.read_csv(path)


def load_feature_importance_by_target(
    results_dir: str | Path,
    targets: tuple[str, ...] = TARGET_ORDER,
) -> dict[str, dict[str, float]]:
    results_dir = Path(results_dir)
    out: dict[str, dict[str, float]] = {}
    for target in targets:
        path = results_dir / target / "feature_importance.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            out[target] = {str(k): float(v) for k, v in json.load(handle).items()}
    return out


# --- Plots ---------------------------------------------------------------


def plot_target_comparison_cross_sectional(
    cs_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Bars: IC overall, mean daily IC, and top-decile hit rate by target.

    All computed against the common absolute-5-day yardstick.
    """
    apply_style()
    frame = cs_df[cs_df["split"] == split] if "split" in cs_df.columns else cs_df
    if frame.empty:
        raise ValueError(f"No cross-sectional metrics found for split={split!r}")
    cs = frame.set_index("target")
    target_order = _target_order(cs.index)

    metrics_to_plot = [
        ("ic_overall", "IC (overall)", 0.0, True),
        ("ic_mean_daily", "IC (mean daily)", 0.0, True),
        ("top_decile_hit_rate", "Top-decile hit rate", 0.2, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, baseline, higher_better) in zip(axes, metrics_to_plot, strict=True):
        values = [float(cs.loc[t, key]) for t in target_order]
        colors = _target_colors(target_order)
        bars = ax.bar(_target_labels(target_order), values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(baseline, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(metric_title(title, higher_better=higher_better))
        _style_target_x_axis(ax)

    fig.suptitle(
        f"Experiment 3 — Cross-Sectional Ranking by Target "
        f"({split.title()}, common 5d yardstick)",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_target_comparison_trading(
    trading_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Bars: Sharpe, annualized return, and max drawdown by target."""
    apply_style()
    subset = trading_df[trading_df["split"] == split].set_index("target")
    if subset.empty:
        raise ValueError(f"No trading metrics found for split={split!r}")
    target_order = _target_order(subset.index)

    metrics_to_plot = [
        ("sharpe_ratio", "Sharpe ratio", "{:.2f}", True),
        ("annualized_return", "Annualized return", "{:.1%}", True),
        ("max_drawdown", "Max drawdown", "{:.1%}", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, fmt, higher_better) in zip(axes, metrics_to_plot, strict=True):
        values = [float(subset.loc[t, key]) for t in target_order]
        colors = _target_colors(target_order)
        bars = ax.bar(_target_labels(target_order), values, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=8)
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(metric_title(title, higher_better=higher_better))
        _style_target_x_axis(ax)

    fig.suptitle(
        f"Experiment 3 — Long/Short Trading by Target ({split.title()})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_target_decile_spread(
    returns_df: pd.DataFrame,
    out_path: str | Path,
    split: str = "test",
) -> Path:
    """Bars: top-minus-bottom prediction-decile spread vs the common 5d return."""
    apply_style()
    frame = returns_df[returns_df["split"] == split] if "split" in returns_df.columns else returns_df
    if frame.empty:
        raise ValueError(f"No decile-return metrics found for split={split!r}")
    rets = frame.set_index("target")
    target_order = _target_order(rets.index)

    values = [float(rets.loc[t, "top_minus_bottom"]) for t in target_order]
    colors = _target_colors(target_order)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(_target_labels(target_order), values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_title(
        metric_title(
            f"Top-minus-bottom decile return ({split.title()}, common 5d)",
            higher_better=True,
        )
    )
    _style_target_x_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_feature_importance_by_target(
    importance_by_target: dict[str, dict[str, float]],
    out_path: str | Path,
) -> Path:
    """Heatmap of feature importance (rows=targets, cols=features)."""
    apply_style()
    targets = _target_order(importance_by_target.keys())
    if not targets:
        raise ValueError("No feature importance data to plot")

    all_features: list[str] = []
    for target in targets:
        for feat in importance_by_target[target]:
            if feat not in all_features:
                all_features.append(feat)

    matrix = np.array(
        [[importance_by_target[t].get(f, 0.0) for f in all_features] for t in targets]
    )

    fig, ax = plt.subplots(figsize=(max(8, len(all_features) * 1.4), 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(all_features)))
    ax.set_xticklabels(all_features, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([TARGET_LABELS.get(t, t).replace("\n", " ") for t in targets], fontsize=10)
    for i in range(len(targets)):
        for j in range(len(all_features)):
            ax.text(
                j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                color="white" if matrix[i, j] < matrix.max() * 0.6 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="XGBoost gain importance")
    ax.set_title("Experiment 3 — Feature Importance by Target", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def plot_delta_vs_baseline(
    deltas_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Grouped bars: Delta Sharpe / Delta ann. return / Delta mean-daily IC vs baseline."""
    apply_style()
    if deltas_df.empty:
        raise ValueError("No delta-vs-baseline metrics to plot")
    frame = deltas_df.set_index("target")
    target_order = _target_order(frame.index)

    materiality = {
        "delta_sharpe": 0.10,
        "delta_annual_return": 0.02,
        "delta_mean_daily_ic": 0.005,
    }
    metrics_to_plot = [
        ("delta_sharpe", "Δ Sharpe (test) vs baseline", materiality["delta_sharpe"], "{:.2f}"),
        (
            "delta_annual_return",
            "Δ Annualized return (test) vs baseline",
            materiality["delta_annual_return"],
            "{:.1%}",
        ),
        (
            "delta_mean_daily_ic",
            "Δ Mean-daily IC (test) vs baseline",
            materiality["delta_mean_daily_ic"],
            "{:.4f}",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (key, title, threshold, fmt) in zip(axes, metrics_to_plot, strict=True):
        values = [float(frame.loc[t, key]) if pd.notna(frame.loc[t, key]) else 0.0 for t in target_order]
        colors = _target_colors(target_order)
        bars = ax.bar(_target_labels(target_order), values, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=9)
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="-", linewidth=1)
        ax.axhline(threshold, color="#888888", linestyle="--", linewidth=1)
        ax.axhline(-threshold, color="#888888", linestyle="--", linewidth=1)
        ax.set_title(title)
        _style_target_x_axis(ax)

    fig.suptitle(
        "Experiment 3 — Materiality of Target Changes vs Baseline "
        "(dashed lines = materiality threshold)",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def generate_experiment3_figures(results_dir: str | Path) -> list[Path]:
    """Write Experiment 3 comparison figures into ``results_dir/figures``."""
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    trading_df = load_target_trading(results_dir)
    cs_df = load_target_cross_sectional(results_dir)

    written: list[Path] = []

    written.append(
        plot_target_comparison_trading(
            trading_df, figures_dir / "target_comparison_trading_test.png", split="test"
        )
    )
    if not trading_df[trading_df["split"] == "val"].empty:
        written.append(
            plot_target_comparison_trading(
                trading_df, figures_dir / "target_comparison_trading_val.png", split="val"
            )
        )

    written.append(
        plot_target_comparison_cross_sectional(
            cs_df, figures_dir / "target_comparison_cross_sectional_test.png", split="test"
        )
    )
    if not cs_df[cs_df["split"] == "val"].empty:
        written.append(
            plot_target_comparison_cross_sectional(
                cs_df, figures_dir / "target_comparison_cross_sectional_val.png", split="val"
            )
        )

    try:
        returns_df = load_target_returns(results_dir)
        written.append(
            plot_target_decile_spread(
                returns_df, figures_dir / "target_decile_spread_test.png", split="test"
            )
        )
    except (FileNotFoundError, ValueError):
        pass

    importance_by_target = load_feature_importance_by_target(results_dir)
    if importance_by_target:
        written.append(
            plot_feature_importance_by_target(
                importance_by_target, figures_dir / "feature_importance_by_target.png"
            )
        )

    try:
        deltas_df = load_target_deltas(results_dir)
        written.append(
            plot_delta_vs_baseline(deltas_df, figures_dir / "delta_vs_baseline.png")
        )
    except (FileNotFoundError, ValueError):
        pass

    return written
