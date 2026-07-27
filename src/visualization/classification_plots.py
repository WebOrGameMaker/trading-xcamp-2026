"""Figures summarizing per-symbol classifier skill on the val/test splits.

These answer the "does the model actually work, and does it generalize from
validation to the out-of-sample test period" questions that matter most when
comparing modeling strategies (e.g. xgboost vs lightgbm, or different feature
sets) across separate training runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    FIGSIZE_SQUARE,
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    REFERENCE_LINE_COLOR,
    SERIES_COLORS,
    SPLIT_COLORS,
    apply_style,
    save_figure,
)

METRIC_ORDER = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier_score": "Brier",
}

PROBA_LABELS = {
    "probability": "Raw",
    "probability_platt": "Platt",
    "probability_isotonic": "Isotonic",
}
PROBA_COLORS = {
    "probability": "#1f77b4",
    "probability_platt": "#ff7f0e",
    "probability_isotonic": "#2ca02c",
}


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float((values * weights).sum() / total_weight)


def plot_metric_overview(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Grouped bar chart of support-weighted classification metrics, val vs test.

    This is the primary "is the model skillful, and does it generalize" figure:
    a shrinking gap between val and test bars indicates the model holds up
    out-of-sample; a large gap indicates overfitting to the validation period.

    Args:
        df: Tidy eval-report frame from ``loaders.load_eval_reports``.
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    splits = [s for s in ("train", "val", "test") if s in df["split"].unique()]
    metrics = tuple(m for m in METRIC_ORDER if m in df.columns)

    summary = {
        split: {
            metric: _weighted_mean(
                df.loc[df["split"] == split, metric],
                df.loc[df["split"] == split, "support"],
            )
            for metric in metrics
        }
        for split in splits
    }

    x = np.arange(len(metrics))
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, split in enumerate(splits):
        offsets = x + (i - (len(splits) - 1) / 2) * width
        heights = [summary[split][m] for m in metrics]
        bars = ax.bar(
            offsets, heights, width=width, label=split.title(), color=SPLIT_COLORS.get(split)
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.axhline(
        0.5, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1, label="Random baseline (0.5)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    # Brier is typically <0.5; keep room above 1.0 for classification scores.
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    n_symbols = df["symbol"].nunique()
    split_label = " vs ".join(s.title() for s in splits)
    ax.set_title(f"Classification Metrics — {split_label} ({n_symbols} symbols, support-weighted)")
    ax.legend(loc="upper right", framealpha=0.9)

    save_figure(fig, out_path)
    return Path(out_path)


def plot_calibration_curves(
    predictions_by_split: dict[str, pd.DataFrame],
    out_path: str | Path,
    n_bins: int = 10,
    probability_columns: tuple[str, ...] | None = None,
) -> Path:
    """Reliability diagrams for raw / Platt / isotonic probabilities.

    Args:
        predictions_by_split: Mapping of split name -> predictions DataFrame
            with ``label`` and probability columns.
        out_path: Destination PNG path.
        n_bins: Equal-width bins for the reliability curve.
        probability_columns: Probability columns to overlay. Defaults to
            whichever of raw/Platt/isotonic are present.

    Returns:
        The path the figure was saved to.
    """
    from src.models.evaluator import compute_reliability_bins

    apply_style()
    splits = [s for s in ("val", "test") if s in predictions_by_split]
    if not splits:
        splits = list(predictions_by_split.keys())

    sample = next(iter(predictions_by_split.values()))
    if probability_columns is None:
        probability_columns = tuple(
            col for col in (
                "probability",
                "probability_platt",
                "probability_isotonic",
            )
            if col in sample.columns
        )
    if not probability_columns:
        raise ValueError("No probability columns available for calibration plot")

    fig, axes = plt.subplots(1, len(splits), figsize=FIGSIZE_WIDE, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, split in zip(axes, splits, strict=True):
        pred = predictions_by_split[split]
        y_true = pred["label"].to_numpy()
        for col in probability_columns:
            if col not in pred.columns:
                continue
            bins = compute_reliability_bins(
                y_true, pred[col].to_numpy(dtype=float), n_bins=n_bins
            )
            if not bins:
                continue
            xs = [b["mean_predicted"] for b in bins]
            ys = [b["fraction_positive"] for b in bins]
            ax.plot(
                xs,
                ys,
                marker="o",
                label=PROBA_LABELS.get(col, col),
                color=PROBA_COLORS.get(col),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_LINE_COLOR, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(split.title())
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Fraction of positives")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.suptitle("Reliability Diagram — Raw vs Calibrated Probabilities", fontweight="bold")
    fig.tight_layout()

    save_figure(fig, out_path)
    return Path(out_path)


def plot_roc_auc_distribution(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Box + strip plot of per-symbol ROC-AUC, val vs test.

    A single aggregate ROC-AUC can hide huge variance across a 100+ symbol
    universe; this shows the full distribution so "the model works" claims
    can be checked against how consistent that skill is symbol-by-symbol.

    Args:
        df: Tidy eval-report frame from ``loaders.load_eval_reports``.
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    import seaborn as sns

    apply_style()
    splits = [s for s in ("train", "val", "test") if s in df["split"].unique()]
    palette = {s: SPLIT_COLORS.get(s) for s in splits}

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.boxplot(
        data=df,
        x="split",
        y="roc_auc",
        order=splits,
        hue="split",
        hue_order=splits,
        palette=palette,
        legend=False,
        showfliers=False,
        width=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="split",
        y="roc_auc",
        order=splits,
        color="black",
        alpha=0.35,
        size=3,
        jitter=0.2,
        ax=ax,
    )

    for i, split in enumerate(splits):
        median = df.loc[df["split"] == split, "roc_auc"].median()
        ax.text(
            i, median, f" median={median:.3f}",
            va="bottom", ha="center", fontsize=10, fontweight="bold",
        )

    ax.axhline(0.5, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
    ax.text(
        len(splits) - 0.5, 0.505, "random baseline",
        color=REFERENCE_LINE_COLOR, fontsize=9, ha="right",
    )
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.title() for s in splits])
    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC")
    n_symbols = df["symbol"].nunique()
    ax.set_title(f"Per-Symbol ROC-AUC Distribution ({n_symbols} symbols)")

    save_figure(fig, out_path)
    return Path(out_path)


def plot_confusion_matrices(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Aggregated, row-normalized confusion matrix heatmaps for val and test.

    Confusion matrices from every per-symbol model are summed so the figure
    reflects universe-wide behavior rather than a single ticker.

    Args:
        df: Tidy eval-report frame from ``loaders.load_eval_reports``.
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    import seaborn as sns

    apply_style()
    splits = [s for s in ("train", "val", "test") if s in df["split"].unique()]

    fig, axes = plt.subplots(1, len(splits), figsize=FIGSIZE_WIDE)
    axes = np.atleast_1d(axes)

    for ax, split in zip(axes, splits, strict=True):
        matrices = df.loc[df["split"] == split, "confusion_matrix"].tolist()
        aggregate = np.sum([np.array(m) for m in matrices], axis=0)
        row_sums = aggregate.sum(axis=1, keepdims=True)
        normalized = np.divide(
            aggregate, row_sums,
            out=np.zeros_like(aggregate, dtype=float),
            where=row_sums != 0,
        )

        annotations = np.array([
            f"{int(aggregate[r, c]):,}\n({normalized[r, c] * 100:.1f}%)"
            for r in range(2)
            for c in range(2)
        ]).reshape(2, 2)

        sns.heatmap(
            normalized,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"{split.title()} (row-normalized)")

    fig.suptitle("Aggregated Confusion Matrix — All Symbols Summed", fontweight="bold")
    fig.tight_layout()

    save_figure(fig, out_path)
    return Path(out_path)


def plot_probability_separation(
    pred_val: pd.DataFrame,
    pred_test: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Histogram of predicted probability split by true label, val vs test.

    Shows how well the model's raw probability output discriminates between
    the two classes — a well-separated pair of distributions indicates a
    genuinely informative model, while heavily overlapping distributions
    indicate the model is close to guessing despite decent point-metrics.

    Args:
        pred_val: Row-level val predictions with ``probability`` and ``label``.
        pred_test: Row-level test predictions with ``probability`` and ``label``.
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    bins = np.linspace(0, 1, 26)
    panels = [("Validation", pred_val), ("Test", pred_test)]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=False)
    for ax, (title, pred) in zip(axes, panels, strict=True):
        for label_value, label_name, color_key in (
            (0, "True label 0", "negative"),
            (1, "True label 1", "positive"),
        ):
            subset = pred.loc[pred["label"] == label_value, "probability"]
            ax.hist(
                subset,
                bins=bins,
                alpha=0.55,
                label=label_name,
                color=SERIES_COLORS[color_key],
                density=True,
            )
        ax.axvline(0.5, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1)

    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper center", framealpha=0.9)
    fig.suptitle("Predicted Probability Separation by True Label", fontweight="bold")
    fig.tight_layout()

    save_figure(fig, out_path)
    return Path(out_path)


def plot_feature_importance(
    importances: dict[str, float],
    out_path: str | Path,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of the top averaged feature importances.

    Args:
        importances: Feature name -> importance mapping (already averaged
            across symbols), as produced by the trainer.
        out_path: Destination PNG path.
        top_n: Number of top features to display.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    top_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    features = [k for k, _ in top_items][::-1]
    values = [v for _, v in top_items][::-1]

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    ax.barh(features, values, color=SPLIT_COLORS["val"])
    ax.set_xlabel("Average importance")
    ax.set_title(f"Top {len(top_items)} Feature Importances (Averaged Across Symbols)")

    save_figure(fig, out_path)
    return Path(out_path)
