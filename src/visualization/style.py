"""Shared matplotlib/seaborn styling so figures stay comparable across runs.

Every plotting function in :mod:`src.visualization` should call :func:`apply_style`
before drawing and pull colors from :data:`SPLIT_COLORS` / :data:`SERIES_COLORS`
instead of hard-coding hex values. This keeps a figure from one training run
(e.g. xgboost) visually consistent with the same figure from another run
(e.g. lightgbm), so they can be placed side by side in a presentation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

FIGURE_DPI = 150
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 6)
FIGSIZE_TALL = (9, 8)

# Consistent colors for recurring series across every figure in the report.
SPLIT_COLORS: dict[str, str] = {
    "train": "#2ca02c",
    "val": "#1f77b4",
    "test": "#d62728",
}
SERIES_COLORS: dict[str, str] = {
    "strategy": "#1f77b4",
    "benchmark": "#7f7f7f",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
}
REFERENCE_LINE_COLOR = "#444444"


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib theme used by all report figures."""
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "font.size": 11,
    })


def save_figure(fig: plt.Figure, out_path) -> None:
    """Save a figure to disk, creating parent directories as needed.

    Args:
        fig: Matplotlib figure to persist.
        out_path: Destination path (str or Path) for the PNG file.
    """
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
