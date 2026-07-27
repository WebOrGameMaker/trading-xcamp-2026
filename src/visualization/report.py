"""Orchestrates the full train/test + backtest visualization report.

Running ``python main.py visualize`` calls :func:`generate_report`, which
loads whatever artifacts are available under ``logs/`` and ``data/processed/``
and writes a fixed, numbered set of PNGs to ``logs/figures/{run_id}/`` so the
same figure name always means the same chart across runs — making it
straightforward to compare two runs (e.g. two model types) side by side in a
presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.visualization import backtest_plots, classification_plots, loaders

logger = get_logger(__name__)


@dataclass
class ReportResult:
    """Summary of what a report run produced."""

    output_dir: Path
    written: list[Path]
    skipped: list[str]


def generate_report(config: AppConfig, run_id: str | None = None) -> ReportResult:
    """Generate the full set of presentation-ready result figures.

    Each section is independent and best-effort: if a prerequisite artifact
    (e.g. predictions parquet, feature importance, equity curve) is missing,
    that figure is skipped with a logged reason instead of failing the whole
    report.

    Args:
        config: Application configuration (used for the benchmark symbol and
            initial cash when recomputing the benchmark equity curve).
        run_id: Specific training run to report on. Defaults to the most
            recent run found in logs/eval_*.json.

    Returns:
        ReportResult with the output directory and lists of written/skipped figures.
    """
    resolved_run_id = run_id or loaders.latest_run_id()
    out_dir = loaders.resolve_figures_dir(resolved_run_id)
    written: list[Path] = []
    skipped: list[str] = []

    eval_df = None
    try:
        eval_df = loaders.load_eval_reports(resolved_run_id)
    except FileNotFoundError as exc:
        skipped.append(f"classification metrics (01-03): {exc}")

    if eval_df is not None:
        written.append(classification_plots.plot_metric_overview(
            eval_df, out_dir / "01_classification_metric_overview.png"
        ))
        written.append(classification_plots.plot_roc_auc_distribution(
            eval_df, out_dir / "02_roc_auc_distribution.png"
        ))
        written.append(classification_plots.plot_confusion_matrices(
            eval_df, out_dir / "03_confusion_matrices.png"
        ))

    try:
        pred_val = loaders.load_predictions("val")
        pred_test = loaders.load_predictions("test")
    except FileNotFoundError as exc:
        skipped.append(f"probability separation / calibration (04, 09): {exc}")
        pred_val = None
        pred_test = None
    else:
        written.append(
            classification_plots.plot_probability_separation(
                pred_val, pred_test, out_dir / "04_probability_separation.png"
            )
        )
        try:
            written.append(
                classification_plots.plot_calibration_curves(
                    {"val": pred_val, "test": pred_test},
                    out_dir / "09_calibration_curve.png",
                )
            )
        except ValueError as exc:
            skipped.append(f"calibration curve (09): {exc}")

    importances = loaders.load_feature_importance()
    if importances:
        written.append(classification_plots.plot_feature_importance(
            importances, out_dir / "05_feature_importance.png"
        ))
    else:
        skipped.append(
            "feature importance (05): models/latest.json not found or has no feature_importance"
        )

    equity = None
    try:
        equity = loaders.load_equity_curve()
    except FileNotFoundError as exc:
        skipped.append(f"equity curve / drawdown / rolling sharpe (06-07): {exc}")

    if equity is not None:
        benchmark = loaders.load_benchmark_equity(config, equity.index)
        written.append(backtest_plots.plot_equity_and_drawdown(
            equity, benchmark, out_dir / "06_equity_and_drawdown.png"
        ))
        written.append(
            backtest_plots.plot_rolling_sharpe(equity, out_dir / "07_rolling_sharpe.png")
        )

    try:
        backtest_metrics = loaders.load_backtest_metrics()
    except FileNotFoundError as exc:
        skipped.append(f"backtest scorecard (08): {exc}")
    else:
        written.append(backtest_plots.plot_backtest_scorecard(
            backtest_metrics, out_dir / "08_backtest_scorecard.png"
        ))

    for reason in skipped:
        logger.warning("Skipped figure — %s", reason)
    logger.info("Wrote %d figures to %s", len(written), out_dir)

    return ReportResult(output_dir=out_dir, written=written, skipped=skipped)
