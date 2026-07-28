"""Regenerate Experiment 1 presentation figures from archived results.

Does not download data, rebuild features, or retrain models. Reads CSVs and
manifests under results/experiment_1/ and writes comparison PNGs into
results/experiment_1/figures/. When local prediction parquets exist, also
regenerates selected per-model figures.

Usage:
    python scripts/plot_experiment1.py
    python scripts/plot_experiment1.py --results-dir results/experiment_1
    python scripts/plot_experiment1.py --no-per-model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging import get_logger, setup_logging
from src.utils.paths import PROJECT_ROOT
from src.visualization.experiment1_plots import generate_experiment1_figures

logger = get_logger(__name__)

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate Experiment 1 presentation figures from archived results",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results/experiment_1 (or a fixture directory)",
    )
    parser.add_argument(
        "--no-per-model",
        action="store_true",
        help="Skip optional per-model figure regeneration",
    )
    return parser


def main() -> None:
    setup_logging(level="INFO")
    args = _build_parser().parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    written = generate_experiment1_figures(
        results_dir,
        regenerate_per_model=not args.no_per_model,
    )
    logger.info("Wrote %d figure(s) under %s", len(written), results_dir / "figures")
    for path in written:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
