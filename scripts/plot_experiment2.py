"""Regenerate Experiment 2 presentation figures from archived results.

Does not download data, rebuild features, or retrain models. Reads CSVs and
manifests under results/experiment_2/ and writes comparison PNGs into
results/experiment_2/figures/.

Usage:
    python scripts/plot_experiment2.py
    python scripts/plot_experiment2.py --results-dir results/experiment_2
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
from src.visualization.experiment2_plots import generate_experiment2_figures

logger = get_logger(__name__)

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Experiment 2 presentation figures from archived results",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results/experiment_2 (or a fixture directory)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(level="INFO")
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    written = generate_experiment2_figures(results_dir)
    logger.info("Wrote %d figure(s) under %s/figures", len(written), results_dir)
    for path in written:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
