"""Multi-target feature dataset for Experiment 3 (H3 — target engineering).

Builds one pooled panel carrying forward returns at three horizons (3d, 5d,
10d) plus a cross-sectional relative 5d target, alongside a binary
evaluation label fixed at the 5-day horizon so every target arm can be
scored on an identical ranking yardstick regardless of what it was trained
to predict. Writes to ``data/processed/features_experiment3.parquet`` and
never touches ``features.parquet`` (used by Experiment 1/2), so prior
experiment artifacts stay reproducible.
"""

from __future__ import annotations

import pandas as pd

from src.data.storage import load_all_raw_bars, save_processed_dataset
from src.data.universe import load_universe
from src.features.engineer import engineer_features
from src.features.labels import (
    assign_cross_sectional_labels,
    compute_forward_returns,
    compute_relative_forward_returns,
    forward_return_column,
)
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

EVAL_HORIZON_DAYS = 5
TARGET_HORIZONS: tuple[int, ...] = (3, 5, 10)
DATASET_NAME = "features_experiment3"


def build_experiment3_feature_dataset(config: AppConfig) -> pd.DataFrame:
    """Build the pooled multi-target panel used by all Experiment 3 arms.

    Args:
        config: Application configuration (universe, label quantile settings).

    Returns:
        Processed feature DataFrame with ``forward_return_3d``,
        ``forward_return_5d``, ``forward_return_10d``,
        ``forward_return_5d_rel``, and a fixed 5-day binary ``label`` column.

    Raises:
        FileNotFoundError: If no cached raw OHLCV data is available.
    """
    universe = load_universe(config.data.universe_file)
    raw_data = load_all_raw_bars(universe.tickers)

    if not raw_data:
        raise FileNotFoundError(
            "No raw data found. Run 'python main.py download' first."
        )

    frames: list[pd.DataFrame] = []
    for symbol, ohlcv in raw_data.items():
        features = engineer_features(ohlcv, indicator_set=config.features.indicator_set)
        features["date"] = features.index
        for horizon in TARGET_HORIZONS:
            features = compute_forward_returns(features, horizon_days=horizon)
        features["symbol"] = symbol
        frames.append(features)
        logger.info("Experiment 3 features for %s: %d rows", symbol, len(features))

    dataset = pd.concat(frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Cross-sectional relative target derives from the absolute 5d column,
    # which every symbol already carries from the per-symbol loop above.
    dataset = compute_relative_forward_returns(dataset, horizon_days=EVAL_HORIZON_DAYS)

    # Common evaluation label pinned to the 5-day horizon regardless of which
    # column a given arm trains on, so ROC-AUC/PR-AUC/hit-rate are comparable
    # across all four targets.
    dataset = assign_cross_sectional_labels(
        dataset,
        horizon_days=EVAL_HORIZON_DAYS,
        positive_quantile=config.labels.positive_quantile,
        min_names=config.labels.min_cross_section,
    )

    # Require every target column (3d/5d/10d/5d_rel) plus the common label to
    # be valid on every retained row. This costs a handful of extra trading
    # dates at the very end of the panel (10d tail) versus dropping only on
    # each arm's own target, but guarantees all four arms train/evaluate on
    # an identical row set -- the strongest form of "identical splits."
    target_cols = [forward_return_column(h) for h in TARGET_HORIZONS]
    target_cols.append(f"{forward_return_column(EVAL_HORIZON_DAYS)}_rel")
    feature_cols = [
        c
        for c in dataset.columns
        if c not in {"symbol", "date", "label", "close", *target_cols}
    ]
    dataset = dataset.dropna(subset=["label", *target_cols, *feature_cols]).reset_index(
        drop=True
    )

    path = save_processed_dataset(dataset, name=DATASET_NAME)
    logger.info(
        "Saved Experiment 3 multi-target dataset: %d rows, %d symbols -> %s",
        len(dataset),
        dataset["symbol"].nunique(),
        path,
    )
    return dataset
