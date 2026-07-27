"""End-to-end feature dataset construction."""

from __future__ import annotations

import pandas as pd

from src.data.storage import load_all_raw_bars, save_processed_dataset
from src.data.universe import load_universe
from src.features.engineer import engineer_features
from src.features.labels import (
    assign_cross_sectional_labels,
    compute_forward_returns,
    drop_unlabeled_rows,
    generate_labels,
)
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_feature_dataset(config: AppConfig) -> pd.DataFrame:
    """Build a combined feature dataset while preserving symbol boundaries.

    Features are engineered per symbol. Labels are either absolute (per-symbol
    threshold) or cross-sectional (pooled within-date quantile), depending on
    ``config.labels.mode``.

    Args:
        config: Application configuration.

    Returns:
        Processed feature DataFrame with labels.
    """
    universe = load_universe(config.data.universe_file)
    raw_data = load_all_raw_bars(universe.tickers)

    if not raw_data:
        raise FileNotFoundError(
            "No raw data found. Run 'python main.py download' first."
        )

    horizon = config.labels.horizon_days
    mode = config.labels.mode
    frames: list[pd.DataFrame] = []

    for symbol, ohlcv in raw_data.items():
        features = engineer_features(ohlcv, indicator_set=config.features.indicator_set)
        features["date"] = features.index
        # Always compute forward returns first; absolute mode also assigns labels
        # here, while cross-sectional mode waits until the panel is pooled.
        if mode == "absolute":
            labeled = generate_labels(
                features,
                horizon_days=horizon,
                threshold=config.labels.threshold,
            )
        else:
            labeled = compute_forward_returns(features, horizon_days=horizon)
        labeled["symbol"] = symbol
        frames.append(labeled)
        logger.info("Features for %s: %d rows", symbol, len(labeled))

    dataset = pd.concat(frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)

    if mode == "cross_sectional":
        dataset = assign_cross_sectional_labels(
            dataset,
            horizon_days=horizon,
            positive_quantile=config.labels.positive_quantile,
            min_names=config.labels.min_cross_section,
        )
    elif mode != "absolute":
        raise ValueError(f"Unknown labels.mode: {mode}")

    dataset = drop_unlabeled_rows(dataset, horizon_days=horizon)
    path = save_processed_dataset(dataset, name="features")
    logger.info(
        "Saved feature dataset: %d rows, %d symbols -> %s",
        len(dataset),
        dataset["symbol"].nunique(),
        path,
    )
    return dataset
