"""Paper trading orchestration."""

from __future__ import annotations

import pandas as pd

from src.data.storage import load_raw_bars
from src.data.universe import load_universe
from src.execution.alpaca_client import AlpacaClient
from src.execution.order_manager import compute_order_intents, execute_orders
from src.features.engineer import engineer_features
from src.models.registry import load_latest_model
from src.strategy.portfolio import build_portfolio_signals, get_latest_targets
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _generate_live_predictions(config: AppConfig) -> pd.DataFrame:
    """Generate predictions for the latest available feature rows per symbol.

    Uses the single pooled model so probabilities are comparable across the
    universe for cross-sectional ranking.

    Args:
        config: Application configuration.

    Returns:
        DataFrame with date, symbol, probability, close columns.
    """
    pipeline, artifact = load_latest_model()
    universe = load_universe(config.data.universe_file)
    prediction_rows: list[dict[str, object]] = []

    for symbol in universe.tickers:
        bars = load_raw_bars(symbol)
        if bars is None or bars.empty:
            logger.warning("No raw bars found for %s; skipping prediction", symbol)
            continue
        features = engineer_features(
            bars,
            indicator_set=config.features.indicator_set,
        )
        usable = features.dropna(subset=artifact.feature_columns)
        if usable.empty:
            logger.warning("No complete feature row found for %s", symbol)
            continue

        latest = usable.iloc[-1]
        feature_values = latest[artifact.feature_columns].to_numpy().reshape(1, -1)
        probability = float(pipeline.predict_proba(feature_values)[0, 1])
        prediction_rows.append({
            "date": usable.index[-1],
            "symbol": symbol,
            "close": latest["close"],
            "probability": probability,
            "prediction": int(probability >= config.strategy.entry_threshold),
        })

    if not prediction_rows:
        raise ValueError("No latest rows could be scored by the pooled model")
    result = pd.DataFrame(prediction_rows)
    result["predicted_rank"] = (
        result.groupby("date")["probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result


def run_paper_trading(config: AppConfig) -> None:
    """Run paper trading rebalance based on latest model predictions.

    Args:
        config: Application configuration.
    """
    if config.execution.paper and "paper" not in config.alpaca.base_url:
        raise ValueError("Paper trading requires ALPACA_BASE_URL to contain 'paper'")

    predictions = _generate_live_predictions(config)
    portfolio = build_portfolio_signals(predictions, config.strategy)
    targets = get_latest_targets(portfolio)

    if targets.empty:
        logger.info("No target positions for latest date")
        return

    client = AlpacaClient(config.alpaca)
    intents = compute_order_intents(
        targets,
        client,
        max_order_value=config.execution.max_order_value,
    )

    execute_orders(intents, client, dry_run=config.execution.dry_run)
    logger.info("Paper trading run complete")
