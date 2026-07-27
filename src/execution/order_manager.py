"""Order management for portfolio rebalancing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from alpaca.trading.enums import OrderSide

from src.execution.alpaca_client import AlpacaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderIntent:
    """Planned order before submission."""

    symbol: str
    side: OrderSide
    qty: float
    target_value: float


def compute_order_intents(
    targets: pd.DataFrame,
    client: AlpacaClient,
    max_order_value: float,
) -> list[OrderIntent]:
    """Compute orders needed to reach target portfolio weights.

    Args:
        targets: DataFrame with symbol and weight columns.
        client: Alpaca client for current positions and equity.
        max_order_value: Maximum dollar value per order.

    Returns:
        List of OrderIntent objects.
    """
    equity = client.get_account_equity()
    current_positions = {p.symbol: p.qty for p in client.get_positions()}
    target_symbols = set(targets["symbol"].tolist())

    intents: list[OrderIntent] = []

    for _, row in targets.iterrows():
        symbol = row["symbol"]
        target_value = max(-max_order_value, min(equity * row["weight"], max_order_value))
        current_qty = current_positions.get(symbol, 0.0)

        from src.data.storage import load_raw_bars
        bars = load_raw_bars(symbol)
        if bars is None or bars.empty:
            logger.warning("No price data for %s, skipping", symbol)
            continue

        price = float(bars["close"].iloc[-1])
        if price <= 0:
            continue

        target_qty = target_value / price
        delta = target_qty - current_qty

        if abs(delta * price) < 1.0:
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        intents.append(OrderIntent(
            symbol=symbol,
            side=side,
            qty=abs(delta),
            target_value=target_value,
        ))

    for symbol, qty in current_positions.items():
        if symbol not in target_symbols and qty != 0:
            from src.data.storage import load_raw_bars
            bars = load_raw_bars(symbol)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0
            # Positive qty = existing long to sell off; negative qty = existing
            # short to buy back (cover).
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            intents.append(OrderIntent(
                symbol=symbol,
                side=side,
                qty=abs(qty),
                target_value=qty * price,
            ))

    return intents


def execute_orders(
    intents: list[OrderIntent],
    client: AlpacaClient,
    dry_run: bool = True,
) -> list[str | None]:
    """Submit order intents to Alpaca.

    Args:
        intents: List of planned orders.
        client: Alpaca trading client.
        dry_run: If True, log without submitting.

    Returns:
        List of order IDs (None for dry-run or skipped).
    """
    order_ids: list[str | None] = []
    for intent in intents:
        order_id = client.submit_market_order(
            symbol=intent.symbol,
            qty=intent.qty,
            side=intent.side,
            dry_run=dry_run,
        )
        order_ids.append(order_id)
    logger.info("Processed %d orders (dry_run=%s)", len(intents), dry_run)
    return order_ids
