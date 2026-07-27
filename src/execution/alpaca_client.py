"""Alpaca trading API client wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.utils.config import AlpacaConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Simplified position representation."""

    symbol: str
    qty: float
    market_value: float
    current_price: float


class AlpacaClient:
    """Thin wrapper around Alpaca paper trading client."""

    def __init__(self, config: AlpacaConfig) -> None:
        """Initialize Alpaca trading client.

        Args:
            config: Alpaca credentials and base URL.

        Raises:
            ValueError: If credentials missing or not paper trading URL.
        """
        if not config.api_key or not config.secret_key:
            raise ValueError("Alpaca API credentials are required")
        if "paper" not in config.base_url:
            raise ValueError("Live trading is disabled in MVP. Use paper API URL.")

        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper="paper" in config.base_url,
        )
        logger.info("Alpaca client initialized (paper=%s)", "paper" in config.base_url)

    def get_account_equity(self) -> float:
        """Return current account equity.

        Returns:
            Total account equity in USD.
        """
        account = self._client.get_account()
        return float(account.equity)

    def get_positions(self) -> list[Position]:
        """Return current open positions.

        Returns:
            List of Position objects.
        """
        positions = self._client.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                current_price=float(p.current_price),
            )
            for p in positions
        ]

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        dry_run: bool = True,
    ) -> str | None:
        """Submit a market order.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (fractional supported on Alpaca).
            side: OrderSide.BUY or OrderSide.SELL.
            dry_run: If True, log order without submitting.

        Returns:
            Order ID if submitted, else None.
        """
        if qty <= 0:
            return None

        if dry_run:
            logger.info("[DRY RUN] %s %s %.4f shares", side.value, symbol, qty)
            return None

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        logger.info("Submitted order %s: %s %s %.4f", order.id, side.value, symbol, qty)
        return str(order.id)

    def get_recent_orders(self, limit: int = 20) -> list[dict]:
        """Fetch recent orders for dashboard display.

        Args:
            limit: Maximum number of orders to return.

        Returns:
            List of order summary dictionaries.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        orders = self._client.get_orders(request)
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side.value),
                "qty": float(o.qty) if o.qty else 0,
                "status": str(o.status.value),
                "submitted_at": str(o.submitted_at),
            }
            for o in orders
        ]
