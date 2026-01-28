"""
Position Manager - Advanced position tracking and management
Handles position lifecycle, P&L tracking, and portfolio management
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionEntry:
    """Single position entry (for averaging)"""
    quantity: int
    price: float
    timestamp: datetime
    order_id: str


@dataclass
class ManagedPosition:
    """
    A managed position with full tracking capabilities.
    Supports partial fills, averaging, and detailed P&L.
    """
    symbol: str
    futu_code: str

    # Position state
    quantity: int = 0  # Positive for long, negative for short
    avg_cost: float = 0.0

    # Entry tracking
    entries: List[PositionEntry] = field(default_factory=list)

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Timestamps
    opened_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)

    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    trailing_stop_high: Optional[float] = None  # Highest price since entry (for trailing)

    # Stats
    total_entries: int = 0
    total_exits: int = 0
    max_quantity: int = 0
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def abs_quantity(self) -> int:
        return abs(self.quantity)

    @property
    def market_value(self) -> float:
        """Market value based on avg cost"""
        return self.abs_quantity * self.avg_cost

    @property
    def direction(self) -> str:
        if self.is_long:
            return "LONG"
        elif self.is_short:
            return "SHORT"
        return "FLAT"

    @property
    def holding_duration_minutes(self) -> int:
        if not self.opened_at:
            return 0
        return int((datetime.now() - self.opened_at).total_seconds() / 60)

    def update_price(self, current_price: float):
        """Update position with current market price"""
        if self.is_flat:
            self.unrealized_pnl = 0.0
            return

        if self.is_long:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.avg_cost - current_price) * self.abs_quantity

        self.last_update = datetime.now()

        # Track extremes
        self.max_unrealized_pnl = max(self.max_unrealized_pnl, self.unrealized_pnl)
        self.min_unrealized_pnl = min(self.min_unrealized_pnl, self.unrealized_pnl)

        # Update trailing stop
        if self.trailing_stop_pct and self.is_long:
            if self.trailing_stop_high is None or current_price > self.trailing_stop_high:
                self.trailing_stop_high = current_price
                self.stop_loss_price = current_price * (1 - self.trailing_stop_pct)

    def add_entry(
        self,
        quantity: int,
        price: float,
        order_id: str,
        timestamp: datetime = None
    ):
        """Add to position (for averaging)"""
        timestamp = timestamp or datetime.now()

        if self.is_flat:
            self.opened_at = timestamp

        entry = PositionEntry(
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            order_id=order_id
        )
        self.entries.append(entry)

        # Calculate new average
        old_value = self.avg_cost * self.abs_quantity
        new_value = price * quantity

        self.quantity += quantity

        if self.abs_quantity > 0:
            self.avg_cost = (old_value + new_value) / self.abs_quantity

        self.total_entries += 1
        self.max_quantity = max(self.max_quantity, self.abs_quantity)

        logger.info(f"Position entry: {self.direction} {quantity} {self.symbol} @ ${price:.2f}")

    def reduce(
        self,
        quantity: int,
        price: float,
        order_id: str,
        timestamp: datetime = None
    ) -> float:
        """
        Reduce position and calculate realized P&L.

        Args:
            quantity: Number of shares to reduce (positive)
            price: Exit price
            order_id: Order ID for tracking
            timestamp: Exit timestamp

        Returns:
            Realized P&L for this reduction
        """
        timestamp = timestamp or datetime.now()

        if self.is_flat:
            return 0.0

        reduce_qty = min(quantity, self.abs_quantity)

        # Calculate P&L
        if self.is_long:
            pnl = (price - self.avg_cost) * reduce_qty
            self.quantity -= reduce_qty
        else:
            pnl = (self.avg_cost - price) * reduce_qty
            self.quantity += reduce_qty

        self.realized_pnl += pnl
        self.total_exits += 1
        self.last_update = timestamp

        if self.is_flat:
            self.opened_at = None
            self.entries.clear()
            self.trailing_stop_high = None

        logger.info(f"Position reduce: {reduce_qty} {self.symbol} @ ${price:.2f}, PnL: ${pnl:+.2f}")

        return pnl

    def set_stop_loss(self, price: float = None, pct: float = None, atr: float = None, multiplier: float = 2.0):
        """Set stop loss price"""
        if price:
            self.stop_loss_price = price
        elif pct and self.avg_cost > 0:
            if self.is_long:
                self.stop_loss_price = self.avg_cost * (1 - pct)
            else:
                self.stop_loss_price = self.avg_cost * (1 + pct)
        elif atr and self.avg_cost > 0:
            if self.is_long:
                self.stop_loss_price = self.avg_cost - (atr * multiplier)
            else:
                self.stop_loss_price = self.avg_cost + (atr * multiplier)

    def set_take_profit(self, price: float = None, pct: float = None, atr: float = None, multiplier: float = 3.0):
        """Set take profit price"""
        if price:
            self.take_profit_price = price
        elif pct and self.avg_cost > 0:
            if self.is_long:
                self.take_profit_price = self.avg_cost * (1 + pct)
            else:
                self.take_profit_price = self.avg_cost * (1 - pct)
        elif atr and self.avg_cost > 0:
            if self.is_long:
                self.take_profit_price = self.avg_cost + (atr * multiplier)
            else:
                self.take_profit_price = self.avg_cost - (atr * multiplier)

    def set_trailing_stop(self, pct: float, current_price: float = None):
        """Set trailing stop"""
        self.trailing_stop_pct = pct
        if current_price:
            self.trailing_stop_high = current_price
            self.stop_loss_price = current_price * (1 - pct)

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if self.stop_loss_price is None or self.is_flat:
            return False

        if self.is_long:
            return current_price <= self.stop_loss_price
        else:
            return current_price >= self.stop_loss_price

    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if self.take_profit_price is None or self.is_flat:
            return False

        if self.is_long:
            return current_price >= self.take_profit_price
        else:
            return current_price <= self.take_profit_price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "futu_code": self.futu_code,
            "direction": self.direction,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "total_pnl": round(self.realized_pnl + self.unrealized_pnl, 2),
            "holding_minutes": self.holding_duration_minutes,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "total_entries": self.total_entries,
            "max_quantity": self.max_quantity,
        }


class PositionManager:
    """
    Manages all positions with thread-safe operations.
    Provides portfolio-level tracking and risk monitoring.
    """

    def __init__(self, starting_cash: float = 100000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash

        self._positions: Dict[str, ManagedPosition] = {}
        self._lock = threading.RLock()

        # Portfolio tracking
        self._total_realized_pnl = 0.0
        self._peak_equity = starting_cash

        # Trade history
        self._closed_positions: List[ManagedPosition] = []

    def get_position(self, futu_code: str) -> ManagedPosition:
        """Get or create position for symbol"""
        with self._lock:
            if futu_code not in self._positions:
                symbol = futu_code.split('.')[-1] if '.' in futu_code else futu_code
                self._positions[futu_code] = ManagedPosition(
                    symbol=symbol,
                    futu_code=futu_code
                )
            return self._positions[futu_code]

    def open_position(
        self,
        futu_code: str,
        quantity: int,
        price: float,
        order_id: str,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ) -> ManagedPosition:
        """Open or add to a position"""
        with self._lock:
            position = self.get_position(futu_code)

            # Deduct cash
            cost = quantity * price
            self.cash -= cost

            # Add entry
            position.add_entry(quantity, price, order_id)

            # Set risk levels
            if stop_loss_pct:
                position.set_stop_loss(pct=stop_loss_pct)
            if take_profit_pct:
                position.set_take_profit(pct=take_profit_pct)

            return position

    def close_position(
        self,
        futu_code: str,
        quantity: int,
        price: float,
        order_id: str,
    ) -> float:
        """Close or reduce a position"""
        with self._lock:
            position = self.get_position(futu_code)

            # Calculate P&L
            pnl = position.reduce(quantity, price, order_id)

            # Add cash back
            proceeds = quantity * price
            self.cash += proceeds

            # Track realized P&L
            self._total_realized_pnl += pnl

            # If fully closed, archive
            if position.is_flat:
                self._closed_positions.append(position)
                del self._positions[futu_code]

            return pnl

    def close_all_positions(self, prices: Dict[str, float]) -> float:
        """Close all positions at given prices"""
        total_pnl = 0.0

        with self._lock:
            for futu_code, position in list(self._positions.items()):
                if not position.is_flat:
                    price = prices.get(futu_code, position.avg_cost)
                    pnl = self.close_position(
                        futu_code,
                        position.abs_quantity,
                        price,
                        f"close-all-{datetime.now().timestamp()}"
                    )
                    total_pnl += pnl

        return total_pnl

    def update_prices(self, prices: Dict[str, float]):
        """Update all positions with current prices"""
        with self._lock:
            for futu_code, price in prices.items():
                if futu_code in self._positions:
                    self._positions[futu_code].update_price(price)

    def check_risk_levels(self, prices: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Check stop loss and take profit for all positions.

        Returns:
            List of (futu_code, trigger_type, price) tuples
        """
        triggers = []

        with self._lock:
            for futu_code, position in self._positions.items():
                if position.is_flat:
                    continue

                price = prices.get(futu_code)
                if not price:
                    continue

                if position.check_stop_loss(price):
                    triggers.append((futu_code, "stop_loss", price))
                elif position.check_take_profit(price):
                    triggers.append((futu_code, "take_profit", price))

        return triggers

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions"""
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L"""
        return self._total_realized_pnl

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value"""
        with self._lock:
            position_value = sum(
                p.abs_quantity * p.avg_cost for p in self._positions.values()
            )
            return self.cash + position_value

    @property
    def equity(self) -> float:
        """Current equity (starting cash + total P&L)"""
        return self.starting_cash + self.total_pnl

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak"""
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

        if self._peak_equity <= 0:
            return 0.0

        return (self._peak_equity - self.equity) / self._peak_equity

    @property
    def open_positions(self) -> List[ManagedPosition]:
        """List of all open positions"""
        with self._lock:
            return [p for p in self._positions.values() if not p.is_flat]

    @property
    def position_count(self) -> int:
        """Number of open positions"""
        return len(self.open_positions)

    def get_exposure(self) -> Dict[str, float]:
        """Get exposure by symbol"""
        with self._lock:
            exposure = {}
            for futu_code, position in self._positions.items():
                if not position.is_flat:
                    exposure[futu_code] = position.market_value / self.portfolio_value
            return exposure

    def get_summary(self) -> dict:
        """Get portfolio summary"""
        return {
            "cash": round(self.cash, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "equity": round(self.equity, 2),
            "total_pnl": round(self.total_pnl, 2),
            "realized_pnl": round(self.total_realized_pnl, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "drawdown_pct": round(self.drawdown * 100, 2),
            "position_count": self.position_count,
            "positions": [p.to_dict() for p in self.open_positions],
        }
