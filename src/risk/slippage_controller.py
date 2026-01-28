"""
Slippage Controller - Monitor and control order slippage
Ensures slippage stays within ≤0.2% target
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SlippageRecord:
    """Record of a single slippage event"""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    expected_price: float
    actual_price: float
    quantity: int
    slippage_pct: float
    slippage_usd: float

    @property
    def is_favorable(self) -> bool:
        """Check if slippage was favorable"""
        if self.side == "buy":
            return self.actual_price < self.expected_price
        else:  # sell
            return self.actual_price > self.expected_price


@dataclass
class SlippageStats:
    """Aggregated slippage statistics"""
    total_orders: int = 0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    min_slippage_pct: float = 0.0
    median_slippage_pct: float = 0.0
    p95_slippage_pct: float = 0.0

    total_slippage_usd: float = 0.0
    favorable_count: int = 0
    unfavorable_count: int = 0

    within_target_pct: float = 0.0  # % of orders within 0.2% target


class SlippageController:
    """
    Controls and monitors order slippage.
    Target: ≤0.2% slippage
    """

    TARGET_SLIPPAGE_PCT = 0.002  # 0.2%

    def __init__(
        self,
        max_history: int = 10000,
        alert_threshold_pct: float = 0.003,  # 0.3%
        consecutive_alert_count: int = 5
    ):
        self.max_history = max_history
        self.alert_threshold_pct = alert_threshold_pct
        self.consecutive_alert_count = consecutive_alert_count

        self._records: deque = deque(maxlen=max_history)
        self._by_symbol: Dict[str, deque] = {}
        self._consecutive_high: int = 0
        self._lock = threading.Lock()

        # Callbacks
        self._alert_callbacks: List = []

    def record_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity: int
    ) -> SlippageRecord:
        """
        Record a slippage event.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            expected_price: Expected execution price
            actual_price: Actual execution price
            quantity: Order quantity

        Returns:
            SlippageRecord
        """
        # Calculate slippage
        if side == "buy":
            slippage_pct = (actual_price - expected_price) / expected_price
        else:  # sell
            slippage_pct = (expected_price - actual_price) / expected_price

        slippage_usd = abs(actual_price - expected_price) * quantity

        record = SlippageRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd
        )

        with self._lock:
            self._records.append(record)

            if symbol not in self._by_symbol:
                self._by_symbol[symbol] = deque(maxlen=1000)
            self._by_symbol[symbol].append(record)

            # Check for consecutive high slippage
            if abs(slippage_pct) > self.alert_threshold_pct:
                self._consecutive_high += 1
                if self._consecutive_high >= self.consecutive_alert_count:
                    self._trigger_alert(record)
            else:
                self._consecutive_high = 0

        # Log if high slippage
        if abs(slippage_pct) > self.TARGET_SLIPPAGE_PCT:
            logger.warning(
                f"High slippage: {symbol} {side} "
                f"expected={expected_price:.4f} actual={actual_price:.4f} "
                f"slippage={slippage_pct:.4%}"
            )

        return record

    def get_stats(self, hours: int = 24) -> SlippageStats:
        """Get slippage statistics for the specified period"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            records = [r for r in self._records if r.timestamp >= cutoff]

        if not records:
            return SlippageStats()

        slippages = [abs(r.slippage_pct) for r in records]
        slippages.sort()

        n = len(slippages)
        within_target = sum(1 for s in slippages if s <= self.TARGET_SLIPPAGE_PCT)

        return SlippageStats(
            total_orders=n,
            avg_slippage_pct=sum(slippages) / n,
            max_slippage_pct=max(slippages),
            min_slippage_pct=min(slippages),
            median_slippage_pct=slippages[n // 2],
            p95_slippage_pct=slippages[int(n * 0.95)],
            total_slippage_usd=sum(r.slippage_usd for r in records),
            favorable_count=sum(1 for r in records if r.is_favorable),
            unfavorable_count=sum(1 for r in records if not r.is_favorable),
            within_target_pct=within_target / n if n > 0 else 0
        )

    def get_symbol_stats(self, symbol: str, hours: int = 24) -> SlippageStats:
        """Get slippage statistics for a specific symbol"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            if symbol not in self._by_symbol:
                return SlippageStats()

            records = [r for r in self._by_symbol[symbol] if r.timestamp >= cutoff]

        if not records:
            return SlippageStats()

        slippages = [abs(r.slippage_pct) for r in records]
        slippages.sort()

        n = len(slippages)
        within_target = sum(1 for s in slippages if s <= self.TARGET_SLIPPAGE_PCT)

        return SlippageStats(
            total_orders=n,
            avg_slippage_pct=sum(slippages) / n,
            max_slippage_pct=max(slippages),
            min_slippage_pct=min(slippages),
            median_slippage_pct=slippages[n // 2],
            p95_slippage_pct=slippages[int(n * 0.95)],
            total_slippage_usd=sum(r.slippage_usd for r in records),
            favorable_count=sum(1 for r in records if r.is_favorable),
            unfavorable_count=sum(1 for r in records if not r.is_favorable),
            within_target_pct=within_target / n if n > 0 else 0
        )

    def is_meeting_target(self, hours: int = 24) -> bool:
        """Check if slippage is meeting the ≤0.2% target"""
        stats = self.get_stats(hours)
        return stats.within_target_pct >= 0.95  # 95% of orders within target

    def get_recommendation(self) -> str:
        """Get recommendation based on current slippage"""
        stats = self.get_stats(24)

        if stats.total_orders < 10:
            return "Insufficient data for recommendation"

        if stats.avg_slippage_pct > self.TARGET_SLIPPAGE_PCT * 2:
            return "CRITICAL: Slippage too high. Consider using limit orders or reducing position size."

        if stats.avg_slippage_pct > self.TARGET_SLIPPAGE_PCT:
            return "WARNING: Slippage above target. Consider using more aggressive limit prices."

        if stats.within_target_pct < 0.90:
            return "INFO: Some orders have high slippage. Monitor market conditions."

        return "OK: Slippage within acceptable range."

    def on_alert(self, callback):
        """Register alert callback"""
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, record: SlippageRecord):
        """Trigger slippage alert"""
        logger.error(
            f"SLIPPAGE ALERT: {self._consecutive_high} consecutive high slippage orders! "
            f"Latest: {record.symbol} {record.slippage_pct:.4%}"
        )

        for callback in self._alert_callbacks:
            try:
                callback(record, self._consecutive_high)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")


class FillRateMonitor:
    """
    Monitor order fill rate.
    Target: ≥95% fill rate
    """

    TARGET_FILL_RATE = 0.95

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._orders: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def record_order(
        self,
        order_id: str,
        symbol: str,
        requested_qty: int,
        filled_qty: int,
        status: str
    ):
        """Record an order fill event"""
        fill_rate = filled_qty / requested_qty if requested_qty > 0 else 0

        with self._lock:
            self._orders.append({
                "timestamp": datetime.now(),
                "order_id": order_id,
                "symbol": symbol,
                "requested_qty": requested_qty,
                "filled_qty": filled_qty,
                "fill_rate": fill_rate,
                "status": status,
                "is_fully_filled": filled_qty >= requested_qty
            })

    def get_fill_rate(self, hours: int = 24) -> float:
        """Get overall fill rate for the period"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            orders = [o for o in self._orders if o["timestamp"] >= cutoff]

        if not orders:
            return 1.0

        fully_filled = sum(1 for o in orders if o["is_fully_filled"])
        return fully_filled / len(orders)

    def get_partial_fill_stats(self, hours: int = 24) -> Dict:
        """Get partial fill statistics"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            orders = [o for o in self._orders if o["timestamp"] >= cutoff]

        if not orders:
            return {"total": 0, "fully_filled": 0, "partial": 0, "unfilled": 0}

        fully_filled = sum(1 for o in orders if o["fill_rate"] >= 1.0)
        partial = sum(1 for o in orders if 0 < o["fill_rate"] < 1.0)
        unfilled = sum(1 for o in orders if o["fill_rate"] == 0)

        return {
            "total": len(orders),
            "fully_filled": fully_filled,
            "partial": partial,
            "unfilled": unfilled,
            "fill_rate": fully_filled / len(orders)
        }

    def is_meeting_target(self, hours: int = 24) -> bool:
        """Check if fill rate meets ≥95% target"""
        return self.get_fill_rate(hours) >= self.TARGET_FILL_RATE


class VolumeTracker:
    """
    Track daily trading volume.
    Target: ≥$50,000 daily volume
    """

    TARGET_DAILY_VOLUME_USD = 50000.0

    def __init__(self):
        self._daily_volumes: Dict[str, float] = {}  # date -> volume
        self._by_symbol: Dict[str, Dict[str, float]] = {}  # symbol -> date -> volume
        self._lock = threading.Lock()

    def record_trade(self, symbol: str, quantity: int, price: float):
        """Record a trade"""
        volume = quantity * price
        today = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            # Total volume
            self._daily_volumes[today] = self._daily_volumes.get(today, 0) + volume

            # By symbol
            if symbol not in self._by_symbol:
                self._by_symbol[symbol] = {}
            self._by_symbol[symbol][today] = self._by_symbol[symbol].get(today, 0) + volume

    def get_today_volume(self) -> float:
        """Get today's total volume"""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            return self._daily_volumes.get(today, 0)

    def get_symbol_volume(self, symbol: str) -> float:
        """Get today's volume for a symbol"""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if symbol not in self._by_symbol:
                return 0
            return self._by_symbol[symbol].get(today, 0)

    def get_average_daily_volume(self, days: int = 30) -> float:
        """Get average daily volume"""
        with self._lock:
            volumes = list(self._daily_volumes.values())[-days:]

        if not volumes:
            return 0

        return sum(volumes) / len(volumes)

    def is_meeting_target(self) -> bool:
        """Check if today's volume meets ≥$50,000 target"""
        return self.get_today_volume() >= self.TARGET_DAILY_VOLUME_USD

    def get_progress(self) -> float:
        """Get progress towards daily target (0-100%)"""
        return min(100.0, (self.get_today_volume() / self.TARGET_DAILY_VOLUME_USD) * 100)


# Singleton instances
_slippage_controller: Optional[SlippageController] = None
_fill_rate_monitor: Optional[FillRateMonitor] = None
_volume_tracker: Optional[VolumeTracker] = None


def get_slippage_controller() -> SlippageController:
    """Get global slippage controller"""
    global _slippage_controller
    if _slippage_controller is None:
        _slippage_controller = SlippageController()
    return _slippage_controller


def get_fill_rate_monitor() -> FillRateMonitor:
    """Get global fill rate monitor"""
    global _fill_rate_monitor
    if _fill_rate_monitor is None:
        _fill_rate_monitor = FillRateMonitor()
    return _fill_rate_monitor


def get_volume_tracker() -> VolumeTracker:
    """Get global volume tracker"""
    global _volume_tracker
    if _volume_tracker is None:
        _volume_tracker = VolumeTracker()
    return _volume_tracker
