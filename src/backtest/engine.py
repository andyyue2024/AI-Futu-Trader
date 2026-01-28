"""
Backtesting Engine - Historical strategy validation
Simulates trading with historical data to validate strategies
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import json

import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.core.statistics import TradingStatistics, Trade, PerformanceMetrics
from src.data.data_processor import DataProcessor, MarketSnapshot
from src.action.futu_executor import TradingAction, Position
from src.model.llm_agent import TradingDecision, DecisionConfidence

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Capital
    starting_capital: float = 100000.0

    # Trading costs
    commission_per_share: float = 0.0  # Futu is commission-free for US stocks
    commission_min: float = 0.0
    slippage_pct: float = 0.001  # 0.1% simulated slippage

    # Position sizing
    max_position_pct: float = 0.25  # Max 25% in single position
    default_position_size: float = 10000.0

    # Risk limits
    max_daily_drawdown: float = 0.03  # 3% daily limit
    max_total_drawdown: float = 0.15  # 15% total limit

    # Execution
    use_close_price: bool = True  # Use close price for fills
    fill_delay_bars: int = 0  # Bars to delay fill (0 = immediate)

    # Symbol filter
    symbols: List[str] = field(default_factory=list)

    # Time filter
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@dataclass
class BacktestBar:
    """Single bar of historical data"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class BacktestResult:
    """Backtesting results"""
    config: BacktestConfig
    metrics: PerformanceMetrics

    # Equity curve
    equity_curve: List[Dict] = field(default_factory=list)

    # Trades
    trades: List[Dict] = field(default_factory=list)

    # Daily stats
    daily_stats: List[Dict] = field(default_factory=list)

    # Signals
    signals: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": {
                "starting_capital": self.config.starting_capital,
                "slippage_pct": self.config.slippage_pct,
                "max_position_pct": self.config.max_position_pct,
            },
            "metrics": self.metrics.to_dict(),
            "total_trades": len(self.trades),
            "trading_days": len(self.daily_stats),
        }

    def summary(self) -> str:
        """Generate summary report"""
        return self.metrics.summary()


class SimulatedExecutor:
    """Simulated order executor for backtesting"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._positions: Dict[str, Position] = {}
        self._pending_orders: List[Dict] = []
        self._current_bar: Optional[BacktestBar] = None

    def set_current_bar(self, bar: BacktestBar):
        """Update current market data"""
        self._current_bar = bar

    def get_position(self, symbol: str) -> Position:
        """Get current position"""
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol.split('.')[-1] if '.' in symbol else symbol,
                futu_code=symbol,
                quantity=0,
                avg_cost=0.0
            )
        return self._positions[symbol]

    def execute_long(
        self,
        symbol: str,
        quantity: int,
        stats: TradingStatistics
    ) -> bool:
        """Execute long order"""
        if not self._current_bar:
            return False

        position = self.get_position(symbol)

        # Calculate fill price with slippage
        fill_price = self._current_bar.close * (1 + self.config.slippage_pct)
        slippage = self.config.slippage_pct

        # Update position
        if position.quantity >= 0:
            # Adding to long or opening new long
            total_cost = position.avg_cost * position.quantity + fill_price * quantity
            position.quantity += quantity
            position.avg_cost = total_cost / position.quantity if position.quantity > 0 else 0
        else:
            # Closing short
            if quantity >= abs(position.quantity):
                # Fully close short
                stats.record_exit(symbol, fill_price, self._current_bar.timestamp, slippage)

                remaining = quantity - abs(position.quantity)
                position.quantity = remaining
                position.avg_cost = fill_price if remaining > 0 else 0

                if remaining > 0:
                    stats.record_entry(
                        trade_id=f"bt-{self._current_bar.timestamp.timestamp()}",
                        symbol=symbol,
                        futu_code=symbol,
                        side="long",
                        quantity=remaining,
                        entry_price=fill_price,
                        entry_time=self._current_bar.timestamp,
                        slippage=slippage
                    )
            else:
                # Partial close of short
                stats.record_exit(symbol, fill_price, self._current_bar.timestamp, slippage)
                position.quantity += quantity

        # Record entry if opening new position
        if position.quantity > 0 and position.quantity == quantity:
            stats.record_entry(
                trade_id=f"bt-{self._current_bar.timestamp.timestamp()}",
                symbol=symbol,
                futu_code=symbol,
                side="long",
                quantity=quantity,
                entry_price=fill_price,
                entry_time=self._current_bar.timestamp,
                slippage=slippage
            )

        return True

    def execute_short(
        self,
        symbol: str,
        quantity: int,
        stats: TradingStatistics
    ) -> bool:
        """Execute short order"""
        if not self._current_bar:
            return False

        position = self.get_position(symbol)

        # Calculate fill price with slippage
        fill_price = self._current_bar.close * (1 - self.config.slippage_pct)
        slippage = self.config.slippage_pct

        # Update position
        if position.quantity <= 0:
            # Adding to short or opening new short
            total_cost = abs(position.avg_cost * position.quantity) + fill_price * quantity
            position.quantity -= quantity
            position.avg_cost = total_cost / abs(position.quantity) if position.quantity != 0 else 0
        else:
            # Closing long
            if quantity >= position.quantity:
                # Fully close long
                stats.record_exit(symbol, fill_price, self._current_bar.timestamp, slippage)

                remaining = quantity - position.quantity
                position.quantity = -remaining
                position.avg_cost = fill_price if remaining > 0 else 0

                if remaining > 0:
                    stats.record_entry(
                        trade_id=f"bt-{self._current_bar.timestamp.timestamp()}",
                        symbol=symbol,
                        futu_code=symbol,
                        side="short",
                        quantity=remaining,
                        entry_price=fill_price,
                        entry_time=self._current_bar.timestamp,
                        slippage=slippage
                    )
            else:
                # Partial close of long
                position.quantity -= quantity

        # Record entry if opening new position
        if position.quantity < 0 and abs(position.quantity) == quantity:
            stats.record_entry(
                trade_id=f"bt-{self._current_bar.timestamp.timestamp()}",
                symbol=symbol,
                futu_code=symbol,
                side="short",
                quantity=quantity,
                entry_price=fill_price,
                entry_time=self._current_bar.timestamp,
                slippage=slippage
            )

        return True

    def execute_flat(
        self,
        symbol: str,
        stats: TradingStatistics
    ) -> bool:
        """Close all positions for symbol"""
        if not self._current_bar:
            return False

        position = self.get_position(symbol)

        if position.quantity == 0:
            return True

        fill_price = self._current_bar.close
        if position.quantity > 0:
            fill_price *= (1 - self.config.slippage_pct)
        else:
            fill_price *= (1 + self.config.slippage_pct)

        stats.record_exit(symbol, fill_price, self._current_bar.timestamp, self.config.slippage_pct)

        position.quantity = 0
        position.avg_cost = 0.0

        return True

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        value = 0.0
        for symbol, position in self._positions.items():
            if position.quantity != 0:
                price = prices.get(symbol, position.avg_cost)
                value += abs(position.quantity) * price
        return value


class BacktestEngine:
    """
    Backtesting engine for strategy validation.
    Simulates trading with historical data.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self._data: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame
        self._data_processor = DataProcessor()
        self._stats: Optional[TradingStatistics] = None
        self._executor: Optional[SimulatedExecutor] = None
        self._signals: List[Dict] = []

    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        required_columns: List[str] = None
    ):
        """
        Load historical data for a symbol.

        Args:
            symbol: Symbol identifier
            data: DataFrame with OHLCV data
            required_columns: Expected column names
        """
        if required_columns is None:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Validate columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure timestamp column
        if 'timestamp' not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
            else:
                raise ValueError("Data must have timestamp column or DatetimeIndex")

        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)

        # Apply date filter
        if self.config.start_date:
            data = data[data['timestamp'].dt.date >= self.config.start_date]
        if self.config.end_date:
            data = data[data['timestamp'].dt.date <= self.config.end_date]

        self._data[symbol] = data
        logger.info(f"Loaded {len(data)} bars for {symbol}")

    def load_data_from_csv(self, symbol: str, filepath: str):
        """Load data from CSV file"""
        data = pd.read_csv(filepath, parse_dates=['timestamp'])
        self.load_data(symbol, data)

    def run(
        self,
        strategy: Callable[[MarketSnapshot, Position, float], TradingDecision],
        progress_callback: Callable[[int, int], None] = None
    ) -> BacktestResult:
        """
        Run backtest with given strategy.

        Args:
            strategy: Function that takes (snapshot, position, equity) and returns TradingDecision
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            BacktestResult with metrics and trades
        """
        if not self._data:
            raise ValueError("No data loaded. Use load_data() first.")

        # Initialize
        self._stats = TradingStatistics(starting_equity=self.config.starting_capital)
        self._executor = SimulatedExecutor(self.config)
        self._signals = []

        cash = self.config.starting_capital
        equity_curve = []

        # Merge all data into timeline
        timeline = self._build_timeline()
        total_bars = len(timeline)

        logger.info(f"Starting backtest with {total_bars} bars")

        current_date = None

        for i, (timestamp, symbol, bar_data) in enumerate(timeline):
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_bars)

            # Create bar
            bar = BacktestBar(
                timestamp=timestamp,
                symbol=symbol,
                open=bar_data['open'],
                high=bar_data['high'],
                low=bar_data['low'],
                close=bar_data['close'],
                volume=bar_data['volume']
            )

            self._executor.set_current_bar(bar)

            # Daily stats update
            bar_date = timestamp.date()
            if current_date != bar_date:
                if current_date:
                    self._stats.finalize_day(current_date)
                current_date = bar_date

            # Build snapshot
            snapshot = self._build_snapshot(symbol, bar)
            if not snapshot:
                continue

            # Get current position
            position = self._executor.get_position(symbol)

            # Calculate current equity
            position_value = abs(position.quantity) * bar.close if position.quantity != 0 else 0
            current_equity = cash + position_value

            # Check circuit breaker
            if self._check_circuit_breaker(current_equity):
                logger.warning(f"Circuit breaker triggered at {timestamp}")
                self._executor.execute_flat(symbol, self._stats)
                continue

            # Get strategy decision
            try:
                decision = strategy(snapshot, position, current_equity)
            except Exception as e:
                logger.error(f"Strategy error at {timestamp}: {e}")
                continue

            # Record signal
            self._signals.append({
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": decision.action.value,
                "confidence": decision.confidence_score,
            })

            # Execute decision
            if decision.action == TradingAction.LONG and decision.should_execute:
                quantity = self._calculate_quantity(current_equity, bar.close)
                if quantity > 0:
                    self._executor.execute_long(symbol, quantity, self._stats)
                    cash -= quantity * bar.close * (1 + self.config.slippage_pct)

            elif decision.action == TradingAction.SHORT and decision.should_execute:
                quantity = self._calculate_quantity(current_equity, bar.close)
                if quantity > 0:
                    self._executor.execute_short(symbol, quantity, self._stats)
                    cash += quantity * bar.close * (1 - self.config.slippage_pct)

            elif decision.action == TradingAction.FLAT:
                if position.quantity != 0:
                    if position.quantity > 0:
                        cash += position.quantity * bar.close * (1 - self.config.slippage_pct)
                    else:
                        cash -= abs(position.quantity) * bar.close * (1 + self.config.slippage_pct)
                    self._executor.execute_flat(symbol, self._stats)

            # Update equity curve
            position_value = abs(position.quantity) * bar.close if position.quantity != 0 else 0
            current_equity = cash + position_value

            equity_curve.append({
                "timestamp": timestamp.isoformat(),
                "equity": current_equity,
                "cash": cash,
                "position_value": position_value,
            })

        # Finalize
        if current_date:
            self._stats.finalize_day(current_date)

        # Close any remaining positions
        for symbol in self._data.keys():
            self._executor.execute_flat(symbol, self._stats)

        # Calculate final metrics
        metrics = self._stats.calculate_metrics()

        # Build result
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=[],
            daily_stats=[s.to_dict() for s in self._stats._daily_stats.values()],
            signals=self._signals
        )

        logger.info(f"Backtest complete: {metrics.total_trades} trades, Sharpe: {metrics.sharpe_ratio:.2f}")

        return result

    def _build_timeline(self) -> List[tuple]:
        """Build chronological timeline of all bars"""
        timeline = []

        for symbol, df in self._data.items():
            for _, row in df.iterrows():
                timeline.append((row['timestamp'], symbol, row))

        # Sort by timestamp
        timeline.sort(key=lambda x: x[0])

        return timeline

    def _build_snapshot(self, symbol: str, bar: BacktestBar) -> Optional[MarketSnapshot]:
        """Build market snapshot from bar"""
        from src.data.futu_quote import KLineData

        # Update data processor
        kline = KLineData(
            symbol=bar.symbol,
            futu_code=symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            turnover=bar.close * bar.volume,
            timestamp=bar.timestamp
        )

        self._data_processor.update_kline(symbol, kline)

        # Get snapshot
        return self._data_processor.get_snapshot(symbol)

    def _calculate_quantity(self, equity: float, price: float) -> int:
        """Calculate position size"""
        if price <= 0:
            return 0

        max_value = equity * self.config.max_position_pct
        position_value = min(self.config.default_position_size, max_value)

        return int(position_value / price)

    def _check_circuit_breaker(self, current_equity: float) -> bool:
        """Check if circuit breaker should trigger"""
        if current_equity <= 0:
            return True

        # Daily drawdown check
        daily_stats = self._stats._daily_stats.get(date.today())
        if daily_stats:
            daily_return = (current_equity - daily_stats.starting_equity) / daily_stats.starting_equity
            if daily_return <= -self.config.max_daily_drawdown:
                return True

        # Total drawdown check
        total_return = (current_equity - self.config.starting_capital) / self.config.starting_capital
        if total_return <= -self.config.max_total_drawdown:
            return True

        return False


def create_simple_strategy(
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    use_macd: bool = True
) -> Callable:
    """
    Create a simple technical strategy for backtesting.

    Args:
        rsi_oversold: RSI level to go long
        rsi_overbought: RSI level to go short
        use_macd: Whether to also use MACD confirmation

    Returns:
        Strategy function
    """
    def strategy(
        snapshot: MarketSnapshot,
        position: Position,
        equity: float
    ) -> TradingDecision:
        indicators = snapshot.indicators

        # Default hold
        action = TradingAction.HOLD
        confidence = 0.5

        # RSI signals
        if indicators.rsi_14 < rsi_oversold:
            action = TradingAction.LONG
            confidence = 0.7 + (rsi_oversold - indicators.rsi_14) / 100
        elif indicators.rsi_14 > rsi_overbought:
            action = TradingAction.SHORT
            confidence = 0.7 + (indicators.rsi_14 - rsi_overbought) / 100

        # MACD confirmation
        if use_macd and action != TradingAction.HOLD:
            if action == TradingAction.LONG and not indicators.macd_bullish:
                confidence *= 0.7
            elif action == TradingAction.SHORT and not indicators.macd_bearish:
                confidence *= 0.7

        # Position management
        if position.quantity != 0:
            # Check for exit
            if position.is_long and indicators.rsi_14 > 70:
                action = TradingAction.FLAT
                confidence = 0.8
            elif position.is_short and indicators.rsi_14 < 30:
                action = TradingAction.FLAT
                confidence = 0.8

        conf_level = (
            DecisionConfidence.HIGH if confidence > 0.8
            else DecisionConfidence.MEDIUM if confidence > 0.6
            else DecisionConfidence.LOW
        )

        return TradingDecision(
            action=action,
            symbol=snapshot.symbol,
            futu_code=snapshot.futu_code,
            confidence=conf_level,
            confidence_score=min(1.0, confidence),
            reasoning=f"RSI: {indicators.rsi_14:.1f}, MACD: {indicators.macd:.4f}"
        )

    return strategy
