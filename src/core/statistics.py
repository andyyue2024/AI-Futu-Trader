"""
Trading Statistics - Comprehensive trading analytics and reporting
Calculates Sharpe ratio, win rate, drawdown, and generates reports
"""
import math
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Single trade record"""
    trade_id: str
    symbol: str
    futu_code: str

    # Entry
    entry_time: datetime
    entry_price: float
    entry_side: str  # "long" or "short"
    quantity: int

    # Exit (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    # Calculated
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period_minutes: int = 0

    # Execution quality
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def close(self, exit_time: datetime, exit_price: float, exit_slippage: float = 0.0):
        """Close the trade and calculate P&L"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_slippage = exit_slippage

        # Calculate P&L
        if self.entry_side == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.holding_period_minutes = int((exit_time - self.entry_time).total_seconds() / 60)


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: date
    starting_equity: float
    ending_equity: float

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Volume
    total_volume: float = 0.0
    total_shares: int = 0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Execution
    avg_slippage: float = 0.0
    avg_latency_ms: float = 0.0
    fill_rate: float = 1.0

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "realized_pnl": round(self.realized_pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "total_volume": round(self.total_volume, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "avg_slippage": round(self.avg_slippage * 100, 4),
        }


@dataclass
class PerformanceMetrics:
    """Overall performance metrics"""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Win/Loss
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Trading activity
    avg_trades_per_day: float = 0.0
    avg_holding_period_minutes: float = 0.0
    total_volume: float = 0.0

    # Execution quality
    avg_slippage: float = 0.0
    fill_rate: float = 1.0
    avg_latency_ms: float = 0.0

    # Time period
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    trading_days: int = 0

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "annualized_return": round(self.annualized_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "profit_factor": round(self.profit_factor, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "avg_trades_per_day": round(self.avg_trades_per_day, 1),
            "avg_slippage_pct": round(self.avg_slippage * 100, 4),
            "trading_days": self.trading_days,
        }

    def summary(self) -> str:
        """Generate text summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                  TRADING PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
║ Period: {self.start_date} to {self.end_date} ({self.trading_days} days)
╠══════════════════════════════════════════════════════════════╣
║ RETURNS                                                       ║
║   Total Return:      ${self.total_return:>12,.2f} ({self.total_return_pct*100:>6.2f}%)
║   Annualized Return:              {self.annualized_return*100:>6.2f}%
╠══════════════════════════════════════════════════════════════╣
║ RISK-ADJUSTED                                                 ║
║   Sharpe Ratio:                   {self.sharpe_ratio:>6.2f}
║   Sortino Ratio:                  {self.sortino_ratio:>6.2f}
║   Calmar Ratio:                   {self.calmar_ratio:>6.2f}
╠══════════════════════════════════════════════════════════════╣
║ DRAWDOWN                                                      ║
║   Max Drawdown:      ${self.max_drawdown:>12,.2f} ({self.max_drawdown_pct*100:>6.2f}%)
║   Max DD Duration:               {self.max_drawdown_duration_days:>3} days
╠══════════════════════════════════════════════════════════════╣
║ TRADES                                                        ║
║   Total Trades:                  {self.total_trades:>5}
║   Win Rate:                       {self.win_rate*100:>5.1f}%
║   Profit Factor:                  {self.profit_factor:>6.2f}
║   Avg Win:           ${self.avg_win:>12,.2f}
║   Avg Loss:          ${self.avg_loss:>12,.2f}
╠══════════════════════════════════════════════════════════════╣
║ EXECUTION                                                     ║
║   Avg Slippage:                   {self.avg_slippage*100:>6.4f}%
║   Fill Rate:                      {self.fill_rate*100:>5.1f}%
║   Avg Latency:                   {self.avg_latency_ms:>5.2f}ms
╚══════════════════════════════════════════════════════════════╝
"""


class TradingStatistics:
    """
    Comprehensive trading statistics calculator.
    Tracks trades, calculates metrics, and generates reports.
    """

    def __init__(self, starting_equity: float = 100000.0, risk_free_rate: float = 0.05):
        self.starting_equity = starting_equity
        self.risk_free_rate = risk_free_rate

        # Trade tracking
        self._trades: List[Trade] = []
        self._open_trades: Dict[str, Trade] = {}  # symbol -> Trade

        # Daily tracking
        self._daily_stats: Dict[date, DailyStats] = {}
        self._daily_returns: List[float] = []

        # Equity curve
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._current_equity = starting_equity
        self._peak_equity = starting_equity

        # Execution tracking
        self._slippage_samples: List[float] = []
        self._latency_samples: List[float] = []
        self._fill_rates: List[float] = []

    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        futu_code: str,
        side: str,
        quantity: int,
        entry_price: float,
        entry_time: datetime = None,
        slippage: float = 0.0
    ) -> Trade:
        """Record a new trade entry"""
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            futu_code=futu_code,
            entry_time=entry_time or datetime.now(),
            entry_price=entry_price,
            entry_side=side,
            quantity=quantity,
            entry_slippage=slippage
        )

        self._open_trades[futu_code] = trade
        self._slippage_samples.append(slippage)

        logger.info(f"Trade entry: {side.upper()} {quantity} {symbol} @ ${entry_price:.2f}")

        return trade

    def record_exit(
        self,
        futu_code: str,
        exit_price: float,
        exit_time: datetime = None,
        slippage: float = 0.0
    ) -> Optional[Trade]:
        """Record a trade exit"""
        if futu_code not in self._open_trades:
            logger.warning(f"No open trade found for {futu_code}")
            return None

        trade = self._open_trades.pop(futu_code)
        trade.close(
            exit_time=exit_time or datetime.now(),
            exit_price=exit_price,
            exit_slippage=slippage
        )

        self._trades.append(trade)
        self._slippage_samples.append(slippage)

        # Update equity
        self._current_equity += trade.pnl
        self._equity_curve.append((trade.exit_time, self._current_equity))

        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        logger.info(
            f"Trade exit: {trade.symbol} PnL: ${trade.pnl:+.2f} ({trade.pnl_pct*100:+.2f}%)"
        )

        return trade

    def record_latency(self, latency_ms: float):
        """Record order latency"""
        self._latency_samples.append(latency_ms)

    def record_fill_rate(self, filled: int, requested: int):
        """Record order fill rate"""
        if requested > 0:
            self._fill_rates.append(filled / requested)

    def update_daily_stats(self, current_date: date = None):
        """Update daily statistics"""
        today = current_date or date.today()

        if today not in self._daily_stats:
            # Initialize new day
            self._daily_stats[today] = DailyStats(
                date=today,
                starting_equity=self._current_equity,
                ending_equity=self._current_equity
            )

        stats = self._daily_stats[today]

        # Get today's closed trades
        today_trades = [
            t for t in self._trades
            if t.is_closed and t.exit_time.date() == today
        ]

        # Update stats
        stats.total_trades = len(today_trades)
        stats.winning_trades = sum(1 for t in today_trades if t.is_winner)
        stats.losing_trades = stats.total_trades - stats.winning_trades
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0

        stats.realized_pnl = sum(t.pnl for t in today_trades)
        stats.ending_equity = self._current_equity
        stats.pnl_pct = (stats.ending_equity - stats.starting_equity) / stats.starting_equity if stats.starting_equity > 0 else 0

        stats.total_volume = sum(t.entry_price * t.quantity for t in today_trades)
        stats.total_shares = sum(t.quantity for t in today_trades)

        # Update drawdown
        if self._peak_equity > 0:
            stats.max_drawdown = self._peak_equity - self._current_equity
            stats.max_drawdown_pct = stats.max_drawdown / self._peak_equity

        # Execution quality
        if self._slippage_samples:
            stats.avg_slippage = statistics.mean(self._slippage_samples[-100:])
        if self._latency_samples:
            stats.avg_latency_ms = statistics.mean(self._latency_samples[-100:])
        if self._fill_rates:
            stats.fill_rate = statistics.mean(self._fill_rates[-100:])

        return stats

    def finalize_day(self, current_date: date = None):
        """Finalize daily stats and record daily return"""
        today = current_date or date.today()
        stats = self.update_daily_stats(today)

        if stats.starting_equity > 0:
            daily_return = stats.pnl_pct
            self._daily_returns.append(daily_return)

        return stats

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = PerformanceMetrics()

        closed_trades = [t for t in self._trades if t.is_closed]

        if not closed_trades:
            return metrics

        # Basic counts
        metrics.total_trades = len(closed_trades)
        metrics.winning_trades = sum(1 for t in closed_trades if t.is_winner)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # P&L
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [t.pnl for t in closed_trades if t.pnl < 0]

        metrics.total_return = sum(t.pnl for t in closed_trades)
        metrics.total_return_pct = metrics.total_return / self.starting_equity if self.starting_equity > 0 else 0

        if wins:
            metrics.avg_win = statistics.mean(wins)
            metrics.largest_win = max(wins)
            metrics.avg_win_pct = statistics.mean([t.pnl_pct for t in closed_trades if t.pnl > 0])

        if losses:
            metrics.avg_loss = statistics.mean([abs(l) for l in losses])
            metrics.largest_loss = abs(min(losses))
            metrics.avg_loss_pct = statistics.mean([abs(t.pnl_pct) for t in closed_trades if t.pnl < 0])

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.0001
        metrics.profit_factor = total_wins / total_losses

        # Time period
        if closed_trades:
            metrics.start_date = min(t.entry_time.date() for t in closed_trades)
            metrics.end_date = max(t.exit_time.date() for t in closed_trades if t.exit_time)
            metrics.trading_days = len(self._daily_stats)

        # Risk metrics (need daily returns)
        if len(self._daily_returns) >= 2:
            metrics.sharpe_ratio = self._calculate_sharpe()
            metrics.sortino_ratio = self._calculate_sortino()

        # Drawdown
        dd_result = self._calculate_max_drawdown()
        metrics.max_drawdown = dd_result[0]
        metrics.max_drawdown_pct = dd_result[1]
        metrics.max_drawdown_duration_days = dd_result[2]

        # Annualized return
        if metrics.trading_days > 0:
            years = metrics.trading_days / 252
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return_pct) ** (1/years) - 1

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

        # Activity
        if metrics.trading_days > 0:
            metrics.avg_trades_per_day = metrics.total_trades / metrics.trading_days

        if closed_trades:
            metrics.avg_holding_period_minutes = statistics.mean(
                [t.holding_period_minutes for t in closed_trades]
            )

        # Execution
        if self._slippage_samples:
            metrics.avg_slippage = statistics.mean(self._slippage_samples)
        if self._fill_rates:
            metrics.fill_rate = statistics.mean(self._fill_rates)
        if self._latency_samples:
            metrics.avg_latency_ms = statistics.mean(self._latency_samples)

        metrics.total_volume = sum(t.entry_price * t.quantity for t in closed_trades)

        return metrics

    def _calculate_sharpe(self) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self._daily_returns) < 2:
            return 0.0

        returns = self._daily_returns
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        daily_rf = self.risk_free_rate / 252
        sharpe = (avg_return - daily_rf) / std_return * math.sqrt(252)

        return sharpe

    def _calculate_sortino(self) -> float:
        """Calculate annualized Sortino ratio"""
        if len(self._daily_returns) < 2:
            return 0.0

        returns = self._daily_returns
        avg_return = statistics.mean(returns)

        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf')

        downside_dev = math.sqrt(statistics.mean([r**2 for r in negative_returns]))

        if downside_dev == 0:
            return 0.0

        daily_rf = self.risk_free_rate / 252
        sortino = (avg_return - daily_rf) / downside_dev * math.sqrt(252)

        return sortino

    def _calculate_max_drawdown(self) -> Tuple[float, float, int]:
        """Calculate maximum drawdown and duration"""
        if not self._equity_curve:
            return 0.0, 0.0, 0

        peak = self.starting_equity
        max_dd = 0.0
        max_dd_pct = 0.0

        # Track drawdown duration
        in_drawdown = False
        dd_start = None
        max_duration = 0

        for ts, equity in self._equity_curve:
            if equity > peak:
                peak = equity
                if in_drawdown:
                    duration = (ts - dd_start).days if dd_start else 0
                    max_duration = max(max_duration, duration)
                    in_drawdown = False
            else:
                dd = peak - equity
                dd_pct = dd / peak if peak > 0 else 0

                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

                if not in_drawdown:
                    in_drawdown = True
                    dd_start = ts

        return max_dd, max_dd_pct, max_duration

    def get_daily_stats(self, target_date: date = None) -> Optional[DailyStats]:
        """Get statistics for a specific day"""
        target = target_date or date.today()
        return self._daily_stats.get(target)

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data"""
        return self._equity_curve.copy()

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get all trades for a symbol"""
        return [t for t in self._trades if t.symbol == symbol or t.futu_code == symbol]

    def export_trades(self, filepath: str = None) -> str:
        """Export trades to JSON"""
        trades_data = []
        for trade in self._trades:
            trades_data.append({
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(),
                "entry_price": trade.entry_price,
                "entry_side": trade.entry_side,
                "quantity": trade.quantity,
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "holding_minutes": trade.holding_period_minutes,
            })

        json_str = json.dumps(trades_data, indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def generate_report(self) -> str:
        """Generate full performance report"""
        metrics = self.calculate_metrics()
        return metrics.summary()
