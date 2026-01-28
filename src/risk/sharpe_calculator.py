"""
Sharpe Ratio Calculator - Calculate and monitor Sharpe ratio
Target: ≥2.0 Sharpe ratio
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Complete risk metrics"""
    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Target compliance
    sharpe_meets_target: bool = False
    drawdown_meets_target: bool = False

    # Period
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    trading_days: int = 0


class SharpeCalculator:
    """
    Calculate and monitor Sharpe ratio in real-time.
    Target: Sharpe ≥ 2.0
    """

    TARGET_SHARPE = 2.0
    TARGET_MAX_DRAWDOWN = 0.15  # 15%
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, starting_equity: float = 100000.0):
        self.starting_equity = starting_equity

        # Daily returns
        self._daily_returns: List[float] = []
        self._daily_equity: List[Tuple[date, float]] = []

        # Trade-level data
        self._trade_pnls: List[float] = []

        # Current state
        self._current_equity = starting_equity
        self._peak_equity = starting_equity

        self._lock = threading.Lock()

    def update_equity(self, equity: float):
        """Update current equity"""
        today = date.today()

        with self._lock:
            # Calculate daily return if we have previous data
            if self._daily_equity:
                last_date, last_equity = self._daily_equity[-1]

                if last_date == today:
                    # Update today's equity
                    self._daily_equity[-1] = (today, equity)
                else:
                    # New day - calculate yesterday's return
                    daily_return = (equity - last_equity) / last_equity if last_equity > 0 else 0
                    self._daily_returns.append(daily_return)
                    self._daily_equity.append((today, equity))
            else:
                self._daily_equity.append((today, equity))

            self._current_equity = equity
            self._peak_equity = max(self._peak_equity, equity)

    def record_trade_pnl(self, pnl: float):
        """Record a trade P&L"""
        with self._lock:
            self._trade_pnls.append(pnl)

    def calculate_sharpe(self, annualized: bool = True) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (R - Rf) / σ
        Where:
            R = Average return
            Rf = Risk-free rate
            σ = Standard deviation of returns
        """
        with self._lock:
            returns = self._daily_returns.copy()

        if len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        # Daily risk-free rate
        rf_daily = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        sharpe = (avg_return - rf_daily) / std_dev

        if annualized:
            sharpe *= math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sharpe

    def calculate_sortino(self, annualized: bool = True) -> float:
        """
        Calculate Sortino ratio (uses downside volatility).

        Sortino = (R - Rf) / σd
        Where σd = Downside deviation
        """
        with self._lock:
            returns = self._daily_returns.copy()

        if len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)

        # Calculate downside deviation (only negative returns)
        rf_daily = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
        negative_returns = [r for r in returns if r < rf_daily]

        if not negative_returns:
            return float('inf')  # No downside risk

        downside_variance = sum((r - rf_daily) ** 2 for r in negative_returns) / len(negative_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0

        if downside_dev == 0:
            return 0.0

        sortino = (avg_return - rf_daily) / downside_dev

        if annualized:
            sortino *= math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sortino

    def calculate_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown.

        Returns:
            (max_drawdown_usd, max_drawdown_pct)
        """
        with self._lock:
            equity_curve = [eq for _, eq in self._daily_equity]

        if not equity_curve:
            return 0.0, 0.0

        peak = equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity

            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def calculate_calmar(self) -> float:
        """
        Calculate Calmar ratio.

        Calmar = Annualized Return / Max Drawdown
        """
        with self._lock:
            returns = self._daily_returns.copy()

        if len(returns) < 2:
            return 0.0

        # Annualized return
        avg_daily = sum(returns) / len(returns)
        annualized_return = avg_daily * self.TRADING_DAYS_PER_YEAR

        _, max_dd_pct = self.calculate_max_drawdown()

        if max_dd_pct == 0:
            return 0.0

        return annualized_return / max_dd_pct

    def get_current_drawdown(self) -> Tuple[float, float]:
        """Get current drawdown from peak"""
        with self._lock:
            dd = self._peak_equity - self._current_equity
            dd_pct = dd / self._peak_equity if self._peak_equity > 0 else 0

        return dd, dd_pct

    def get_metrics(self) -> RiskMetrics:
        """Get complete risk metrics"""
        with self._lock:
            returns = self._daily_returns.copy()
            pnls = self._trade_pnls.copy()
            equity_curve = self._daily_equity.copy()

        metrics = RiskMetrics()

        if not equity_curve:
            return metrics

        # Date range
        metrics.start_date = equity_curve[0][0]
        metrics.end_date = equity_curve[-1][0]
        metrics.trading_days = len(equity_curve)

        # Returns
        metrics.total_return = self._current_equity - self.starting_equity
        metrics.total_return_pct = metrics.total_return / self.starting_equity if self.starting_equity > 0 else 0

        if len(returns) > 0:
            avg_daily = sum(returns) / len(returns)
            metrics.annualized_return = avg_daily * self.TRADING_DAYS_PER_YEAR

            # Volatility
            if len(returns) > 1:
                variance = sum((r - avg_daily) ** 2 for r in returns) / (len(returns) - 1)
                metrics.volatility = math.sqrt(variance)
                metrics.annualized_volatility = metrics.volatility * math.sqrt(self.TRADING_DAYS_PER_YEAR)

                # Downside volatility
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    down_var = sum(r ** 2 for r in negative_returns) / len(negative_returns)
                    metrics.downside_volatility = math.sqrt(down_var)

        # Risk-adjusted metrics
        metrics.sharpe_ratio = self.calculate_sharpe()
        metrics.sortino_ratio = self.calculate_sortino()
        metrics.calmar_ratio = self.calculate_calmar()

        # Drawdown
        metrics.max_drawdown, metrics.max_drawdown_pct = self.calculate_max_drawdown()
        metrics.current_drawdown, metrics.current_drawdown_pct = self.get_current_drawdown()

        # Trade metrics
        if pnls:
            metrics.total_trades = len(pnls)
            metrics.winning_trades = sum(1 for p in pnls if p > 0)
            metrics.losing_trades = sum(1 for p in pnls if p < 0)
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]

            if wins:
                metrics.avg_win = sum(wins) / len(wins)
            if losses:
                metrics.avg_loss = sum(losses) / len(losses)

            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 1
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Target compliance
        metrics.sharpe_meets_target = metrics.sharpe_ratio >= self.TARGET_SHARPE
        metrics.drawdown_meets_target = metrics.max_drawdown_pct <= self.TARGET_MAX_DRAWDOWN

        return metrics

    def is_meeting_sharpe_target(self) -> bool:
        """Check if Sharpe ratio meets ≥2.0 target"""
        return self.calculate_sharpe() >= self.TARGET_SHARPE

    def is_meeting_drawdown_target(self) -> bool:
        """Check if max drawdown meets ≤15% target"""
        _, max_dd_pct = self.calculate_max_drawdown()
        return max_dd_pct <= self.TARGET_MAX_DRAWDOWN

    def get_status_report(self) -> str:
        """Get human-readable status report"""
        metrics = self.get_metrics()

        report = []
        report.append("=" * 50)
        report.append("Risk Metrics Report")
        report.append("=" * 50)
        report.append(f"Period: {metrics.start_date} to {metrics.end_date} ({metrics.trading_days} days)")
        report.append("")
        report.append("Returns:")
        report.append(f"  Total Return: ${metrics.total_return:+,.2f} ({metrics.total_return_pct:+.2%})")
        report.append(f"  Annualized Return: {metrics.annualized_return:.2%}")
        report.append("")
        report.append("Risk:")
        report.append(f"  Volatility (Annual): {metrics.annualized_volatility:.2%}")
        report.append(f"  Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})")
        report.append(f"  Current Drawdown: ${metrics.current_drawdown:,.2f} ({metrics.current_drawdown_pct:.2%})")
        report.append("")
        report.append("Risk-Adjusted:")
        status_sharpe = "✅" if metrics.sharpe_meets_target else "❌"
        status_dd = "✅" if metrics.drawdown_meets_target else "❌"
        report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f} (target: ≥{self.TARGET_SHARPE}) {status_sharpe}")
        report.append(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        report.append(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
        report.append(f"  Max Drawdown: {metrics.max_drawdown_pct:.2%} (target: ≤{self.TARGET_MAX_DRAWDOWN:.0%}) {status_dd}")
        report.append("")
        report.append("Trades:")
        report.append(f"  Total: {metrics.total_trades}, Win: {metrics.winning_trades}, Loss: {metrics.losing_trades}")
        report.append(f"  Win Rate: {metrics.win_rate:.1%}")
        report.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
        report.append("=" * 50)

        return "\n".join(report)


# Singleton instance
_sharpe_calculator: Optional[SharpeCalculator] = None


def get_sharpe_calculator(starting_equity: float = 100000.0) -> SharpeCalculator:
    """Get global Sharpe calculator"""
    global _sharpe_calculator
    if _sharpe_calculator is None:
        _sharpe_calculator = SharpeCalculator(starting_equity)
    return _sharpe_calculator
