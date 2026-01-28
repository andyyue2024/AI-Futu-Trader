"""
AI-Trader Strategy Evaluator
Evaluates trading strategies with comprehensive metrics

Based on HKUDS AI-Trader methodology for strategy assessment
"""
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

from src.core.logger import get_logger

logger = get_logger(__name__)


class StrategyGrade(Enum):
    """Strategy performance grade"""
    A_PLUS = "A+"   # Excellent - Sharpe > 3, Drawdown < 5%
    A = "A"         # Very Good - Sharpe > 2.5, Drawdown < 10%
    B_PLUS = "B+"   # Good - Sharpe > 2, Drawdown < 15%
    B = "B"         # Acceptable - Sharpe > 1.5, Drawdown < 20%
    C = "C"         # Needs Improvement - Sharpe > 1, Drawdown < 25%
    D = "D"         # Poor - Sharpe < 1 or Drawdown > 25%
    F = "F"         # Fail - Negative Sharpe or Drawdown > 30%


@dataclass
class TradeRecord:
    """Single trade record for evaluation"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # "long" or "short"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period_minutes: int = 0
    slippage_pct: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class StrategyMetrics:
    """Comprehensive strategy metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner_pnl: float = 0.0
    avg_loser_pnl: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trade metrics
    profit_factor: float = 0.0
    avg_holding_period_minutes: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Execution metrics
    avg_slippage_pct: float = 0.0
    fill_rate: float = 0.0

    # Grade
    grade: StrategyGrade = StrategyGrade.D

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "grade": self.grade.value
        }


@dataclass
class BacktestVsLiveComparison:
    """Comparison between backtest and live performance"""
    backtest_metrics: StrategyMetrics
    live_metrics: StrategyMetrics

    # Differences
    return_diff_pct: float = 0.0
    sharpe_diff: float = 0.0
    win_rate_diff: float = 0.0
    slippage_diff_pct: float = 0.0

    # Assessment
    is_consistent: bool = False
    degradation_warning: bool = False
    recommendations: List[str] = field(default_factory=list)


class StrategyEvaluator:
    """
    Comprehensive strategy evaluator.
    Analyzes trading performance and provides actionable insights.
    """

    # Thresholds for grading
    GRADE_THRESHOLDS = {
        StrategyGrade.A_PLUS: {"min_sharpe": 3.0, "max_dd": 0.05},
        StrategyGrade.A: {"min_sharpe": 2.5, "max_dd": 0.10},
        StrategyGrade.B_PLUS: {"min_sharpe": 2.0, "max_dd": 0.15},
        StrategyGrade.B: {"min_sharpe": 1.5, "max_dd": 0.20},
        StrategyGrade.C: {"min_sharpe": 1.0, "max_dd": 0.25},
        StrategyGrade.D: {"min_sharpe": 0.0, "max_dd": 0.30},
    }

    def __init__(self, starting_capital: float = 100000.0):
        self.starting_capital = starting_capital
        self._trades: List[TradeRecord] = []
        self._daily_returns: List[float] = []
        self._equity_curve: List[Tuple[date, float]] = []

    def add_trade(self, trade: TradeRecord):
        """Add a trade record"""
        self._trades.append(trade)

    def add_trades(self, trades: List[TradeRecord]):
        """Add multiple trade records"""
        self._trades.extend(trades)

    def set_equity_curve(self, equity_curve: List[Tuple[date, float]]):
        """Set equity curve for drawdown calculation"""
        self._equity_curve = equity_curve

        # Calculate daily returns
        if len(equity_curve) > 1:
            self._daily_returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1][1]
                curr_equity = equity_curve[i][1]
                if prev_equity > 0:
                    self._daily_returns.append((curr_equity - prev_equity) / prev_equity)

    def evaluate(self) -> StrategyMetrics:
        """
        Evaluate strategy performance and return comprehensive metrics.
        """
        metrics = StrategyMetrics()

        if not self._trades:
            return metrics

        # Filter closed trades
        closed_trades = [t for t in self._trades if t.is_closed]

        if not closed_trades:
            return metrics

        # Basic metrics
        metrics.total_trades = len(closed_trades)
        metrics.winning_trades = sum(1 for t in closed_trades if t.is_winner)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # P&L metrics
        pnls = [t.pnl for t in closed_trades]
        metrics.total_pnl = sum(pnls)
        metrics.total_return_pct = metrics.total_pnl / self.starting_capital
        metrics.avg_trade_pnl = sum(pnls) / len(pnls)

        winners = [t.pnl for t in closed_trades if t.is_winner]
        losers = [abs(t.pnl) for t in closed_trades if not t.is_winner]

        if winners:
            metrics.avg_winner_pnl = sum(winners) / len(winners)
        if losers:
            metrics.avg_loser_pnl = sum(losers) / len(losers)

        # Profit factor
        total_wins = sum(winners) if winners else 0
        total_losses = sum(losers) if losers else 1
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Risk metrics from daily returns
        if self._daily_returns and len(self._daily_returns) > 1:
            metrics.sharpe_ratio = self._calculate_sharpe()
            metrics.sortino_ratio = self._calculate_sortino()

        # Drawdown
        if self._equity_curve:
            metrics.max_drawdown_pct, metrics.max_drawdown_duration_days = self._calculate_max_drawdown()
            if metrics.max_drawdown_pct > 0:
                annualized_return = metrics.total_return_pct * (252 / max(len(self._equity_curve), 1))
                metrics.calmar_ratio = annualized_return / metrics.max_drawdown_pct

        # Trade metrics
        holding_periods = [t.holding_period_minutes for t in closed_trades if t.holding_period_minutes > 0]
        if holding_periods:
            metrics.avg_holding_period_minutes = sum(holding_periods) / len(holding_periods)

        metrics.max_consecutive_wins = self._calculate_max_consecutive(closed_trades, True)
        metrics.max_consecutive_losses = self._calculate_max_consecutive(closed_trades, False)

        # Execution metrics
        slippages = [t.slippage_pct for t in closed_trades if t.slippage_pct > 0]
        if slippages:
            metrics.avg_slippage_pct = sum(slippages) / len(slippages)

        # Grade
        metrics.grade = self._calculate_grade(metrics)

        return metrics

    def compare_backtest_vs_live(
        self,
        backtest_trades: List[TradeRecord],
        live_trades: List[TradeRecord]
    ) -> BacktestVsLiveComparison:
        """
        Compare backtest performance with live trading.
        """
        # Evaluate backtest
        bt_evaluator = StrategyEvaluator(self.starting_capital)
        bt_evaluator.add_trades(backtest_trades)
        bt_metrics = bt_evaluator.evaluate()

        # Evaluate live
        live_evaluator = StrategyEvaluator(self.starting_capital)
        live_evaluator.add_trades(live_trades)
        live_metrics = live_evaluator.evaluate()

        comparison = BacktestVsLiveComparison(
            backtest_metrics=bt_metrics,
            live_metrics=live_metrics
        )

        # Calculate differences
        comparison.return_diff_pct = live_metrics.total_return_pct - bt_metrics.total_return_pct
        comparison.sharpe_diff = live_metrics.sharpe_ratio - bt_metrics.sharpe_ratio
        comparison.win_rate_diff = live_metrics.win_rate - bt_metrics.win_rate
        comparison.slippage_diff_pct = live_metrics.avg_slippage_pct - bt_metrics.avg_slippage_pct

        # Assess consistency
        comparison.is_consistent = (
            abs(comparison.return_diff_pct) < 0.1 and
            abs(comparison.sharpe_diff) < 0.5 and
            abs(comparison.win_rate_diff) < 0.1
        )

        # Check for degradation
        comparison.degradation_warning = (
            comparison.return_diff_pct < -0.05 or
            comparison.sharpe_diff < -0.5 or
            live_metrics.avg_slippage_pct > bt_metrics.avg_slippage_pct * 2
        )

        # Generate recommendations
        if comparison.slippage_diff_pct > 0.001:
            comparison.recommendations.append("Consider using limit orders to reduce slippage")
        if comparison.win_rate_diff < -0.1:
            comparison.recommendations.append("Review entry timing - live win rate significantly lower")
        if comparison.degradation_warning:
            comparison.recommendations.append("Strategy may be overfit - consider retraining")

        return comparison

    def generate_report(self) -> str:
        """Generate human-readable performance report"""
        metrics = self.evaluate()

        lines = [
            "=" * 60,
            "STRATEGY PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Grade: {metrics.grade.value}",
            "",
            "TRADING SUMMARY:",
            f"  Total Trades: {metrics.total_trades}",
            f"  Win Rate: {metrics.win_rate:.1%}",
            f"  Winning Trades: {metrics.winning_trades}",
            f"  Losing Trades: {metrics.losing_trades}",
            "",
            "P&L METRICS:",
            f"  Total P&L: ${metrics.total_pnl:,.2f}",
            f"  Total Return: {metrics.total_return_pct:.2%}",
            f"  Avg Trade P&L: ${metrics.avg_trade_pnl:,.2f}",
            f"  Avg Winner: ${metrics.avg_winner_pnl:,.2f}",
            f"  Avg Loser: ${metrics.avg_loser_pnl:,.2f}",
            f"  Profit Factor: {metrics.profit_factor:.2f}",
            "",
            "RISK METRICS:",
            f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {metrics.sortino_ratio:.2f}",
            f"  Calmar Ratio: {metrics.calmar_ratio:.2f}",
            f"  Max Drawdown: {metrics.max_drawdown_pct:.2%}",
            f"  Max DD Duration: {metrics.max_drawdown_duration_days} days",
            "",
            "EXECUTION:",
            f"  Avg Slippage: {metrics.avg_slippage_pct:.4%}",
            f"  Avg Holding: {metrics.avg_holding_period_minutes:.0f} min",
            f"  Max Consecutive Wins: {metrics.max_consecutive_wins}",
            f"  Max Consecutive Losses: {metrics.max_consecutive_losses}",
            "",
            "=" * 60
        ]

        return "\n".join(lines)

    def _calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self._daily_returns) < 2:
            return 0.0

        avg_return = sum(self._daily_returns) / len(self._daily_returns)
        variance = sum((r - avg_return) ** 2 for r in self._daily_returns) / (len(self._daily_returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        rf_daily = risk_free_rate / 252
        sharpe = (avg_return - rf_daily) / std_dev * math.sqrt(252)

        return sharpe

    def _calculate_sortino(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sortino ratio"""
        if len(self._daily_returns) < 2:
            return 0.0

        avg_return = sum(self._daily_returns) / len(self._daily_returns)
        rf_daily = risk_free_rate / 252

        negative_returns = [r for r in self._daily_returns if r < rf_daily]
        if not negative_returns:
            return float('inf')

        downside_variance = sum((r - rf_daily) ** 2 for r in negative_returns) / len(negative_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0

        if downside_dev == 0:
            return 0.0

        sortino = (avg_return - rf_daily) / downside_dev * math.sqrt(252)

        return sortino

    def _calculate_max_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if not self._equity_curve:
            return 0.0, 0

        peak = self._equity_curve[0][1]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = None

        for dt, equity in self._equity_curve:
            if equity > peak:
                peak = equity
                current_dd_start = None
            else:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

                if current_dd_start is None:
                    current_dd_start = dt
                else:
                    duration = (dt - current_dd_start).days
                    if duration > max_dd_duration:
                        max_dd_duration = duration

        return max_dd, max_dd_duration

    def _calculate_max_consecutive(self, trades: List[TradeRecord], winners: bool) -> int:
        """Calculate max consecutive wins or losses"""
        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.is_winner == winners:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_grade(self, metrics: StrategyMetrics) -> StrategyGrade:
        """Calculate strategy grade based on metrics"""
        if metrics.sharpe_ratio < 0 or metrics.max_drawdown_pct > 0.30:
            return StrategyGrade.F

        for grade, thresholds in self.GRADE_THRESHOLDS.items():
            if (metrics.sharpe_ratio >= thresholds["min_sharpe"] and
                metrics.max_drawdown_pct <= thresholds["max_dd"]):
                return grade

        return StrategyGrade.D


class DynamicRiskManager:
    """
    Dynamic risk management with adaptive stop-loss and take-profit.
    Based on AI-Trader's intelligent risk control.
    """

    def __init__(
        self,
        initial_stop_loss_pct: float = 0.02,
        initial_take_profit_pct: float = 0.04,
        atr_multiplier: float = 2.0
    ):
        self.initial_stop_loss_pct = initial_stop_loss_pct
        self.initial_take_profit_pct = initial_take_profit_pct
        self.atr_multiplier = atr_multiplier

    def calculate_dynamic_stops(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        side: str,
        pnl_pct: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels.

        Returns:
            (stop_loss_price, take_profit_price)
        """
        atr_distance = atr * self.atr_multiplier

        if side == "long":
            # Initial stops
            stop_loss = entry_price - atr_distance
            take_profit = entry_price + atr_distance * 2

            # Trailing stop - move up as profit increases
            if pnl_pct > 0.01:  # >1% profit
                trailing_stop = current_price - atr_distance
                stop_loss = max(stop_loss, trailing_stop)

            if pnl_pct > 0.02:  # >2% profit, lock in half
                stop_loss = max(stop_loss, entry_price + (current_price - entry_price) * 0.5)

            if pnl_pct > 0.03:  # >3% profit, extend target
                take_profit = current_price + atr_distance * 1.5

        else:  # short
            stop_loss = entry_price + atr_distance
            take_profit = entry_price - atr_distance * 2

            if pnl_pct > 0.01:
                trailing_stop = current_price + atr_distance
                stop_loss = min(stop_loss, trailing_stop)

            if pnl_pct > 0.02:
                stop_loss = min(stop_loss, entry_price - (entry_price - current_price) * 0.5)

            if pnl_pct > 0.03:
                take_profit = current_price - atr_distance * 1.5

        return stop_loss, take_profit

    def should_exit(
        self,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        side: str,
        holding_minutes: int = 0,
        max_holding_minutes: int = 480  # 8 hours
    ) -> Tuple[bool, str]:
        """
        Determine if position should be exited.

        Returns:
            (should_exit, reason)
        """
        if side == "long":
            if current_price <= stop_loss:
                return True, "stop_loss"
            if current_price >= take_profit:
                return True, "take_profit"
        else:
            if current_price >= stop_loss:
                return True, "stop_loss"
            if current_price <= take_profit:
                return True, "take_profit"

        # Time-based exit
        if holding_minutes >= max_holding_minutes:
            return True, "max_holding_time"

        return False, ""


class RewardFunction:
    """
    Reward function for reinforcement learning optimization.
    Implements AI-Trader's reward shaping methodology.
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        sharpe_weight: float = 0.5,
        drawdown_penalty: float = 2.0,
        slippage_penalty: float = 1.0
    ):
        self.pnl_weight = pnl_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.slippage_penalty = slippage_penalty

    def calculate_trade_reward(
        self,
        pnl_pct: float,
        holding_period_minutes: int,
        slippage_pct: float = 0.0
    ) -> float:
        """
        Calculate reward for a single trade.
        """
        # Base reward from P&L
        reward = pnl_pct * self.pnl_weight * 100  # Scale up

        # Penalize excessive holding time
        if holding_period_minutes > 120:  # > 2 hours
            time_penalty = (holding_period_minutes - 120) / 120 * 0.1
            reward -= time_penalty

        # Penalize slippage
        if slippage_pct > 0.001:  # > 0.1%
            reward -= slippage_pct * self.slippage_penalty * 100

        return reward

    def calculate_episode_reward(
        self,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate: float
    ) -> float:
        """
        Calculate reward for a complete trading episode/day.
        """
        # Return component
        reward = total_return_pct * self.pnl_weight * 100

        # Sharpe component (bonus for risk-adjusted returns)
        if sharpe_ratio > 0:
            reward += sharpe_ratio * self.sharpe_weight

        # Drawdown penalty
        if max_drawdown_pct > 0.03:  # > 3%
            reward -= (max_drawdown_pct - 0.03) * self.drawdown_penalty * 100

        # Win rate bonus
        if win_rate > 0.6:
            reward += (win_rate - 0.5) * 10

        return reward

    def calculate_step_reward(
        self,
        action: str,
        price_change_pct: float,
        position_pnl_pct: float,
        signal_confidence: float
    ) -> float:
        """
        Calculate immediate reward for a trading step/action.
        """
        reward = 0.0

        if action == "hold":
            # Small reward for staying with correct position
            if position_pnl_pct > 0:
                reward = 0.1
            else:
                reward = -0.05

        elif action in ["long", "short"]:
            # Reward based on confidence and subsequent move
            if action == "long" and price_change_pct > 0:
                reward = price_change_pct * signal_confidence * 10
            elif action == "short" and price_change_pct < 0:
                reward = abs(price_change_pct) * signal_confidence * 10
            else:
                reward = -abs(price_change_pct) * 5  # Penalty for wrong direction

        elif action == "flat":
            # Reward for closing profitable position
            if position_pnl_pct > 0:
                reward = position_pnl_pct * 50
            else:
                # Small penalty for cutting loss early (but better than bigger loss)
                reward = position_pnl_pct * 10  # Negative, but scaled down

        return reward


# Export functions
def get_strategy_evaluator(starting_capital: float = 100000.0) -> StrategyEvaluator:
    """Get a new strategy evaluator instance"""
    return StrategyEvaluator(starting_capital)


def get_dynamic_risk_manager(
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04
) -> DynamicRiskManager:
    """Get a dynamic risk manager instance"""
    return DynamicRiskManager(stop_loss_pct, take_profit_pct)


def get_reward_function() -> RewardFunction:
    """Get the default reward function"""
    return RewardFunction()
