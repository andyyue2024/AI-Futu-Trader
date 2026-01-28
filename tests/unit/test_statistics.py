"""
Unit tests for trading statistics
"""
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch


class TestTrade:
    """Test Trade class"""

    def test_trade_creation(self):
        """Test Trade creation"""
        from src.core.statistics import Trade

        trade = Trade(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100
        )

        assert trade.trade_id == "test-001"
        assert trade.is_closed is False
        assert trade.pnl == 0.0

    def test_trade_close_long(self):
        """Test closing a long trade"""
        from src.core.statistics import Trade

        trade = Trade(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="long",
            quantity=100
        )

        # Close with profit
        trade.close(
            exit_time=datetime.now() + timedelta(minutes=30),
            exit_price=52.0
        )

        assert trade.is_closed is True
        assert trade.pnl == 200.0  # (52-50) * 100
        assert trade.pnl_pct == pytest.approx(0.04, rel=0.001)
        assert trade.is_winner is True

    def test_trade_close_short(self):
        """Test closing a short trade"""
        from src.core.statistics import Trade

        trade = Trade(
            trade_id="test-002",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            entry_time=datetime.now(),
            entry_price=50.0,
            entry_side="short",
            quantity=100
        )

        # Close with profit (price went down)
        trade.close(
            exit_time=datetime.now() + timedelta(minutes=30),
            exit_price=48.0
        )

        assert trade.pnl == 200.0  # (50-48) * 100
        assert trade.is_winner is True


class TestDailyStats:
    """Test DailyStats class"""

    def test_daily_stats_creation(self):
        """Test DailyStats creation"""
        from src.core.statistics import DailyStats

        stats = DailyStats(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=101000.0,
            realized_pnl=1000.0,
            total_trades=10,
            winning_trades=7,
            losing_trades=3
        )

        assert stats.realized_pnl == 1000.0
        assert stats.total_trades == 10

    def test_daily_stats_to_dict(self):
        """Test DailyStats dictionary conversion"""
        from src.core.statistics import DailyStats

        stats = DailyStats(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=101000.0,
            pnl_pct=0.01,
            win_rate=0.7
        )

        data = stats.to_dict()

        assert "date" in data
        assert "pnl_pct" in data
        assert data["win_rate"] == 70.0


class TestTradingStatistics:
    """Test TradingStatistics class"""

    def test_statistics_creation(self):
        """Test TradingStatistics creation"""
        from src.core.statistics import TradingStatistics

        stats = TradingStatistics(starting_equity=100000.0)

        assert stats.starting_equity == 100000.0
        assert stats._current_equity == 100000.0

    def test_record_entry_and_exit(self):
        """Test recording entries and exits"""
        from src.core.statistics import TradingStatistics

        stats = TradingStatistics(starting_equity=100000.0)

        # Record entry
        trade = stats.record_entry(
            trade_id="test-001",
            symbol="TQQQ",
            futu_code="US.TQQQ",
            side="long",
            quantity=100,
            entry_price=50.0
        )

        assert "US.TQQQ" in stats._open_trades

        # Record exit
        closed = stats.record_exit(
            futu_code="US.TQQQ",
            exit_price=52.0
        )

        assert closed is not None
        assert closed.pnl == 200.0
        assert "US.TQQQ" not in stats._open_trades

    def test_calculate_metrics(self):
        """Test metrics calculation"""
        from src.core.statistics import TradingStatistics

        stats = TradingStatistics(starting_equity=100000.0)

        # Record some trades
        for i in range(10):
            pnl = 100 if i < 6 else -80  # 60% win rate
            entry_price = 50.0
            exit_price = 50.0 + (pnl / 100)

            stats.record_entry(
                trade_id=f"test-{i}",
                symbol="TQQQ",
                futu_code="US.TQQQ",
                side="long",
                quantity=100,
                entry_price=entry_price
            )
            stats.record_exit("US.TQQQ", exit_price)

        metrics = stats.calculate_metrics()

        assert metrics.total_trades == 10
        assert metrics.winning_trades == 6
        assert metrics.win_rate == pytest.approx(0.6, rel=0.01)

    def test_generate_report(self):
        """Test report generation"""
        from src.core.statistics import TradingStatistics

        stats = TradingStatistics(starting_equity=100000.0)

        # Add a trade
        stats.record_entry("t1", "TQQQ", "US.TQQQ", "long", 100, 50.0)
        stats.record_exit("US.TQQQ", 51.0)

        report = stats.generate_report()

        assert "TRADING PERFORMANCE REPORT" in report
        assert "Sharpe Ratio" in report


class TestPerformanceMetrics:
    """Test PerformanceMetrics class"""

    def test_metrics_to_dict(self):
        """Test metrics dictionary conversion"""
        from src.core.statistics import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=0.01,
            sharpe_ratio=2.5,
            win_rate=0.65
        )

        data = metrics.to_dict()

        assert data["sharpe_ratio"] == 2.5
        assert data["total_return_pct"] == 1.0
        assert data["win_rate"] == 65.0

    def test_metrics_summary(self):
        """Test metrics summary generation"""
        from src.core.statistics import PerformanceMetrics
        from datetime import date

        metrics = PerformanceMetrics(
            total_return=5000.0,
            total_return_pct=0.05,
            sharpe_ratio=2.1,
            max_drawdown_pct=0.03,
            total_trades=50,
            win_rate=0.62,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            trading_days=22
        )

        summary = metrics.summary()

        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Win Rate" in summary
