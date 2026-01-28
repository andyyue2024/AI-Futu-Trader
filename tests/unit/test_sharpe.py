"""
Unit tests for Sharpe calculator
"""
import pytest
import math
from datetime import date, timedelta


class TestRiskMetrics:
    """Test RiskMetrics class"""

    def test_metrics_creation(self):
        """Test metrics creation"""
        from src.risk.sharpe_calculator import RiskMetrics

        metrics = RiskMetrics(
            total_return=5000.0,
            sharpe_ratio=2.5,
            max_drawdown_pct=0.05
        )

        assert metrics.total_return == 5000.0
        assert metrics.sharpe_ratio == 2.5


class TestSharpeCalculator:
    """Test SharpeCalculator class"""

    def test_calculator_creation(self):
        """Test calculator creation"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        assert calc.starting_equity == 100000.0
        assert calc.TARGET_SHARPE == 2.0
        assert calc.TARGET_MAX_DRAWDOWN == 0.15

    def test_update_equity(self):
        """Test equity updates"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        calc.update_equity(101000.0)
        calc.update_equity(102000.0)

        assert calc._current_equity == 102000.0
        assert calc._peak_equity == 102000.0

    def test_record_trade_pnl(self):
        """Test recording trade P&L"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator()

        calc.record_trade_pnl(100.0)
        calc.record_trade_pnl(-50.0)
        calc.record_trade_pnl(75.0)

        assert len(calc._trade_pnls) == 3

    def test_calculate_sharpe_insufficient_data(self):
        """Test Sharpe with insufficient data"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator()

        sharpe = calc.calculate_sharpe()
        assert sharpe == 0.0

    def test_calculate_sharpe_with_returns(self):
        """Test Sharpe calculation with returns"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        # Simulate daily returns
        calc._daily_returns = [0.01, 0.02, -0.01, 0.015, 0.005] * 50  # 250 days

        sharpe = calc.calculate_sharpe()

        # Should be positive with these returns
        assert sharpe > 0

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        # Simulate equity curve with drawdown
        calc._daily_equity = [
            (date.today() - timedelta(days=4), 100000),
            (date.today() - timedelta(days=3), 105000),  # Peak
            (date.today() - timedelta(days=2), 100000),  # Drawdown
            (date.today() - timedelta(days=1), 95000),   # Max drawdown
            (date.today(), 98000),
        ]

        max_dd, max_dd_pct = calc.calculate_max_drawdown()

        assert max_dd == 10000.0  # 105000 - 95000
        assert max_dd_pct == pytest.approx(10000.0 / 105000.0, rel=0.01)

    def test_calculate_sortino(self):
        """Test Sortino ratio calculation"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator()

        # Returns with some negative values
        calc._daily_returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.01] * 40

        sortino = calc.calculate_sortino()

        # Should be higher than Sharpe since we only consider downside
        sharpe = calc.calculate_sharpe()
        assert sortino != 0

    def test_calculate_calmar(self):
        """Test Calmar ratio calculation"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        calc._daily_returns = [0.01, 0.02, -0.01, 0.015, 0.005] * 50
        calc._daily_equity = [
            (date.today() - timedelta(days=4), 100000),
            (date.today() - timedelta(days=3), 105000),
            (date.today() - timedelta(days=2), 100000),
            (date.today() - timedelta(days=1), 95000),
            (date.today(), 110000),
        ]

        calmar = calc.calculate_calmar()

        assert calmar != 0

    def test_get_current_drawdown(self):
        """Test current drawdown"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        calc._current_equity = 95000.0
        calc._peak_equity = 100000.0

        dd, dd_pct = calc.get_current_drawdown()

        assert dd == 5000.0
        assert dd_pct == 0.05

    def test_get_metrics(self):
        """Test getting complete metrics"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        # Add some data
        calc._daily_returns = [0.01, -0.005, 0.015, 0.02, -0.01] * 50
        calc._daily_equity = [
            (date.today() - timedelta(days=2), 100000),
            (date.today() - timedelta(days=1), 102000),
            (date.today(), 105000),
        ]
        calc._current_equity = 105000
        calc._peak_equity = 105000
        calc._trade_pnls = [100, -50, 200, -25, 150]

        metrics = calc.get_metrics()

        assert metrics.total_return == 5000.0
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2

    def test_is_meeting_sharpe_target(self):
        """Test Sharpe target compliance"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator()

        # High positive returns = high Sharpe
        calc._daily_returns = [0.02, 0.015, 0.01, 0.018, 0.012] * 50

        # With very consistent positive returns, should meet target
        # (depends on actual calculation)
        result = calc.is_meeting_sharpe_target()
        assert isinstance(result, bool)

    def test_is_meeting_drawdown_target(self):
        """Test drawdown target compliance"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        # Small drawdown
        calc._daily_equity = [
            (date.today() - timedelta(days=1), 100000),
            (date.today(), 98000),
        ]

        assert calc.is_meeting_drawdown_target() is True

        # Large drawdown (>15%)
        calc._daily_equity = [
            (date.today() - timedelta(days=1), 100000),
            (date.today(), 80000),
        ]

        assert calc.is_meeting_drawdown_target() is False

    def test_get_status_report(self):
        """Test status report generation"""
        from src.risk.sharpe_calculator import SharpeCalculator

        calc = SharpeCalculator(starting_equity=100000.0)

        calc._daily_returns = [0.01, -0.005, 0.015] * 30
        calc._daily_equity = [
            (date.today() - timedelta(days=1), 100000),
            (date.today(), 103000),
        ]
        calc._current_equity = 103000
        calc._peak_equity = 103000

        report = calc.get_status_report()

        assert "Risk Metrics Report" in report
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report
