"""
Unit tests for backtest engine
"""
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd


class TestBacktestConfig:
    """Test BacktestConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        from src.backtest.engine import BacktestConfig

        config = BacktestConfig()

        assert config.starting_capital == 100000.0
        assert config.slippage_pct == 0.001
        assert config.max_daily_drawdown == 0.03
        assert config.max_total_drawdown == 0.15

    def test_custom_config(self):
        """Test custom configuration"""
        from src.backtest.engine import BacktestConfig

        config = BacktestConfig(
            starting_capital=50000.0,
            slippage_pct=0.002,
            max_position_pct=0.2
        )

        assert config.starting_capital == 50000.0
        assert config.slippage_pct == 0.002
        assert config.max_position_pct == 0.2


class TestSimulatedExecutor:
    """Test SimulatedExecutor class"""

    def test_executor_creation(self):
        """Test executor creation"""
        from src.backtest.engine import BacktestConfig, SimulatedExecutor

        config = BacktestConfig()
        executor = SimulatedExecutor(config)

        assert executor.config == config
        assert len(executor._positions) == 0

    def test_get_position(self):
        """Test position retrieval"""
        from src.backtest.engine import BacktestConfig, SimulatedExecutor

        config = BacktestConfig()
        executor = SimulatedExecutor(config)

        position = executor.get_position("US.TQQQ")

        assert position.futu_code == "US.TQQQ"
        assert position.is_flat is True


class TestBacktestEngine:
    """Test BacktestEngine class"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [50.0 + i * 0.01 for i in range(100)],
            'high': [50.1 + i * 0.01 for i in range(100)],
            'low': [49.9 + i * 0.01 for i in range(100)],
            'close': [50.0 + i * 0.01 for i in range(100)],
            'volume': [100000] * 100
        })
        return data

    def test_engine_creation(self):
        """Test engine creation"""
        from src.backtest.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig()
        engine = BacktestEngine(config)

        assert engine.config == config

    def test_load_data(self, sample_data):
        """Test loading data"""
        from src.backtest.engine import BacktestEngine

        engine = BacktestEngine()
        engine.load_data("US.TQQQ", sample_data)

        assert "US.TQQQ" in engine._data
        assert len(engine._data["US.TQQQ"]) == 100

    def test_simple_strategy_creation(self):
        """Test simple strategy creation"""
        from src.backtest.engine import create_simple_strategy

        strategy = create_simple_strategy(
            rsi_oversold=30,
            rsi_overbought=70,
            use_macd=True
        )

        assert callable(strategy)


class TestBacktestResult:
    """Test BacktestResult class"""

    def test_result_creation(self):
        """Test result creation"""
        from src.backtest.engine import BacktestResult, BacktestConfig
        from src.core.statistics import PerformanceMetrics

        result = BacktestResult(
            config=BacktestConfig(),
            metrics=PerformanceMetrics(sharpe_ratio=2.0, total_trades=50),
            equity_curve=[{"timestamp": "2024-01-01", "equity": 100000}],
            trades=[],
            daily_stats=[]
        )

        assert result.metrics.sharpe_ratio == 2.0
        assert len(result.equity_curve) == 1

    def test_result_to_dict(self):
        """Test result dictionary conversion"""
        from src.backtest.engine import BacktestResult, BacktestConfig
        from src.core.statistics import PerformanceMetrics

        result = BacktestResult(
            config=BacktestConfig(),
            metrics=PerformanceMetrics(sharpe_ratio=2.0),
        )

        data = result.to_dict()

        assert "config" in data
        assert "metrics" in data
