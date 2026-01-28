"""
Unit tests for strategy configuration
"""
import pytest
import tempfile
import json
from pathlib import Path


class TestStrategyType:
    """Test StrategyType enum"""

    def test_strategy_type_values(self):
        """Test strategy type values"""
        from src.core.strategy_config import StrategyType

        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.LLM_BASED.value == "llm_based"


class TestIndicatorConfig:
    """Test IndicatorConfig class"""

    def test_indicator_creation(self):
        """Test indicator config creation"""
        from src.core.strategy_config import IndicatorConfig

        config = IndicatorConfig(
            name="rsi",
            params={"period": 14},
            weight=1.0,
            buy_threshold=30.0,
            sell_threshold=70.0
        )

        assert config.name == "rsi"
        assert config.params["period"] == 14

    def test_indicator_to_dict(self):
        """Test indicator dictionary conversion"""
        from src.core.strategy_config import IndicatorConfig

        config = IndicatorConfig(
            name="macd",
            params={"fast": 12, "slow": 26},
            enabled=True
        )

        data = config.to_dict()

        assert data["name"] == "macd"
        assert data["enabled"] is True
        assert "params" in data


class TestRiskConfig:
    """Test RiskConfig class"""

    def test_risk_config_defaults(self):
        """Test risk config default values"""
        from src.core.strategy_config import RiskConfig

        config = RiskConfig()

        assert config.position_size_method == "fixed"
        assert config.max_daily_drawdown == 0.03
        assert config.max_total_drawdown == 0.15
        assert config.use_stop_loss is True

    def test_risk_config_custom(self):
        """Test custom risk config"""
        from src.core.strategy_config import RiskConfig

        config = RiskConfig(
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
            trailing_stop=True
        )

        assert config.stop_loss_pct == 0.025
        assert config.take_profit_pct == 0.05
        assert config.trailing_stop is True


class TestEntryConfig:
    """Test EntryConfig class"""

    def test_entry_config_defaults(self):
        """Test entry config defaults"""
        from src.core.strategy_config import EntryConfig

        config = EntryConfig()

        assert config.min_confidence == 0.6
        assert config.require_volume_confirmation is True
        assert config.require_trend_alignment is True

    def test_entry_config_to_dict(self):
        """Test entry config dictionary conversion"""
        from src.core.strategy_config import EntryConfig

        config = EntryConfig(min_confidence=0.75)
        data = config.to_dict()

        assert data["min_confidence"] == 0.75


class TestExitConfig:
    """Test ExitConfig class"""

    def test_exit_config_defaults(self):
        """Test exit config defaults"""
        from src.core.strategy_config import ExitConfig

        config = ExitConfig()

        assert config.exit_before_close is True
        assert config.exit_on_signal_reversal is True

    def test_partial_profit_levels(self):
        """Test partial profit configuration"""
        from src.core.strategy_config import ExitConfig

        config = ExitConfig(
            partial_profit_levels=[
                {"pct": 0.02, "close_pct": 0.5},
                {"pct": 0.04, "close_pct": 0.5}
            ]
        )

        assert len(config.partial_profit_levels) == 2


class TestStrategyConfig:
    """Test StrategyConfig class"""

    def test_strategy_config_creation(self):
        """Test strategy config creation"""
        from src.core.strategy_config import StrategyConfig

        config = StrategyConfig(
            name="test_strategy",
            symbols=["US.TQQQ", "US.QQQ"]
        )

        assert config.name == "test_strategy"
        assert len(config.symbols) == 2
        assert len(config.indicators) > 0  # Default indicators

    def test_strategy_config_to_dict(self):
        """Test strategy config dictionary conversion"""
        from src.core.strategy_config import StrategyConfig

        config = StrategyConfig(name="test")
        data = config.to_dict()

        assert data["name"] == "test"
        assert "risk" in data
        assert "entry" in data
        assert "exit" in data
        assert "indicators" in data

    def test_strategy_config_save_load_json(self):
        """Test saving and loading JSON config"""
        from src.core.strategy_config import StrategyConfig

        config = StrategyConfig(
            name="test_save",
            symbols=["US.AAPL"]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            config.save(filepath)

            loaded = StrategyConfig.load(filepath)

            assert loaded.name == "test_save"
            assert "US.AAPL" in loaded.symbols
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_strategy_from_dict(self):
        """Test creating strategy from dictionary"""
        from src.core.strategy_config import StrategyConfig

        data = {
            "name": "from_dict",
            "symbols": ["US.MSFT"],
            "llm_provider": "anthropic",
            "risk": {
                "stop_loss_pct": 0.02
            }
        }

        config = StrategyConfig.from_dict(data)

        assert config.name == "from_dict"
        assert config.llm_provider == "anthropic"
        assert config.risk.stop_loss_pct == 0.02


class TestPrebuiltStrategies:
    """Test pre-built strategies"""

    def test_list_strategies(self):
        """Test listing available strategies"""
        from src.core.strategy_config import list_strategies

        strategies = list_strategies()

        assert len(strategies) >= 3
        assert "aggressive_momentum" in strategies
        assert "conservative_swing" in strategies

    def test_get_strategy(self):
        """Test getting a pre-built strategy"""
        from src.core.strategy_config import get_strategy

        strategy = get_strategy("aggressive_momentum")

        assert strategy.name == "aggressive_momentum"
        assert "US.TQQQ" in strategy.symbols

    def test_get_invalid_strategy(self):
        """Test getting an invalid strategy"""
        from src.core.strategy_config import get_strategy

        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")
