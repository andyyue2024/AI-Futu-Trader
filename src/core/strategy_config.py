"""
Strategy Configuration - Configurable trading strategy parameters
Allows zero-code strategy adjustment through configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import yaml
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Available strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    LLM_BASED = "llm_based"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Weight in signal combination

    # Thresholds
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": self.params,
            "weight": self.weight,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position sizing
    position_size_method: str = "fixed"  # fixed, percent, volatility, kelly
    fixed_position_size: float = 10000.0
    position_size_pct: float = 0.1  # 10% of portfolio
    max_position_pct: float = 0.25  # 25% max in single position

    # Stop loss
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2%
    stop_loss_atr_multiplier: float = 2.0  # 2x ATR
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.015  # 1.5%

    # Take profit
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4%
    take_profit_atr_multiplier: float = 3.0

    # Circuit breakers
    max_daily_drawdown: float = 0.03  # 3%
    max_total_drawdown: float = 0.15  # 15%
    max_daily_trades: int = 50
    max_daily_loss: float = 5000.0

    # Time-based
    max_holding_period_minutes: int = 480  # 8 hours
    no_new_positions_before_close_minutes: int = 30

    def to_dict(self) -> dict:
        return {
            "position_size_method": self.position_size_method,
            "fixed_position_size": self.fixed_position_size,
            "position_size_pct": self.position_size_pct,
            "max_position_pct": self.max_position_pct,
            "use_stop_loss": self.use_stop_loss,
            "stop_loss_pct": self.stop_loss_pct,
            "use_take_profit": self.use_take_profit,
            "take_profit_pct": self.take_profit_pct,
            "max_daily_drawdown": self.max_daily_drawdown,
            "max_total_drawdown": self.max_total_drawdown,
        }


@dataclass
class EntryConfig:
    """Entry signal configuration"""
    # Confirmation
    min_confidence: float = 0.6  # Minimum LLM confidence
    require_volume_confirmation: bool = True
    min_volume_ratio: float = 1.2  # 1.2x average volume

    # Trend filter
    require_trend_alignment: bool = True
    trend_indicator: str = "sma_20"  # Price must be above/below this

    # Momentum filter
    require_momentum: bool = True
    min_rsi: float = 25.0
    max_rsi: float = 75.0

    # Volatility filter
    max_spread_pct: float = 0.005  # Max 0.5% spread
    min_atr: float = 0.0  # Minimum volatility
    max_atr: float = float('inf')

    # Time filter
    allowed_sessions: List[str] = field(default_factory=lambda: ["regular", "pre_market", "after_hours"])

    def to_dict(self) -> dict:
        return {
            "min_confidence": self.min_confidence,
            "require_volume_confirmation": self.require_volume_confirmation,
            "min_volume_ratio": self.min_volume_ratio,
            "require_trend_alignment": self.require_trend_alignment,
            "max_spread_pct": self.max_spread_pct,
        }


@dataclass
class ExitConfig:
    """Exit signal configuration"""
    # Profit taking
    partial_profit_levels: List[Dict] = field(default_factory=list)
    # e.g., [{"pct": 0.02, "close_pct": 0.5}, {"pct": 0.04, "close_pct": 0.5}]

    # Time-based exit
    exit_before_close: bool = True
    exit_before_close_minutes: int = 15

    # Signal-based exit
    exit_on_signal_reversal: bool = True
    exit_on_momentum_loss: bool = True

    # Volatility-based exit
    exit_on_volatility_spike: bool = True
    volatility_spike_multiplier: float = 2.0

    def to_dict(self) -> dict:
        return {
            "partial_profit_levels": self.partial_profit_levels,
            "exit_before_close": self.exit_before_close,
            "exit_before_close_minutes": self.exit_before_close_minutes,
            "exit_on_signal_reversal": self.exit_on_signal_reversal,
        }


@dataclass
class StrategyConfig:
    """Complete strategy configuration"""
    # Identity
    name: str = "default"
    version: str = "1.0.0"
    description: str = ""
    strategy_type: StrategyType = StrategyType.LLM_BASED

    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["US.TQQQ", "US.QQQ"])

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.1

    # Indicators
    indicators: List[IndicatorConfig] = field(default_factory=list)

    # Risk, Entry, Exit configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)

    # Execution
    order_type: str = "market"  # market, limit, adaptive
    limit_offset_pct: float = 0.001  # For limit orders
    max_slippage_pct: float = 0.002  # 0.2%

    # Scheduling
    trading_interval_seconds: int = 60  # 1 minute
    rebalance_interval_minutes: int = 0  # 0 = no rebalancing

    def __post_init__(self):
        # Initialize default indicators if empty
        if not self.indicators:
            self.indicators = self._default_indicators()

    def _default_indicators(self) -> List[IndicatorConfig]:
        """Default indicator configuration"""
        return [
            IndicatorConfig(
                name="rsi",
                params={"period": 14},
                weight=1.0,
                buy_threshold=30.0,
                sell_threshold=70.0
            ),
            IndicatorConfig(
                name="macd",
                params={"fast": 12, "slow": 26, "signal": 9},
                weight=1.0
            ),
            IndicatorConfig(
                name="bollinger",
                params={"period": 20, "std": 2},
                weight=0.8
            ),
            IndicatorConfig(
                name="atr",
                params={"period": 14},
                weight=0.5
            ),
            IndicatorConfig(
                name="adx",
                params={"period": 14},
                weight=0.7
            ),
            IndicatorConfig(
                name="volume_sma",
                params={"period": 20},
                weight=0.6
            ),
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "strategy_type": self.strategy_type.value,
            "symbols": self.symbols,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "indicators": [i.to_dict() for i in self.indicators],
            "risk": self.risk.to_dict(),
            "entry": self.entry.to_dict(),
            "exit": self.exit.to_dict(),
            "order_type": self.order_type,
            "max_slippage_pct": self.max_slippage_pct,
            "trading_interval_seconds": self.trading_interval_seconds,
        }

    def save(self, filepath: str):
        """Save configuration to file"""
        path = Path(filepath)
        data = self.to_dict()

        if path.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in ('.yaml', '.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Strategy config saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'StrategyConfig':
        """Load configuration from file"""
        path = Path(filepath)

        if path.suffix == '.json':
            with open(filepath) as f:
                data = json.load(f)
        elif path.suffix in ('.yaml', '.yml'):
            with open(filepath) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyConfig':
        """Create from dictionary"""
        # Parse nested configs
        indicators = [
            IndicatorConfig(**ind)
            for ind in data.get('indicators', [])
        ]

        risk_data = data.get('risk', {})
        risk = RiskConfig(**risk_data) if risk_data else RiskConfig()

        entry_data = data.get('entry', {})
        entry = EntryConfig(**entry_data) if entry_data else EntryConfig()

        exit_data = data.get('exit', {})
        exit_config = ExitConfig(**exit_data) if exit_data else ExitConfig()

        strategy_type = StrategyType(data.get('strategy_type', 'llm_based'))

        return cls(
            name=data.get('name', 'default'),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            strategy_type=strategy_type,
            symbols=data.get('symbols', ['US.TQQQ', 'US.QQQ']),
            llm_provider=data.get('llm_provider', 'openai'),
            llm_model=data.get('llm_model', 'gpt-4-turbo-preview'),
            llm_temperature=data.get('llm_temperature', 0.1),
            indicators=indicators,
            risk=risk,
            entry=entry,
            exit=exit_config,
            order_type=data.get('order_type', 'market'),
            limit_offset_pct=data.get('limit_offset_pct', 0.001),
            max_slippage_pct=data.get('max_slippage_pct', 0.002),
            trading_interval_seconds=data.get('trading_interval_seconds', 60),
            rebalance_interval_minutes=data.get('rebalance_interval_minutes', 0),
        )


# Pre-built strategy configurations
STRATEGIES = {
    "aggressive_momentum": StrategyConfig(
        name="aggressive_momentum",
        description="Aggressive momentum strategy for leveraged ETFs",
        strategy_type=StrategyType.MOMENTUM,
        symbols=["US.TQQQ", "US.SOXL", "US.SPXL"],
        risk=RiskConfig(
            position_size_pct=0.15,
            max_position_pct=0.30,
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
        ),
        entry=EntryConfig(
            min_confidence=0.65,
            min_volume_ratio=1.5,
        ),
    ),

    "conservative_swing": StrategyConfig(
        name="conservative_swing",
        description="Conservative swing trading for large caps",
        strategy_type=StrategyType.TREND_FOLLOWING,
        symbols=["US.QQQ", "US.SPY", "US.AAPL"],
        risk=RiskConfig(
            position_size_pct=0.10,
            max_position_pct=0.20,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            max_daily_drawdown=0.02,
        ),
        entry=EntryConfig(
            min_confidence=0.75,
            require_trend_alignment=True,
        ),
    ),

    "mean_reversion": StrategyConfig(
        name="mean_reversion",
        description="Mean reversion strategy using Bollinger Bands",
        strategy_type=StrategyType.MEAN_REVERSION,
        symbols=["US.QQQ", "US.SPY"],
        risk=RiskConfig(
            position_size_pct=0.08,
            stop_loss_pct=0.02,
            take_profit_pct=0.02,
        ),
        entry=EntryConfig(
            min_confidence=0.7,
            min_rsi=20.0,
            max_rsi=80.0,
        ),
    ),
}


def get_strategy(name: str) -> StrategyConfig:
    """Get a pre-built strategy by name"""
    if name in STRATEGIES:
        return STRATEGIES[name]
    raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")


def list_strategies() -> List[str]:
    """List available pre-built strategies"""
    return list(STRATEGIES.keys())
