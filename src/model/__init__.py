"""
Model module - LLM-based trading decision engine
Based on HKUDS AI-Trader architecture
"""
from .llm_agent import LLMAgent, TradingDecision, DecisionConfidence, EnhancedLLMAgent
from .prompts import PromptTemplates
from .ai_trader_core import (
    AITraderCore,
    CoTReasoning,
    CoTStep,
    MarketRegime,
    SentimentLevel,
    SentimentAnalysis,
    MultiTimeframeAnalysis,
    UncertaintyEstimate,
    get_ai_trader_core,
)
from .strategy_evaluator import (
    StrategyEvaluator,
    StrategyMetrics,
    StrategyGrade,
    TradeRecord,
    BacktestVsLiveComparison,
    DynamicRiskManager,
    RewardFunction,
    get_strategy_evaluator,
    get_dynamic_risk_manager,
    get_reward_function,
)
from .signal_aggregator import (
    SignalAggregator,
    TradingSignal,
    AggregatedSignal,
    SignalType,
    SignalStrength,
    NewsSentimentAnalyzer,
    TechnicalSignalGenerator,
    get_signal_aggregator,
    get_news_analyzer,
    get_technical_generator,
)
from .memory import (
    TradingMemory,
    MemoryType,
    MemoryEntry,
    TradeMemory,
    PatternMemory,
    ReflectionModule,
    get_trading_memory,
    get_reflection_module,
)
from .action_parser import (
    ActionParser,
    ParsedAction,
    ActionType,
    ActionParseError,
    EnvironmentInterface,
    get_action_parser,
    get_environment,
)

__all__ = [
    # LLM Agent
    "LLMAgent",
    "EnhancedLLMAgent",
    "TradingDecision",
    "DecisionConfidence",
    "PromptTemplates",
    # AI-Trader Core
    "AITraderCore",
    "CoTReasoning",
    "CoTStep",
    "MarketRegime",
    "SentimentLevel",
    "SentimentAnalysis",
    "MultiTimeframeAnalysis",
    "UncertaintyEstimate",
    "get_ai_trader_core",
    # Strategy Evaluator
    "StrategyEvaluator",
    "StrategyMetrics",
    "StrategyGrade",
    "TradeRecord",
    "BacktestVsLiveComparison",
    "DynamicRiskManager",
    "RewardFunction",
    "get_strategy_evaluator",
    "get_dynamic_risk_manager",
    "get_reward_function",
    # Signal Aggregator
    "SignalAggregator",
    "TradingSignal",
    "AggregatedSignal",
    "SignalType",
    "SignalStrength",
    "NewsSentimentAnalyzer",
    "TechnicalSignalGenerator",
    "get_signal_aggregator",
    "get_news_analyzer",
    "get_technical_generator",
    # Memory Module
    "TradingMemory",
    "MemoryType",
    "MemoryEntry",
    "TradeMemory",
    "PatternMemory",
    "ReflectionModule",
    "get_trading_memory",
    "get_reflection_module",
    # Action Parser
    "ActionParser",
    "ParsedAction",
    "ActionType",
    "ActionParseError",
    "EnvironmentInterface",
    "get_action_parser",
    "get_environment",
]
