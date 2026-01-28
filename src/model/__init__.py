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

__all__ = [
    "LLMAgent",
    "EnhancedLLMAgent",
    "TradingDecision",
    "DecisionConfidence",
    "PromptTemplates",
    "AITraderCore",
    "CoTReasoning",
    "CoTStep",
    "MarketRegime",
    "SentimentLevel",
    "SentimentAnalysis",
    "MultiTimeframeAnalysis",
    "UncertaintyEstimate",
    "get_ai_trader_core",
]
