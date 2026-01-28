"""
AI-Trader Core - Enhanced Chain-of-Thought Trading Agent
Based on HKUDS/AI-Trader architecture

This module implements the core AI-Trader reasoning framework:
1. Multi-step Chain-of-Thought (CoT) reasoning
2. Market sentiment analysis
3. Multi-timeframe analysis
4. Self-reflection and learning
5. Uncertainty quantification
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import re

from src.core.logger import get_logger
from src.data.data_processor import MarketSnapshot
from src.action.futu_executor import TradingAction, Position

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"
    HIGH_VOLATILITY = "high_volatility"


class SentimentLevel(Enum):
    """Market sentiment level"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class CoTStep:
    """Single step in Chain-of-Thought reasoning"""
    step_number: int
    title: str
    analysis: str
    conclusion: str
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class CoTReasoning:
    """Complete Chain-of-Thought reasoning process"""
    steps: List[CoTStep] = field(default_factory=list)
    final_decision: str = ""
    overall_confidence: float = 0.0
    reasoning_time_ms: float = 0.0

    def to_text(self) -> str:
        """Convert to human-readable text"""
        lines = ["=== Chain-of-Thought Reasoning ===\n"]
        for step in self.steps:
            lines.append(f"Step {step.step_number}: {step.title}")
            lines.append(f"  Analysis: {step.analysis}")
            lines.append(f"  Conclusion: {step.conclusion}")
            lines.append(f"  Confidence: {step.confidence:.1%}")
            lines.append("")
        lines.append(f"Final Decision: {self.final_decision}")
        lines.append(f"Overall Confidence: {self.overall_confidence:.1%}")
        return "\n".join(lines)


@dataclass
class MultiTimeframeAnalysis:
    """Multi-timeframe analysis result"""
    timeframe: str  # "1m", "5m", "15m", "1h", "1d"
    trend: str  # "up", "down", "sideways"
    strength: float  # 0-1
    key_levels: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)


@dataclass
class SentimentAnalysis:
    """Market sentiment analysis"""
    level: SentimentLevel
    score: float  # -1 to 1
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def is_extreme(self) -> bool:
        return self.level in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification for predictions"""
    epistemic: float  # Model uncertainty (lack of knowledge)
    aleatoric: float  # Data uncertainty (inherent noise)
    total: float = 0.0

    def __post_init__(self):
        self.total = (self.epistemic ** 2 + self.aleatoric ** 2) ** 0.5


class AITraderCore:
    """
    Core AI-Trader engine implementing HKUDS AI-Trader methodology.

    Key features:
    1. Chain-of-Thought (CoT) reasoning for transparent decision making
    2. Multi-timeframe analysis for robust signals
    3. Market regime detection
    4. Sentiment analysis
    5. Uncertainty quantification
    6. Self-reflection and learning
    """

    # AI-Trader decision thresholds
    CONFIDENCE_THRESHOLD = 0.6
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    UNCERTAINTY_THRESHOLD = 0.4

    # Risk parameters
    MAX_POSITION_PCT = 0.25
    DEFAULT_RISK_PER_TRADE = 0.02

    def __init__(self):
        self._trade_history: List[Dict] = []
        self._reflection_cache: Dict[str, Any] = {}
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []

    def generate_cot_reasoning(
        self,
        snapshot: MarketSnapshot,
        position: Optional[Position] = None,
    ) -> CoTReasoning:
        """
        Generate Chain-of-Thought reasoning for trading decision.

        This implements the core AI-Trader methodology:
        1. Observe market state
        2. Analyze technical indicators
        3. Assess market regime
        4. Consider position context
        5. Evaluate risk/reward
        6. Make final decision
        """
        steps = []

        # Step 1: Market Observation
        step1 = CoTStep(
            step_number=1,
            title="Market State Observation",
            analysis=self._analyze_market_state(snapshot),
            conclusion=self._conclude_market_state(snapshot),
            confidence=0.9,
            supporting_evidence=[
                f"Price: ${snapshot.last_price:.2f}",
                f"Volume: {snapshot.volume:,}",
                f"Spread: {snapshot.spread:.4f}"
            ]
        )
        steps.append(step1)

        # Step 2: Technical Analysis
        step2 = CoTStep(
            step_number=2,
            title="Technical Indicator Analysis",
            analysis=self._analyze_technicals(snapshot),
            conclusion=self._conclude_technicals(snapshot),
            confidence=self._calculate_technical_confidence(snapshot),
            supporting_evidence=self._get_technical_evidence(snapshot)
        )
        steps.append(step2)

        # Step 3: Trend Analysis
        step3 = CoTStep(
            step_number=3,
            title="Trend and Momentum Analysis",
            analysis=self._analyze_trend(snapshot),
            conclusion=self._conclude_trend(snapshot),
            confidence=self._calculate_trend_confidence(snapshot),
            supporting_evidence=self._get_trend_evidence(snapshot)
        )
        steps.append(step3)

        # Step 4: Risk Assessment
        step4 = CoTStep(
            step_number=4,
            title="Risk Assessment",
            analysis=self._analyze_risk(snapshot, position),
            conclusion=self._conclude_risk(snapshot, position),
            confidence=0.85,
            supporting_evidence=self._get_risk_evidence(snapshot, position)
        )
        steps.append(step4)

        # Step 5: Position Context
        if position and not position.is_flat:
            step5 = CoTStep(
                step_number=5,
                title="Position Context Analysis",
                analysis=self._analyze_position(snapshot, position),
                conclusion=self._conclude_position(snapshot, position),
                confidence=0.9,
                supporting_evidence=self._get_position_evidence(position)
            )
            steps.append(step5)

        # Calculate overall confidence
        confidences = [s.confidence for s in steps]
        overall_confidence = sum(confidences) / len(confidences)

        # Make final decision
        final_decision = self._make_final_decision(steps, snapshot, position)

        return CoTReasoning(
            steps=steps,
            final_decision=final_decision,
            overall_confidence=overall_confidence
        )

    def detect_market_regime(self, snapshot: MarketSnapshot) -> MarketRegime:
        """Detect current market regime"""
        indicators = snapshot.indicators

        # Check volatility first
        if indicators.atr_14 > snapshot.last_price * 0.03:
            return MarketRegime.HIGH_VOLATILITY

        # Check trend strength
        adx = indicators.adx_14 if hasattr(indicators, 'adx_14') else 25

        # RSI-based regime
        rsi = indicators.rsi_14
        macd_hist = indicators.macd_histogram

        if rsi > 70 and macd_hist > 0 and adx > 30:
            return MarketRegime.STRONG_BULLISH
        elif rsi > 55 and macd_hist > 0:
            return MarketRegime.BULLISH
        elif rsi < 30 and macd_hist < 0 and adx > 30:
            return MarketRegime.STRONG_BEARISH
        elif rsi < 45 and macd_hist < 0:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def analyze_sentiment(self, snapshot: MarketSnapshot) -> SentimentAnalysis:
        """Analyze market sentiment from technical factors"""
        indicators = snapshot.indicators

        factors = {}

        # RSI factor
        rsi = indicators.rsi_14
        if rsi > 80:
            factors["rsi"] = 0.9  # Extreme greed
        elif rsi > 65:
            factors["rsi"] = 0.5
        elif rsi < 20:
            factors["rsi"] = -0.9  # Extreme fear
        elif rsi < 35:
            factors["rsi"] = -0.5
        else:
            factors["rsi"] = 0.0

        # Volume factor
        vol_ratio = indicators.volume_ratio
        if vol_ratio > 2.0:
            factors["volume"] = 0.3 if snapshot.change_day > 0 else -0.3
        else:
            factors["volume"] = 0.0

        # Price momentum factor
        if snapshot.change_day > 0.02:
            factors["momentum"] = 0.5
        elif snapshot.change_day < -0.02:
            factors["momentum"] = -0.5
        else:
            factors["momentum"] = 0.0

        # Bollinger position
        bb_position = indicators.bollinger_position
        if bb_position > 0.9:
            factors["bollinger"] = 0.6
        elif bb_position < 0.1:
            factors["bollinger"] = -0.6
        else:
            factors["bollinger"] = 0.0

        # Calculate overall score
        score = sum(factors.values()) / max(len(factors), 1)

        # Determine level
        if score > 0.7:
            level = SentimentLevel.EXTREME_GREED
        elif score > 0.3:
            level = SentimentLevel.GREED
        elif score < -0.7:
            level = SentimentLevel.EXTREME_FEAR
        elif score < -0.3:
            level = SentimentLevel.FEAR
        else:
            level = SentimentLevel.NEUTRAL

        return SentimentAnalysis(level=level, score=score, factors=factors)

    def multi_timeframe_analysis(
        self,
        snapshots: Dict[str, MarketSnapshot]
    ) -> List[MultiTimeframeAnalysis]:
        """Perform multi-timeframe analysis"""
        results = []

        for tf, snap in snapshots.items():
            # Determine trend
            if snap.indicators.sma_5 > snap.indicators.sma_20:
                if snap.indicators.macd_histogram > 0:
                    trend = "up"
                    strength = min(1.0, snap.indicators.adx_14 / 50) if hasattr(snap.indicators, 'adx_14') else 0.5
                else:
                    trend = "sideways"
                    strength = 0.3
            elif snap.indicators.sma_5 < snap.indicators.sma_20:
                if snap.indicators.macd_histogram < 0:
                    trend = "down"
                    strength = min(1.0, snap.indicators.adx_14 / 50) if hasattr(snap.indicators, 'adx_14') else 0.5
                else:
                    trend = "sideways"
                    strength = 0.3
            else:
                trend = "sideways"
                strength = 0.2

            # Key levels
            key_levels = {
                "resistance": snap.indicators.bollinger_upper,
                "support": snap.indicators.bollinger_lower,
                "pivot": snap.indicators.sma_20
            }

            # Signals
            signals = []
            if snap.indicators.rsi_14 < 30:
                signals.append("oversold")
            if snap.indicators.rsi_14 > 70:
                signals.append("overbought")
            if snap.indicators.macd_histogram > 0 and trend == "up":
                signals.append("bullish_momentum")
            if snap.indicators.macd_histogram < 0 and trend == "down":
                signals.append("bearish_momentum")

            results.append(MultiTimeframeAnalysis(
                timeframe=tf,
                trend=trend,
                strength=strength,
                key_levels=key_levels,
                signals=signals
            ))

        return results

    def estimate_uncertainty(
        self,
        snapshot: MarketSnapshot,
        decision_confidence: float
    ) -> UncertaintyEstimate:
        """Estimate uncertainty in the prediction"""
        # Epistemic uncertainty (model uncertainty)
        # Higher when indicators give conflicting signals
        indicators = snapshot.indicators

        signals = []
        signals.append(1 if indicators.rsi_14 > 50 else -1)
        signals.append(1 if indicators.macd_histogram > 0 else -1)
        signals.append(1 if indicators.sma_5 > indicators.sma_20 else -1)

        # Disagreement among indicators
        signal_agreement = abs(sum(signals)) / len(signals)
        epistemic = 1.0 - signal_agreement

        # Aleatoric uncertainty (data noise)
        # Higher during high volatility
        volatility = indicators.atr_14 / snapshot.last_price if snapshot.last_price > 0 else 0.02
        aleatoric = min(1.0, volatility * 20)  # Scale to 0-1

        return UncertaintyEstimate(epistemic=epistemic, aleatoric=aleatoric)

    def self_reflect(
        self,
        recent_trades: List[Dict],
        performance_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Self-reflection on trading performance.
        AI-Trader core feature for continuous learning.
        """
        if not recent_trades:
            return {"status": "insufficient_data"}

        # Analyze winning vs losing trades
        winners = [t for t in recent_trades if t.get("pnl", 0) > 0]
        losers = [t for t in recent_trades if t.get("pnl", 0) < 0]

        win_rate = len(winners) / len(recent_trades) if recent_trades else 0

        # Identify patterns
        winning_patterns = self._identify_patterns(winners)
        losing_patterns = self._identify_patterns(losers)

        # Suggested adjustments
        adjustments = []

        if win_rate < 0.4:
            adjustments.append("increase_confidence_threshold")
        if win_rate > 0.7:
            adjustments.append("consider_larger_positions")

        avg_win = sum(t.get("pnl", 0) for t in winners) / len(winners) if winners else 0
        avg_loss = abs(sum(t.get("pnl", 0) for t in losers) / len(losers)) if losers else 0

        if avg_loss > avg_win * 1.5:
            adjustments.append("tighten_stop_losses")

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "winning_patterns": winning_patterns,
            "losing_patterns": losing_patterns,
            "adjustments": adjustments,
            "confidence_calibration": "decrease" if win_rate < 0.5 else "maintain"
        }

    # Private helper methods

    def _analyze_market_state(self, snapshot: MarketSnapshot) -> str:
        return f"Current price ${snapshot.last_price:.2f} with {snapshot.change_day:+.2%} daily change. Volume ratio: {snapshot.indicators.volume_ratio:.1f}x average."

    def _conclude_market_state(self, snapshot: MarketSnapshot) -> str:
        if snapshot.change_day > 0.01:
            return "Market showing bullish momentum"
        elif snapshot.change_day < -0.01:
            return "Market showing bearish pressure"
        return "Market in consolidation"

    def _analyze_technicals(self, snapshot: MarketSnapshot) -> str:
        ind = snapshot.indicators
        return f"RSI at {ind.rsi_14:.1f}, MACD histogram {ind.macd_histogram:+.4f}, price {'above' if snapshot.last_price > ind.sma_20 else 'below'} 20-SMA."

    def _conclude_technicals(self, snapshot: MarketSnapshot) -> str:
        ind = snapshot.indicators
        if ind.rsi_14 > 70:
            return "Overbought - potential reversal or continuation with caution"
        elif ind.rsi_14 < 30:
            return "Oversold - potential bounce or continued weakness"
        elif ind.macd_histogram > 0:
            return "Bullish technical setup"
        elif ind.macd_histogram < 0:
            return "Bearish technical setup"
        return "Neutral technical conditions"

    def _calculate_technical_confidence(self, snapshot: MarketSnapshot) -> float:
        ind = snapshot.indicators
        # Higher confidence when indicators align
        signals = []
        signals.append(ind.rsi_14 > 50)
        signals.append(ind.macd_histogram > 0)
        signals.append(snapshot.last_price > ind.sma_20)

        agreement = sum(signals) / len(signals)
        return 0.5 + abs(agreement - 0.5)

    def _get_technical_evidence(self, snapshot: MarketSnapshot) -> List[str]:
        ind = snapshot.indicators
        return [
            f"RSI(14): {ind.rsi_14:.1f}",
            f"MACD Histogram: {ind.macd_histogram:+.4f}",
            f"SMA(20): ${ind.sma_20:.2f}",
            f"Bollinger %B: {ind.bollinger_position:.2%}"
        ]

    def _analyze_trend(self, snapshot: MarketSnapshot) -> str:
        ind = snapshot.indicators
        if ind.sma_5 > ind.sma_20 > ind.sma_50:
            return "Strong uptrend with aligned moving averages"
        elif ind.sma_5 < ind.sma_20 < ind.sma_50:
            return "Strong downtrend with aligned moving averages"
        return "Mixed trend signals, MAs not aligned"

    def _conclude_trend(self, snapshot: MarketSnapshot) -> str:
        ind = snapshot.indicators
        if ind.sma_5 > ind.sma_20:
            return "Short-term trend is UP"
        elif ind.sma_5 < ind.sma_20:
            return "Short-term trend is DOWN"
        return "Trend is SIDEWAYS"

    def _calculate_trend_confidence(self, snapshot: MarketSnapshot) -> float:
        ind = snapshot.indicators
        # Distance between MAs indicates trend strength
        ma_diff = abs(ind.sma_5 - ind.sma_20) / ind.sma_20
        return min(0.95, 0.5 + ma_diff * 10)

    def _get_trend_evidence(self, snapshot: MarketSnapshot) -> List[str]:
        ind = snapshot.indicators
        return [
            f"SMA(5): ${ind.sma_5:.2f}",
            f"SMA(20): ${ind.sma_20:.2f}",
            f"SMA(50): ${ind.sma_50:.2f}"
        ]

    def _analyze_risk(self, snapshot: MarketSnapshot, position: Optional[Position]) -> str:
        atr = snapshot.indicators.atr_14
        volatility_pct = atr / snapshot.last_price * 100
        return f"Current volatility {volatility_pct:.2f}% (ATR-based). Spread: ${snapshot.spread:.4f}"

    def _conclude_risk(self, snapshot: MarketSnapshot, position: Optional[Position]) -> str:
        atr = snapshot.indicators.atr_14
        volatility_pct = atr / snapshot.last_price * 100

        if volatility_pct > 3:
            return "HIGH RISK - Elevated volatility"
        elif volatility_pct > 1.5:
            return "MEDIUM RISK - Normal volatility"
        return "LOW RISK - Low volatility environment"

    def _get_risk_evidence(self, snapshot: MarketSnapshot, position: Optional[Position]) -> List[str]:
        evidence = [
            f"ATR(14): ${snapshot.indicators.atr_14:.4f}",
            f"Spread: ${snapshot.spread:.4f}"
        ]
        if position and not position.is_flat:
            evidence.append(f"Position P&L: ${position.unrealized_pnl:.2f}")
        return evidence

    def _analyze_position(self, snapshot: MarketSnapshot, position: Position) -> str:
        pnl_pct = position.unrealized_pnl / position.market_value * 100 if position.market_value > 0 else 0
        direction = "LONG" if position.is_long else "SHORT"
        return f"Current {direction} position with {pnl_pct:+.2f}% unrealized P&L"

    def _conclude_position(self, snapshot: MarketSnapshot, position: Position) -> str:
        pnl_pct = position.unrealized_pnl / position.market_value * 100 if position.market_value > 0 else 0

        if pnl_pct > 5:
            return "Consider taking partial profits"
        elif pnl_pct < -3:
            return "Consider stop-loss or position reduction"
        return "Position within normal range"

    def _get_position_evidence(self, position: Position) -> List[str]:
        return [
            f"Quantity: {position.quantity}",
            f"Avg Cost: ${position.avg_cost:.4f}",
            f"Market Value: ${position.market_value:.2f}",
            f"Unrealized P&L: ${position.unrealized_pnl:.2f}"
        ]

    def _make_final_decision(
        self,
        steps: List[CoTStep],
        snapshot: MarketSnapshot,
        position: Optional[Position]
    ) -> str:
        """Synthesize steps into final decision"""
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0

        for step in steps:
            conclusion = step.conclusion.lower()
            if any(word in conclusion for word in ["bullish", "up", "buy", "long", "bounce"]):
                bullish_signals += step.confidence
            elif any(word in conclusion for word in ["bearish", "down", "sell", "short", "decline"]):
                bearish_signals += step.confidence

        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return "HOLD"

        bullish_ratio = bullish_signals / total_signals

        # Consider existing position
        if position and not position.is_flat:
            if position.is_long and bullish_ratio < 0.3:
                return "FLAT"  # Close long if bearish
            elif position.is_short and bullish_ratio > 0.7:
                return "FLAT"  # Close short if bullish

        # Make decision
        if bullish_ratio > 0.65:
            return "LONG"
        elif bullish_ratio < 0.35:
            return "SHORT"
        else:
            return "HOLD"

    def _identify_patterns(self, trades: List[Dict]) -> List[str]:
        """Identify patterns in trades"""
        patterns = []

        if not trades:
            return patterns

        # Time patterns
        morning_trades = [t for t in trades if t.get("hour", 12) < 11]
        if len(morning_trades) / len(trades) > 0.6:
            patterns.append("morning_trading_bias")

        # Symbol patterns
        symbols = [t.get("symbol") for t in trades]
        if symbols:
            from collections import Counter
            most_common = Counter(symbols).most_common(1)[0]
            if most_common[1] / len(trades) > 0.5:
                patterns.append(f"concentrated_in_{most_common[0]}")

        return patterns


# Singleton instance
_ai_trader_core: Optional[AITraderCore] = None


def get_ai_trader_core() -> AITraderCore:
    """Get global AI-Trader core instance"""
    global _ai_trader_core
    if _ai_trader_core is None:
        _ai_trader_core = AITraderCore()
    return _ai_trader_core
