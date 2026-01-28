"""
AI-Trader Signal Aggregator
Combines multiple signal sources for robust trading decisions

Based on HKUDS AI-Trader multi-signal fusion methodology
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import math

from src.core.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    TECHNICAL = "technical"
    LLM = "llm"
    SENTIMENT = "sentiment"
    NEWS = "news"
    FLOW = "flow"  # Order flow
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0


@dataclass
class TradingSignal:
    """Individual trading signal"""
    signal_type: SignalType
    direction: str  # "long", "short", "flat", "hold"
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def numeric_direction(self) -> int:
        """Convert direction to numeric value"""
        if self.direction == "long":
            return 1
        elif self.direction == "short":
            return -1
        return 0

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score for aggregation"""
        return self.numeric_direction * self.confidence * self.strength.value


@dataclass
class AggregatedSignal:
    """Result of signal aggregation"""
    final_direction: str
    final_confidence: float
    total_score: float
    signal_count: int
    agreement_ratio: float  # How many signals agree
    contributing_signals: List[TradingSignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_strong(self) -> bool:
        """Check if aggregated signal is strong enough to act on"""
        return abs(self.total_score) > 2.0 and self.agreement_ratio > 0.6

    def to_dict(self) -> dict:
        return {
            "direction": self.final_direction,
            "confidence": self.final_confidence,
            "total_score": self.total_score,
            "signal_count": self.signal_count,
            "agreement_ratio": self.agreement_ratio,
            "is_strong": self.is_strong,
            "timestamp": self.timestamp.isoformat()
        }


class SignalAggregator:
    """
    Aggregates multiple trading signals into a single decision.
    Uses weighted voting and confidence-based fusion.
    """

    # Default weights for different signal types
    DEFAULT_WEIGHTS = {
        SignalType.LLM: 2.0,
        SignalType.TECHNICAL: 1.5,
        SignalType.SENTIMENT: 1.0,
        SignalType.NEWS: 0.8,
        SignalType.FLOW: 1.2,
        SignalType.MOMENTUM: 1.0,
        SignalType.MEAN_REVERSION: 0.9,
    }

    def __init__(self, weights: Dict[SignalType, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._signal_history: List[TradingSignal] = []
        self._max_history = 1000

    def add_signal(self, signal: TradingSignal):
        """Add a signal to the aggregator"""
        self._signal_history.append(signal)
        if len(self._signal_history) > self._max_history:
            self._signal_history.pop(0)

    def aggregate(
        self,
        signals: List[TradingSignal],
        min_confidence: float = 0.5,
        max_age_seconds: int = 300
    ) -> AggregatedSignal:
        """
        Aggregate multiple signals into a single decision.

        Args:
            signals: List of signals to aggregate
            min_confidence: Minimum confidence to include a signal
            max_age_seconds: Maximum age of signals to consider

        Returns:
            AggregatedSignal with the combined decision
        """
        now = datetime.now()

        # Filter signals
        valid_signals = [
            s for s in signals
            if s.confidence >= min_confidence
            and (now - s.timestamp).total_seconds() <= max_age_seconds
        ]

        if not valid_signals:
            return AggregatedSignal(
                final_direction="hold",
                final_confidence=0.0,
                total_score=0.0,
                signal_count=0,
                agreement_ratio=0.0
            )

        # Calculate weighted scores
        total_score = 0.0
        total_weight = 0.0
        direction_counts = {"long": 0, "short": 0, "flat": 0, "hold": 0}

        for signal in valid_signals:
            weight = self.weights.get(signal.signal_type, 1.0)
            weighted_score = signal.weighted_score * weight
            total_score += weighted_score
            total_weight += weight
            direction_counts[signal.direction] += 1

        # Determine final direction
        if total_score > 1.0:
            final_direction = "long"
        elif total_score < -1.0:
            final_direction = "short"
        elif abs(total_score) < 0.3:
            final_direction = "hold"
        else:
            # Use majority vote
            final_direction = max(direction_counts, key=direction_counts.get)

        # Calculate agreement ratio
        if final_direction in ["long", "short"]:
            agreed_count = direction_counts[final_direction]
        else:
            agreed_count = direction_counts["hold"] + direction_counts["flat"]

        agreement_ratio = agreed_count / len(valid_signals) if valid_signals else 0

        # Calculate combined confidence
        confidences = [s.confidence for s in valid_signals if s.direction == final_direction]
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Adjust confidence based on agreement
        final_confidence *= (0.5 + agreement_ratio * 0.5)

        return AggregatedSignal(
            final_direction=final_direction,
            final_confidence=min(1.0, final_confidence),
            total_score=total_score,
            signal_count=len(valid_signals),
            agreement_ratio=agreement_ratio,
            contributing_signals=valid_signals,
            timestamp=now
        )

    def get_signal_history(
        self,
        signal_type: SignalType = None,
        hours: int = 24
    ) -> List[TradingSignal]:
        """Get historical signals"""
        cutoff = datetime.now() - timedelta(hours=hours)

        signals = [
            s for s in self._signal_history
            if s.timestamp >= cutoff
        ]

        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        return signals


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for trading signals.
    Part of AI-Trader's multi-source decision framework.
    """

    # Keywords for sentiment analysis
    BULLISH_KEYWORDS = [
        "surge", "rally", "breakout", "bullish", "upgrade", "beat", "exceeds",
        "strong", "growth", "profit", "record", "optimistic", "buy", "long"
    ]

    BEARISH_KEYWORDS = [
        "crash", "plunge", "breakdown", "bearish", "downgrade", "miss", "below",
        "weak", "decline", "loss", "concern", "pessimistic", "sell", "short"
    ]

    def __init__(self):
        self._news_cache: List[Dict] = []
        self._sentiment_history: List[Tuple[datetime, float]] = []

    def analyze_headline(self, headline: str, symbol: str = None) -> Tuple[float, SignalStrength]:
        """
        Analyze a news headline for sentiment.

        Returns:
            (sentiment_score, strength) where score is -1 to 1
        """
        headline_lower = headline.lower()

        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in headline_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in headline_lower)

        total = bullish_count + bearish_count

        if total == 0:
            return 0.0, SignalStrength.NEUTRAL

        sentiment = (bullish_count - bearish_count) / total

        if abs(sentiment) > 0.7:
            strength = SignalStrength.VERY_STRONG
        elif abs(sentiment) > 0.5:
            strength = SignalStrength.STRONG
        elif abs(sentiment) > 0.3:
            strength = SignalStrength.MODERATE
        elif abs(sentiment) > 0.1:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK

        return sentiment, strength

    def generate_signal(
        self,
        headlines: List[str],
        symbol: str
    ) -> TradingSignal:
        """Generate a trading signal from news headlines"""
        if not headlines:
            return TradingSignal(
                signal_type=SignalType.NEWS,
                direction="hold",
                strength=SignalStrength.NEUTRAL,
                confidence=0.0,
                source="news",
                reasoning="No news available"
            )

        sentiments = [self.analyze_headline(h, symbol) for h in headlines]
        scores = [s[0] for s in sentiments]

        avg_sentiment = sum(scores) / len(scores)

        if avg_sentiment > 0.3:
            direction = "long"
        elif avg_sentiment < -0.3:
            direction = "short"
        else:
            direction = "hold"

        # Confidence based on consistency
        positive = sum(1 for s in scores if s > 0)
        negative = sum(1 for s in scores if s < 0)
        total = len(scores)

        consistency = max(positive, negative) / total if total > 0 else 0
        confidence = abs(avg_sentiment) * consistency

        # Determine strength
        max_strength = max(s[1].value for s in sentiments)
        strength = SignalStrength(max_strength)

        return TradingSignal(
            signal_type=SignalType.NEWS,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source="news_analyzer",
            reasoning=f"Analyzed {len(headlines)} headlines, avg sentiment: {avg_sentiment:.2f}",
            metadata={"headline_count": len(headlines), "avg_sentiment": avg_sentiment}
        )


class TechnicalSignalGenerator:
    """
    Generates technical analysis signals.
    Works with AI-Trader's indicator framework.
    """

    def __init__(self):
        pass

    def generate_rsi_signal(
        self,
        rsi: float,
        prev_rsi: float = None
    ) -> TradingSignal:
        """Generate signal from RSI"""
        if rsi < 30:
            direction = "long"
            strength = SignalStrength.STRONG if rsi < 20 else SignalStrength.MODERATE
            reasoning = f"RSI oversold at {rsi:.1f}"
        elif rsi > 70:
            direction = "short"
            strength = SignalStrength.STRONG if rsi > 80 else SignalStrength.MODERATE
            reasoning = f"RSI overbought at {rsi:.1f}"
        else:
            direction = "hold"
            strength = SignalStrength.WEAK
            reasoning = f"RSI neutral at {rsi:.1f}"

        confidence = abs(50 - rsi) / 50

        return TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source="rsi",
            reasoning=reasoning,
            metadata={"rsi": rsi}
        )

    def generate_macd_signal(
        self,
        macd: float,
        signal_line: float,
        histogram: float,
        prev_histogram: float = None
    ) -> TradingSignal:
        """Generate signal from MACD"""
        # Check for crossover
        if prev_histogram is not None:
            if prev_histogram < 0 and histogram > 0:
                direction = "long"
                strength = SignalStrength.STRONG
                reasoning = "MACD bullish crossover"
            elif prev_histogram > 0 and histogram < 0:
                direction = "short"
                strength = SignalStrength.STRONG
                reasoning = "MACD bearish crossover"
            else:
                if histogram > 0:
                    direction = "long"
                    strength = SignalStrength.WEAK
                    reasoning = "MACD positive"
                else:
                    direction = "short"
                    strength = SignalStrength.WEAK
                    reasoning = "MACD negative"
        else:
            if histogram > 0:
                direction = "long"
                strength = SignalStrength.MODERATE
                reasoning = f"MACD histogram positive: {histogram:.4f}"
            elif histogram < 0:
                direction = "short"
                strength = SignalStrength.MODERATE
                reasoning = f"MACD histogram negative: {histogram:.4f}"
            else:
                direction = "hold"
                strength = SignalStrength.NEUTRAL
                reasoning = "MACD neutral"

        confidence = min(1.0, abs(histogram) * 10)

        return TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source="macd",
            reasoning=reasoning,
            metadata={"macd": macd, "histogram": histogram}
        )

    def generate_ma_crossover_signal(
        self,
        fast_ma: float,
        slow_ma: float,
        price: float
    ) -> TradingSignal:
        """Generate signal from moving average crossover"""
        if fast_ma > slow_ma and price > fast_ma:
            direction = "long"
            strength = SignalStrength.STRONG
            reasoning = "Price above rising MAs"
        elif fast_ma > slow_ma:
            direction = "long"
            strength = SignalStrength.MODERATE
            reasoning = "Fast MA above slow MA"
        elif fast_ma < slow_ma and price < fast_ma:
            direction = "short"
            strength = SignalStrength.STRONG
            reasoning = "Price below falling MAs"
        elif fast_ma < slow_ma:
            direction = "short"
            strength = SignalStrength.MODERATE
            reasoning = "Fast MA below slow MA"
        else:
            direction = "hold"
            strength = SignalStrength.NEUTRAL
            reasoning = "MAs converging"

        ma_diff_pct = abs(fast_ma - slow_ma) / slow_ma
        confidence = min(1.0, ma_diff_pct * 20)

        return TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source="ma_crossover",
            reasoning=reasoning,
            metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "price": price}
        )

    def generate_bollinger_signal(
        self,
        price: float,
        upper_band: float,
        lower_band: float,
        middle_band: float
    ) -> TradingSignal:
        """Generate signal from Bollinger Bands"""
        band_width = upper_band - lower_band
        position = (price - lower_band) / band_width if band_width > 0 else 0.5

        if position < 0.1:  # Near lower band
            direction = "long"
            strength = SignalStrength.STRONG
            reasoning = "Price at lower Bollinger Band"
        elif position < 0.3:
            direction = "long"
            strength = SignalStrength.MODERATE
            reasoning = "Price near lower Bollinger Band"
        elif position > 0.9:  # Near upper band
            direction = "short"
            strength = SignalStrength.STRONG
            reasoning = "Price at upper Bollinger Band"
        elif position > 0.7:
            direction = "short"
            strength = SignalStrength.MODERATE
            reasoning = "Price near upper Bollinger Band"
        else:
            direction = "hold"
            strength = SignalStrength.WEAK
            reasoning = "Price within Bollinger Bands"

        confidence = abs(0.5 - position) * 2

        return TradingSignal(
            signal_type=SignalType.MEAN_REVERSION,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source="bollinger",
            reasoning=reasoning,
            metadata={"bb_position": position}
        )


# Singleton instances
_signal_aggregator: Optional[SignalAggregator] = None
_news_analyzer: Optional[NewsSentimentAnalyzer] = None
_technical_generator: Optional[TechnicalSignalGenerator] = None


def get_signal_aggregator() -> SignalAggregator:
    """Get global signal aggregator"""
    global _signal_aggregator
    if _signal_aggregator is None:
        _signal_aggregator = SignalAggregator()
    return _signal_aggregator


def get_news_analyzer() -> NewsSentimentAnalyzer:
    """Get global news analyzer"""
    global _news_analyzer
    if _news_analyzer is None:
        _news_analyzer = NewsSentimentAnalyzer()
    return _news_analyzer


def get_technical_generator() -> TechnicalSignalGenerator:
    """Get global technical signal generator"""
    global _technical_generator
    if _technical_generator is None:
        _technical_generator = TechnicalSignalGenerator()
    return _technical_generator
