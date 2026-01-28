"""
Unit tests for Signal Aggregator
"""
import pytest
from datetime import datetime, timedelta


class TestSignalType:
    """Test SignalType enum"""

    def test_signal_types(self):
        from src.model.signal_aggregator import SignalType

        assert SignalType.TECHNICAL.value == "technical"
        assert SignalType.LLM.value == "llm"
        assert SignalType.NEWS.value == "news"


class TestSignalStrength:
    """Test SignalStrength enum"""

    def test_strength_values(self):
        from src.model.signal_aggregator import SignalStrength

        assert SignalStrength.VERY_STRONG.value == 5
        assert SignalStrength.NEUTRAL.value == 0


class TestTradingSignal:
    """Test TradingSignal class"""

    def test_signal_creation(self):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        signal = TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction="long",
            strength=SignalStrength.STRONG,
            confidence=0.8,
            source="rsi",
            reasoning="RSI oversold"
        )

        assert signal.direction == "long"
        assert signal.confidence == 0.8

    def test_numeric_direction(self):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        long_signal = TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction="long",
            strength=SignalStrength.MODERATE,
            confidence=0.7
        )

        short_signal = TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction="short",
            strength=SignalStrength.MODERATE,
            confidence=0.7
        )

        assert long_signal.numeric_direction == 1
        assert short_signal.numeric_direction == -1

    def test_weighted_score(self):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        signal = TradingSignal(
            signal_type=SignalType.TECHNICAL,
            direction="long",
            strength=SignalStrength.STRONG,  # value = 4
            confidence=0.8
        )

        expected = 1 * 0.8 * 4  # direction * confidence * strength
        assert signal.weighted_score == pytest.approx(expected)


class TestAggregatedSignal:
    """Test AggregatedSignal class"""

    def test_is_strong(self):
        from src.model.signal_aggregator import AggregatedSignal

        strong = AggregatedSignal(
            final_direction="long",
            final_confidence=0.8,
            total_score=3.0,
            signal_count=5,
            agreement_ratio=0.8
        )

        weak = AggregatedSignal(
            final_direction="long",
            final_confidence=0.5,
            total_score=1.0,
            signal_count=5,
            agreement_ratio=0.4
        )

        assert strong.is_strong is True
        assert weak.is_strong is False


class TestSignalAggregator:
    """Test SignalAggregator class"""

    @pytest.fixture
    def aggregator(self):
        from src.model.signal_aggregator import SignalAggregator
        return SignalAggregator()

    @pytest.fixture
    def sample_signals(self):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        return [
            TradingSignal(
                signal_type=SignalType.TECHNICAL,
                direction="long",
                strength=SignalStrength.STRONG,
                confidence=0.8
            ),
            TradingSignal(
                signal_type=SignalType.LLM,
                direction="long",
                strength=SignalStrength.MODERATE,
                confidence=0.7
            ),
            TradingSignal(
                signal_type=SignalType.SENTIMENT,
                direction="long",
                strength=SignalStrength.WEAK,
                confidence=0.6
            ),
        ]

    def test_aggregator_creation(self, aggregator):
        assert aggregator.DEFAULT_WEIGHTS is not None

    def test_aggregate_long_signals(self, aggregator, sample_signals):
        result = aggregator.aggregate(sample_signals)

        assert result.final_direction == "long"
        assert result.signal_count == 3
        assert result.agreement_ratio == 1.0  # All agree

    def test_aggregate_mixed_signals(self, aggregator):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        signals = [
            TradingSignal(SignalType.TECHNICAL, "long", SignalStrength.STRONG, 0.8),
            TradingSignal(SignalType.LLM, "short", SignalStrength.STRONG, 0.8),
            TradingSignal(SignalType.SENTIMENT, "hold", SignalStrength.WEAK, 0.5),
        ]

        result = aggregator.aggregate(signals)

        # Agreement should be low
        assert result.agreement_ratio < 0.5

    def test_aggregate_with_min_confidence(self, aggregator):
        from src.model.signal_aggregator import TradingSignal, SignalType, SignalStrength

        signals = [
            TradingSignal(SignalType.TECHNICAL, "long", SignalStrength.STRONG, 0.9),
            TradingSignal(SignalType.LLM, "short", SignalStrength.WEAK, 0.3),  # Low confidence
        ]

        result = aggregator.aggregate(signals, min_confidence=0.5)

        # Only high confidence signal should count
        assert result.signal_count == 1
        assert result.final_direction == "long"


class TestNewsSentimentAnalyzer:
    """Test NewsSentimentAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        from src.model.signal_aggregator import NewsSentimentAnalyzer
        return NewsSentimentAnalyzer()

    def test_analyze_bullish_headline(self, analyzer):
        headline = "Stock surges on strong earnings beat, rally continues"

        sentiment, strength = analyzer.analyze_headline(headline)

        assert sentiment > 0

    def test_analyze_bearish_headline(self, analyzer):
        headline = "Stock plunges after earnings miss, concerns about weak guidance"

        sentiment, strength = analyzer.analyze_headline(headline)

        assert sentiment < 0

    def test_analyze_neutral_headline(self, analyzer):
        headline = "Company announces new product launch date"

        sentiment, strength = analyzer.analyze_headline(headline)

        assert abs(sentiment) < 0.3

    def test_generate_signal(self, analyzer):
        headlines = [
            "Stock surges on strong earnings",
            "Analysts upgrade rating",
            "Company beats expectations"
        ]

        signal = analyzer.generate_signal(headlines, "TQQQ")

        assert signal.direction == "long"
        assert signal.confidence > 0


class TestTechnicalSignalGenerator:
    """Test TechnicalSignalGenerator class"""

    @pytest.fixture
    def generator(self):
        from src.model.signal_aggregator import TechnicalSignalGenerator
        return TechnicalSignalGenerator()

    def test_rsi_oversold(self, generator):
        signal = generator.generate_rsi_signal(rsi=25.0)

        assert signal.direction == "long"
        assert signal.confidence > 0

    def test_rsi_overbought(self, generator):
        signal = generator.generate_rsi_signal(rsi=75.0)

        assert signal.direction == "short"

    def test_rsi_neutral(self, generator):
        signal = generator.generate_rsi_signal(rsi=50.0)

        assert signal.direction == "hold"

    def test_macd_bullish(self, generator):
        signal = generator.generate_macd_signal(
            macd=0.5,
            signal_line=0.3,
            histogram=0.2,
            prev_histogram=-0.1  # Crossover
        )

        assert signal.direction == "long"

    def test_ma_crossover_bullish(self, generator):
        signal = generator.generate_ma_crossover_signal(
            fast_ma=51.0,
            slow_ma=50.0,
            price=52.0
        )

        assert signal.direction == "long"

    def test_bollinger_oversold(self, generator):
        signal = generator.generate_bollinger_signal(
            price=48.5,
            upper_band=52.0,
            lower_band=48.0,
            middle_band=50.0
        )

        assert signal.direction == "long"
