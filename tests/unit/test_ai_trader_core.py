"""
Unit tests for AI-Trader Core
Tests Chain-of-Thought reasoning and market analysis
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock


class TestCoTStep:
    """Test CoTStep class"""

    def test_step_creation(self):
        """Test CoT step creation"""
        from src.model.ai_trader_core import CoTStep

        step = CoTStep(
            step_number=1,
            title="Market Analysis",
            analysis="Price is trending up",
            conclusion="Bullish bias",
            confidence=0.85,
            supporting_evidence=["RSI > 50", "MACD positive"]
        )

        assert step.step_number == 1
        assert step.confidence == 0.85
        assert len(step.supporting_evidence) == 2


class TestCoTReasoning:
    """Test CoTReasoning class"""

    def test_reasoning_to_text(self):
        """Test reasoning to text conversion"""
        from src.model.ai_trader_core import CoTReasoning, CoTStep

        steps = [
            CoTStep(
                step_number=1,
                title="Step 1",
                analysis="Analysis",
                conclusion="Conclusion",
                confidence=0.8
            )
        ]

        reasoning = CoTReasoning(
            steps=steps,
            final_decision="LONG",
            overall_confidence=0.8
        )

        text = reasoning.to_text()

        assert "Chain-of-Thought" in text
        assert "Step 1" in text
        assert "LONG" in text


class TestMarketRegime:
    """Test MarketRegime enum"""

    def test_regime_values(self):
        """Test regime enum values"""
        from src.model.ai_trader_core import MarketRegime

        assert MarketRegime.STRONG_BULLISH.value == "strong_bullish"
        assert MarketRegime.BEARISH.value == "bearish"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"


class TestSentimentAnalysis:
    """Test SentimentAnalysis"""

    def test_sentiment_is_extreme(self):
        """Test extreme sentiment detection"""
        from src.model.ai_trader_core import SentimentAnalysis, SentimentLevel

        extreme = SentimentAnalysis(
            level=SentimentLevel.EXTREME_GREED,
            score=0.9
        )

        neutral = SentimentAnalysis(
            level=SentimentLevel.NEUTRAL,
            score=0.0
        )

        assert extreme.is_extreme is True
        assert neutral.is_extreme is False


class TestUncertaintyEstimate:
    """Test UncertaintyEstimate"""

    def test_total_calculation(self):
        """Test total uncertainty calculation"""
        from src.model.ai_trader_core import UncertaintyEstimate

        estimate = UncertaintyEstimate(
            epistemic=0.3,
            aleatoric=0.4
        )

        # Total should be sqrt(0.3^2 + 0.4^2) = 0.5
        assert abs(estimate.total - 0.5) < 0.01


class TestAITraderCore:
    """Test AITraderCore class"""

    @pytest.fixture
    def mock_snapshot(self):
        """Create mock market snapshot"""
        snapshot = MagicMock()
        snapshot.symbol = "TQQQ"
        snapshot.futu_code = "US.TQQQ"
        snapshot.last_price = 50.0
        snapshot.change_day = 0.02
        snapshot.volume = 1000000
        snapshot.spread = 0.02
        snapshot.high = 51.0
        snapshot.low = 49.0

        # Mock indicators
        indicators = MagicMock()
        indicators.rsi_14 = 55.0
        indicators.macd_histogram = 0.05
        indicators.sma_5 = 50.5
        indicators.sma_20 = 49.5
        indicators.sma_50 = 48.0
        indicators.atr_14 = 1.0
        indicators.bollinger_upper = 52.0
        indicators.bollinger_lower = 48.0
        indicators.bollinger_position = 0.6
        indicators.adx_14 = 30
        indicators.volume_ratio = 1.2

        snapshot.indicators = indicators

        return snapshot

    @pytest.fixture
    def core(self):
        """Create AI-Trader core instance"""
        from src.model.ai_trader_core import AITraderCore
        return AITraderCore()

    def test_core_creation(self, core):
        """Test core creation"""
        assert core.CONFIDENCE_THRESHOLD == 0.6
        assert core.MAX_POSITION_PCT == 0.25

    def test_generate_cot_reasoning(self, core, mock_snapshot):
        """Test CoT reasoning generation"""
        reasoning = core.generate_cot_reasoning(mock_snapshot)

        assert len(reasoning.steps) >= 4
        assert reasoning.final_decision in ["LONG", "SHORT", "FLAT", "HOLD"]
        assert 0 <= reasoning.overall_confidence <= 1

    def test_detect_market_regime(self, core, mock_snapshot):
        """Test market regime detection"""
        from src.model.ai_trader_core import MarketRegime

        regime = core.detect_market_regime(mock_snapshot)

        assert isinstance(regime, MarketRegime)

    def test_detect_bullish_regime(self, core, mock_snapshot):
        """Test bullish regime detection"""
        from src.model.ai_trader_core import MarketRegime

        # Set bullish indicators
        mock_snapshot.indicators.rsi_14 = 65
        mock_snapshot.indicators.macd_histogram = 0.1

        regime = core.detect_market_regime(mock_snapshot)

        assert regime in [MarketRegime.BULLISH, MarketRegime.STRONG_BULLISH]

    def test_detect_bearish_regime(self, core, mock_snapshot):
        """Test bearish regime detection"""
        from src.model.ai_trader_core import MarketRegime

        # Set bearish indicators
        mock_snapshot.indicators.rsi_14 = 35
        mock_snapshot.indicators.macd_histogram = -0.1

        regime = core.detect_market_regime(mock_snapshot)

        assert regime in [MarketRegime.BEARISH, MarketRegime.STRONG_BEARISH, MarketRegime.NEUTRAL]

    def test_analyze_sentiment(self, core, mock_snapshot):
        """Test sentiment analysis"""
        from src.model.ai_trader_core import SentimentLevel

        sentiment = core.analyze_sentiment(mock_snapshot)

        assert isinstance(sentiment.level, SentimentLevel)
        assert -1 <= sentiment.score <= 1
        assert "rsi" in sentiment.factors

    def test_estimate_uncertainty(self, core, mock_snapshot):
        """Test uncertainty estimation"""
        uncertainty = core.estimate_uncertainty(mock_snapshot, 0.7)

        assert 0 <= uncertainty.epistemic <= 1
        assert 0 <= uncertainty.aleatoric <= 1
        assert uncertainty.total >= 0

    def test_self_reflect(self, core):
        """Test self-reflection"""
        recent_trades = [
            {"symbol": "TQQQ", "pnl": 100, "hour": 10},
            {"symbol": "TQQQ", "pnl": -50, "hour": 14},
            {"symbol": "QQQ", "pnl": 75, "hour": 11},
        ]

        metrics = {"win_rate": 0.66, "avg_return": 0.01}

        reflection = core.self_reflect(recent_trades, metrics)

        assert "win_rate" in reflection
        assert "adjustments" in reflection
        assert reflection["win_rate"] == pytest.approx(0.66, rel=0.1)


class TestMultiTimeframeAnalysis:
    """Test multi-timeframe analysis"""

    def test_analysis_structure(self):
        """Test analysis result structure"""
        from src.model.ai_trader_core import MultiTimeframeAnalysis

        analysis = MultiTimeframeAnalysis(
            timeframe="1m",
            trend="up",
            strength=0.7,
            key_levels={"support": 49.0, "resistance": 51.0},
            signals=["bullish_momentum"]
        )

        assert analysis.timeframe == "1m"
        assert analysis.trend == "up"
        assert "support" in analysis.key_levels


class TestGetAITraderCore:
    """Test singleton getter"""

    def test_singleton(self):
        """Test singleton instance"""
        from src.model.ai_trader_core import get_ai_trader_core

        core1 = get_ai_trader_core()
        core2 = get_ai_trader_core()

        assert core1 is core2
