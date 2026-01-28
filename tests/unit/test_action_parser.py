"""
Unit tests for Action Parser
"""
import pytest
from datetime import datetime


class TestActionType:
    """Test ActionType enum"""

    def test_action_types(self):
        from src.model.action_parser import ActionType

        assert ActionType.LONG.value == "long"
        assert ActionType.SHORT.value == "short"
        assert ActionType.FLAT.value == "flat"
        assert ActionType.HOLD.value == "hold"


class TestParsedAction:
    """Test ParsedAction class"""

    def test_action_creation(self):
        from src.model.action_parser import ParsedAction, ActionType

        action = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.8,
            quantity=100,
            reasoning="Bullish signal"
        )

        assert action.symbol == "TQQQ"
        assert action.confidence == 0.8

    def test_to_trading_action(self):
        from src.model.action_parser import ParsedAction, ActionType
        from src.action.futu_executor import TradingAction

        action = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.8
        )

        trading_action = action.to_trading_action()

        assert trading_action == TradingAction.LONG

    def test_is_executable(self):
        from src.model.action_parser import ParsedAction, ActionType

        executable = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.8
        )

        not_executable = ParsedAction(
            action_type=ActionType.HOLD,
            symbol="TQQQ",
            confidence=0.8
        )

        low_confidence = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.3
        )

        assert executable.is_executable() is True
        assert not_executable.is_executable() is False
        assert low_confidence.is_executable() is False


class TestActionParser:
    """Test ActionParser class"""

    @pytest.fixture
    def parser(self):
        from src.model.action_parser import ActionParser
        return ActionParser()

    def test_parse_json(self, parser):
        llm_output = '''
        Based on my analysis:
        {"action": "long", "confidence": 0.85, "reasoning": "Strong momentum"}
        '''

        action = parser.parse(llm_output, "TQQQ")

        assert action.action_type.value == "long"
        assert action.confidence == 0.85
        assert action.parse_method == "json"

    def test_parse_json_with_targets(self, parser):
        llm_output = '''
        {
            "action": "long",
            "confidence": 0.8,
            "entry_price": 50.0,
            "stop_loss": 48.0,
            "take_profit": 55.0,
            "position_size_pct": 10
        }
        '''

        action = parser.parse(llm_output, "TQQQ")

        assert action.entry_price == 50.0
        assert action.stop_loss == 48.0
        assert action.take_profit == 55.0
        assert action.position_pct == 0.10

    def test_parse_natural_language_long(self, parser):
        llm_output = "I strongly recommend going long on this stock as momentum is bullish"

        action = parser.parse(llm_output, "TQQQ")

        assert action.action_type.value == "long"
        assert action.parse_method == "natural_language"

    def test_parse_natural_language_short(self, parser):
        llm_output = "Time to short this position, bearish signals everywhere"

        action = parser.parse(llm_output, "TQQQ")

        assert action.action_type.value == "short"

    def test_parse_natural_language_hold(self, parser):
        llm_output = "We should hold and wait for better opportunities"

        action = parser.parse(llm_output, "TQQQ")

        assert action.action_type.value == "hold"

    def test_parse_confidence_extraction(self, parser):
        llm_output = "I am 75% confident we should buy this stock"

        action = parser.parse(llm_output, "TQQQ")

        assert action.confidence == pytest.approx(0.75, rel=0.1)

    def test_parse_strong_confidence(self, parser):
        llm_output = "I am very confident and strongly recommend buying"

        action = parser.parse(llm_output, "TQQQ")

        assert action.confidence >= 0.7

    def test_validate_action_valid(self, parser):
        from src.model.action_parser import ParsedAction, ActionType

        action = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.8,
            position_pct=0.10,
            stop_loss=48.0
        )

        is_valid, issues = parser.validate_action(action)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_action_low_confidence(self, parser):
        from src.model.action_parser import ParsedAction, ActionType

        action = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.3,
            stop_loss=48.0
        )

        is_valid, issues = parser.validate_action(action)

        assert is_valid is False
        assert any("confidence" in i.lower() for i in issues)

    def test_validate_action_no_stop_loss(self, parser):
        from src.model.action_parser import ParsedAction, ActionType

        action = ParsedAction(
            action_type=ActionType.LONG,
            symbol="TQQQ",
            confidence=0.8
        )

        is_valid, issues = parser.validate_action(action)

        assert "stop loss" in " ".join(issues).lower()


class TestEnvironmentInterface:
    """Test EnvironmentInterface class"""

    @pytest.fixture
    def env(self):
        from src.model.action_parser import EnvironmentInterface
        return EnvironmentInterface()

    def test_update_market_state(self, env):
        env.update_market_state("TQQQ", {"price": 50.0, "volume": 1000000})

        state = env.get_market_state("TQQQ")

        assert state["price"] == 50.0

    def test_update_position(self, env):
        env.update_position("TQQQ", {
            "quantity": 100,
            "market_value": 5000.0,
            "unrealized_pnl": 50.0
        })

        position = env.get_position("TQQQ")

        assert position["quantity"] == 100

    def test_get_portfolio_state(self, env):
        env.update_position("TQQQ", {
            "market_value": 5000.0,
            "unrealized_pnl": 50.0
        })
        env.update_position("QQQ", {
            "market_value": 10000.0,
            "unrealized_pnl": -25.0
        })

        state = env.get_portfolio_state()

        assert state["total_value"] == 15000.0
        assert state["total_pnl"] == 25.0
        assert state["position_count"] == 2

    def test_reset(self, env):
        env.update_position("TQQQ", {"quantity": 100})
        env.reset()

        assert env.get_position("TQQQ") is None
