"""
AI-Trader Action Parser
Parses LLM outputs into executable trading actions

Based on HKUDS AI-Trader action parsing framework
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re
import json

from src.core.logger import get_logger
from src.action.futu_executor import TradingAction

logger = get_logger(__name__)


class ActionParseError(Exception):
    """Exception raised when action parsing fails"""
    pass


class ActionType(Enum):
    """Types of trading actions"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HOLD = "hold"
    INCREASE = "increase"   # Add to position
    DECREASE = "decrease"   # Reduce position
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class ParsedAction:
    """Parsed trading action from LLM output"""
    action_type: ActionType
    symbol: str
    confidence: float

    # Position sizing
    quantity: Optional[int] = None
    position_pct: Optional[float] = None  # Percentage of portfolio

    # Price targets
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Timing
    urgency: str = "normal"  # immediate, normal, patient
    valid_until: Optional[datetime] = None

    # Reasoning
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)

    # Metadata
    raw_output: str = ""
    parse_method: str = ""
    parse_time_ms: float = 0.0

    def to_trading_action(self) -> TradingAction:
        """Convert to TradingAction enum"""
        mapping = {
            ActionType.LONG: TradingAction.LONG,
            ActionType.SHORT: TradingAction.SHORT,
            ActionType.FLAT: TradingAction.FLAT,
            ActionType.HOLD: TradingAction.HOLD,
            ActionType.INCREASE: TradingAction.LONG,  # Same direction
            ActionType.DECREASE: TradingAction.FLAT,  # Reduce
            ActionType.STOP_LOSS: TradingAction.FLAT,
            ActionType.TAKE_PROFIT: TradingAction.FLAT,
        }
        return mapping.get(self.action_type, TradingAction.HOLD)

    def is_executable(self) -> bool:
        """Check if action can be executed"""
        return (
            self.action_type not in [ActionType.HOLD] and
            self.confidence >= 0.5 and
            self.symbol is not None
        )

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "confidence": self.confidence,
            "quantity": self.quantity,
            "position_pct": self.position_pct,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "urgency": self.urgency,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors
        }


class ActionParser:
    """
    Parses LLM outputs into structured trading actions.

    Supports multiple output formats:
    1. JSON format
    2. Natural language
    3. Structured text
    """

    # Action keywords for natural language parsing
    LONG_KEYWORDS = ["buy", "long", "bullish", "go long", "enter long", "increase"]
    SHORT_KEYWORDS = ["sell", "short", "bearish", "go short", "enter short"]
    FLAT_KEYWORDS = ["close", "flat", "exit", "liquidate", "close position"]
    HOLD_KEYWORDS = ["hold", "wait", "no action", "stay", "maintain"]

    def __init__(self):
        self._parse_cache: Dict[str, ParsedAction] = {}

    def parse(
        self,
        llm_output: str,
        symbol: str,
        default_confidence: float = 0.5
    ) -> ParsedAction:
        """
        Parse LLM output into a trading action.

        Args:
            llm_output: Raw LLM output string
            symbol: Trading symbol
            default_confidence: Default confidence if not specified

        Returns:
            ParsedAction
        """
        import time
        start = time.time()

        # Try JSON parsing first
        try:
            action = self._parse_json(llm_output, symbol)
            action.parse_method = "json"
            action.parse_time_ms = (time.time() - start) * 1000
            return action
        except (json.JSONDecodeError, ActionParseError):
            pass

        # Try structured text parsing
        try:
            action = self._parse_structured(llm_output, symbol)
            action.parse_method = "structured"
            action.parse_time_ms = (time.time() - start) * 1000
            return action
        except ActionParseError:
            pass

        # Fall back to natural language parsing
        action = self._parse_natural_language(llm_output, symbol, default_confidence)
        action.parse_method = "natural_language"
        action.parse_time_ms = (time.time() - start) * 1000

        return action

    def _parse_json(self, output: str, symbol: str) -> ParsedAction:
        """Parse JSON formatted output"""
        # Extract JSON from output
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if not json_match:
            raise ActionParseError("No JSON found in output")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            raise ActionParseError("Invalid JSON format")

        # Parse action type
        action_str = data.get("action", "hold").lower()
        action_type = self._string_to_action_type(action_str)

        # Parse confidence
        confidence = float(data.get("confidence", 0.5))

        # Parse position sizing
        quantity = data.get("quantity")
        position_pct = data.get("position_size_pct", data.get("position_pct"))
        if position_pct:
            position_pct = float(position_pct) / 100 if position_pct > 1 else float(position_pct)

        # Parse prices
        entry_price = data.get("entry_price")
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")

        # Parse reasoning
        reasoning = data.get("reasoning", "")
        key_factors = data.get("key_factors", [])

        return ParsedAction(
            action_type=action_type,
            symbol=symbol,
            confidence=confidence,
            quantity=quantity,
            position_pct=position_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            key_factors=key_factors,
            raw_output=output
        )

    def _parse_structured(self, output: str, symbol: str) -> ParsedAction:
        """Parse structured text output"""
        lines = output.strip().split('\n')

        data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().lower()] = value.strip()

        if 'action' not in data:
            raise ActionParseError("No action field found")

        action_type = self._string_to_action_type(data['action'])
        confidence = float(data.get('confidence', '0.5').replace('%', '')) / 100

        return ParsedAction(
            action_type=action_type,
            symbol=symbol,
            confidence=confidence,
            reasoning=data.get('reasoning', data.get('reason', '')),
            raw_output=output
        )

    def _parse_natural_language(
        self,
        output: str,
        symbol: str,
        default_confidence: float
    ) -> ParsedAction:
        """Parse natural language output"""
        output_lower = output.lower()

        # Determine action from keywords
        action_type = ActionType.HOLD

        for kw in self.LONG_KEYWORDS:
            if kw in output_lower:
                action_type = ActionType.LONG
                break

        for kw in self.SHORT_KEYWORDS:
            if kw in output_lower:
                action_type = ActionType.SHORT
                break

        for kw in self.FLAT_KEYWORDS:
            if kw in output_lower:
                action_type = ActionType.FLAT
                break

        # Extract confidence from text
        confidence = default_confidence
        confidence_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:confidence|confident)', output_lower)
        if confidence_match:
            conf_value = float(confidence_match.group(1))
            confidence = conf_value / 100 if conf_value > 1 else conf_value

        # Look for confidence indicators
        if any(word in output_lower for word in ["strongly", "definitely", "clearly", "very confident"]):
            confidence = max(confidence, 0.8)
        elif any(word in output_lower for word in ["slightly", "maybe", "possibly", "uncertain"]):
            confidence = min(confidence, 0.5)

        # Extract price targets
        prices = re.findall(r'\$(\d+(?:\.\d+)?)', output)
        stop_loss = None
        take_profit = None

        if "stop" in output_lower and prices:
            stop_loss = float(prices[0])
        if "target" in output_lower or "profit" in output_lower:
            if len(prices) >= 2:
                take_profit = float(prices[-1])

        return ParsedAction(
            action_type=action_type,
            symbol=symbol,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=output[:500],
            raw_output=output
        )

    def _string_to_action_type(self, action_str: str) -> ActionType:
        """Convert string to ActionType"""
        action_str = action_str.lower().strip()

        mapping = {
            "long": ActionType.LONG,
            "buy": ActionType.LONG,
            "short": ActionType.SHORT,
            "sell": ActionType.SHORT,
            "flat": ActionType.FLAT,
            "close": ActionType.FLAT,
            "exit": ActionType.FLAT,
            "hold": ActionType.HOLD,
            "wait": ActionType.HOLD,
            "increase": ActionType.INCREASE,
            "decrease": ActionType.DECREASE,
            "stop_loss": ActionType.STOP_LOSS,
            "take_profit": ActionType.TAKE_PROFIT,
        }

        return mapping.get(action_str, ActionType.HOLD)

    def validate_action(
        self,
        action: ParsedAction,
        current_position: Optional[Dict] = None,
        risk_limits: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a parsed action against rules and limits.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check confidence threshold
        if action.confidence < 0.5:
            issues.append(f"Low confidence: {action.confidence:.2%}")

        # Check position sizing
        if action.position_pct and action.position_pct > 0.25:
            issues.append(f"Position too large: {action.position_pct:.1%} > 25%")

        # Check stop loss
        if action.action_type in [ActionType.LONG, ActionType.SHORT]:
            if action.stop_loss is None:
                issues.append("No stop loss specified")

        # Validate against current position
        if current_position:
            if current_position.get("direction") == "long" and action.action_type == ActionType.LONG:
                if action.action_type != ActionType.INCREASE:
                    issues.append("Already in long position")

        # Check risk limits
        if risk_limits:
            max_position = risk_limits.get("max_position_pct", 0.25)
            if action.position_pct and action.position_pct > max_position:
                issues.append(f"Exceeds max position: {action.position_pct:.1%} > {max_position:.1%}")

        return len(issues) == 0, issues


class EnvironmentInterface:
    """
    AI-Trader Environment Interface.

    Provides a unified interface for the trading agent to interact with:
    1. Market data
    2. Execution system
    3. Risk management
    4. Portfolio state
    """

    def __init__(self):
        self._current_state: Dict[str, Any] = {}
        self._positions: Dict[str, Dict] = {}
        self._order_history: List[Dict] = []

    def get_market_state(self, symbol: str) -> Dict[str, Any]:
        """Get current market state for a symbol"""
        return self._current_state.get(symbol, {})

    def update_market_state(self, symbol: str, state: Dict[str, Any]):
        """Update market state"""
        self._current_state[symbol] = state

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol"""
        return self._positions.get(symbol)

    def update_position(self, symbol: str, position: Dict):
        """Update position"""
        self._positions[symbol] = position

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get overall portfolio state"""
        total_value = sum(p.get("market_value", 0) for p in self._positions.values())
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in self._positions.values())

        return {
            "positions": list(self._positions.keys()),
            "total_value": total_value,
            "total_pnl": total_pnl,
            "position_count": len(self._positions)
        }

    def execute_action(self, action: ParsedAction) -> Dict[str, Any]:
        """
        Execute a trading action.

        Returns execution result.
        """
        result = {
            "action": action.action_type.value,
            "symbol": action.symbol,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }

        # Record order
        self._order_history.append({
            **result,
            "confidence": action.confidence,
            "quantity": action.quantity
        })

        return result

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history"""
        return self._order_history[-limit:]

    def reset(self):
        """Reset environment state"""
        self._current_state = {}
        self._positions = {}
        self._order_history = []


# Singleton instances
_action_parser: Optional[ActionParser] = None
_environment: Optional[EnvironmentInterface] = None


def get_action_parser() -> ActionParser:
    """Get global action parser"""
    global _action_parser
    if _action_parser is None:
        _action_parser = ActionParser()
    return _action_parser


def get_environment() -> EnvironmentInterface:
    """Get global environment interface"""
    global _environment
    if _environment is None:
        _environment = EnvironmentInterface()
    return _environment
