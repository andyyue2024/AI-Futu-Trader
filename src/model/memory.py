"""
AI-Trader Memory Module
Implements trading memory for learning and pattern recognition

Based on HKUDS AI-Trader memory architecture:
- Short-term memory for recent trades
- Long-term memory for patterns
- Episodic memory for specific events
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import json
import hashlib

from src.core.logger import get_logger

logger = get_logger(__name__)


class MemoryType(Enum):
    """Types of memory"""
    SHORT_TERM = "short_term"      # Recent trades (last few hours)
    LONG_TERM = "long_term"        # Patterns over days/weeks
    EPISODIC = "episodic"          # Specific important events
    SEMANTIC = "semantic"          # General market knowledge


@dataclass
class MemoryEntry:
    """Single memory entry"""
    memory_id: str
    memory_type: MemoryType
    timestamp: datetime
    content: Dict[str, Any]
    importance: float = 0.5  # 0-1, higher is more important
    access_count: int = 0
    last_accessed: datetime = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

    def access(self):
        """Mark memory as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "importance": self.importance,
            "access_count": self.access_count,
            "tags": self.tags
        }


@dataclass
class TradeMemory:
    """Memory of a specific trade"""
    trade_id: str
    symbol: str
    action: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    pnl_pct: float

    # Context at time of trade
    market_regime: str
    sentiment: str
    confidence: float
    reasoning: str

    # Technical state
    rsi: float = 0.0
    macd: float = 0.0
    volume_ratio: float = 1.0

    # Outcome
    was_successful: bool = False
    lesson_learned: str = ""


@dataclass
class PatternMemory:
    """Memory of a recognized pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    occurrence_count: int = 0
    success_rate: float = 0.0
    avg_return: float = 0.0
    last_seen: datetime = None
    examples: List[str] = field(default_factory=list)


class TradingMemory:
    """
    AI-Trader Memory System.

    Implements hierarchical memory for:
    1. Short-term: Recent trades and market states
    2. Long-term: Learned patterns and strategies
    3. Episodic: Important events (circuit breakers, big wins/losses)
    """

    def __init__(
        self,
        short_term_size: int = 100,
        long_term_size: int = 1000,
        episodic_size: int = 50
    ):
        self.short_term_size = short_term_size
        self.long_term_size = long_term_size
        self.episodic_size = episodic_size

        # Memory stores
        self._short_term: deque = deque(maxlen=short_term_size)
        self._long_term: Dict[str, MemoryEntry] = {}
        self._episodic: deque = deque(maxlen=episodic_size)

        # Trade memories
        self._trade_memories: Dict[str, TradeMemory] = {}

        # Pattern memories
        self._patterns: Dict[str, PatternMemory] = {}

        # Index for fast retrieval
        self._symbol_index: Dict[str, List[str]] = {}
        self._tag_index: Dict[str, List[str]] = {}

    def remember_trade(self, trade: TradeMemory):
        """Store a trade in memory"""
        self._trade_memories[trade.trade_id] = trade

        # Create memory entry
        entry = MemoryEntry(
            memory_id=trade.trade_id,
            memory_type=MemoryType.SHORT_TERM,
            timestamp=trade.entry_time,
            content=self._trade_to_dict(trade),
            importance=self._calculate_trade_importance(trade),
            tags=[trade.symbol, trade.action, trade.market_regime]
        )

        self._short_term.append(entry)

        # Index by symbol
        if trade.symbol not in self._symbol_index:
            self._symbol_index[trade.symbol] = []
        self._symbol_index[trade.symbol].append(trade.trade_id)

        # Check if should be promoted to episodic
        if self._is_significant_trade(trade):
            self._add_to_episodic(entry)

        logger.debug(f"Remembered trade: {trade.trade_id}")

    def remember_pattern(self, pattern: PatternMemory):
        """Store a pattern in long-term memory"""
        self._patterns[pattern.pattern_id] = pattern

        entry = MemoryEntry(
            memory_id=pattern.pattern_id,
            memory_type=MemoryType.LONG_TERM,
            timestamp=datetime.now(),
            content={
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "success_rate": pattern.success_rate,
                "avg_return": pattern.avg_return
            },
            importance=pattern.success_rate,
            tags=["pattern", pattern.pattern_name]
        )

        self._long_term[pattern.pattern_id] = entry

    def remember_event(self, event_type: str, details: Dict[str, Any], importance: float = 0.8):
        """Store an important event in episodic memory"""
        event_id = self._generate_id(event_type + str(datetime.now()))

        entry = MemoryEntry(
            memory_id=event_id,
            memory_type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
            content={"event_type": event_type, **details},
            importance=importance,
            tags=[event_type]
        )

        self._episodic.append(entry)
        logger.info(f"Remembered event: {event_type}")

    def recall_recent_trades(self, symbol: str = None, count: int = 10) -> List[TradeMemory]:
        """Recall recent trades"""
        trades = list(self._trade_memories.values())

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        # Sort by time, most recent first
        trades.sort(key=lambda t: t.entry_time, reverse=True)

        return trades[:count]

    def recall_similar_situations(
        self,
        current_state: Dict[str, Any],
        top_k: int = 5
    ) -> List[TradeMemory]:
        """
        Recall past trades in similar market situations.
        Uses similarity matching on market state.
        """
        similar_trades = []

        current_rsi = current_state.get("rsi", 50)
        current_regime = current_state.get("regime", "neutral")
        current_sentiment = current_state.get("sentiment", "neutral")

        for trade in self._trade_memories.values():
            # Calculate similarity score
            rsi_sim = 1 - abs(trade.rsi - current_rsi) / 100
            regime_sim = 1.0 if trade.market_regime == current_regime else 0.3
            sentiment_sim = 1.0 if trade.sentiment == current_sentiment else 0.3

            similarity = (rsi_sim + regime_sim + sentiment_sim) / 3

            similar_trades.append((trade, similarity))

        # Sort by similarity
        similar_trades.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in similar_trades[:top_k]]

    def recall_patterns(self, symbol: str = None) -> List[PatternMemory]:
        """Recall learned patterns"""
        patterns = list(self._patterns.values())

        if symbol:
            patterns = [p for p in patterns if symbol in p.examples]

        # Sort by success rate
        patterns.sort(key=lambda p: p.success_rate, reverse=True)

        return patterns

    def recall_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Recall memories by tag"""
        memories = []

        for entry in self._short_term:
            if tag in entry.tags:
                entry.access()
                memories.append(entry)

        for entry in self._episodic:
            if tag in entry.tags:
                entry.access()
                memories.append(entry)

        for entry in self._long_term.values():
            if tag in entry.tags:
                entry.access()
                memories.append(entry)

        return memories

    def consolidate_memories(self):
        """
        Consolidate short-term memories into long-term patterns.
        Called periodically (e.g., end of day).
        """
        # Find patterns in recent trades
        recent_trades = list(self._trade_memories.values())[-50:]

        if len(recent_trades) < 10:
            return

        # Analyze win/loss patterns
        winners = [t for t in recent_trades if t.was_successful]
        losers = [t for t in recent_trades if not t.was_successful]

        # Pattern: Market regime success
        regime_stats = {}
        for trade in recent_trades:
            regime = trade.market_regime
            if regime not in regime_stats:
                regime_stats[regime] = {"wins": 0, "losses": 0}
            if trade.was_successful:
                regime_stats[regime]["wins"] += 1
            else:
                regime_stats[regime]["losses"] += 1

        # Create patterns from analysis
        for regime, stats in regime_stats.items():
            total = stats["wins"] + stats["losses"]
            if total >= 5:
                success_rate = stats["wins"] / total

                pattern = PatternMemory(
                    pattern_id=f"regime_{regime}",
                    pattern_name=f"{regime.capitalize()} Regime Pattern",
                    description=f"Trading performance in {regime} market regime",
                    occurrence_count=total,
                    success_rate=success_rate,
                    last_seen=datetime.now()
                )

                self.remember_pattern(pattern)

        logger.info(f"Consolidated {len(recent_trades)} trades into patterns")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "episodic_count": len(self._episodic),
            "trade_count": len(self._trade_memories),
            "pattern_count": len(self._patterns),
            "symbols_tracked": list(self._symbol_index.keys())
        }

    def forget_old_memories(self, days: int = 30):
        """Remove old, unimportant memories"""
        cutoff = datetime.now() - timedelta(days=days)

        # Keep important memories, remove old unaccessed ones
        to_remove = []
        for mem_id, entry in self._long_term.items():
            if (entry.timestamp < cutoff and
                entry.importance < 0.7 and
                entry.access_count < 3):
                to_remove.append(mem_id)

        for mem_id in to_remove:
            del self._long_term[mem_id]

        logger.info(f"Forgot {len(to_remove)} old memories")

    def _trade_to_dict(self, trade: TradeMemory) -> dict:
        """Convert trade memory to dict"""
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "action": trade.action,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "market_regime": trade.market_regime,
            "sentiment": trade.sentiment,
            "was_successful": trade.was_successful
        }

    def _calculate_trade_importance(self, trade: TradeMemory) -> float:
        """Calculate importance score for a trade"""
        importance = 0.5

        # Large P&L increases importance
        if abs(trade.pnl_pct) > 0.05:
            importance += 0.2

        # High confidence trades are important
        if trade.confidence > 0.8:
            importance += 0.1

        # Extreme market conditions
        if trade.market_regime in ["strong_bullish", "strong_bearish", "high_volatility"]:
            importance += 0.1

        return min(1.0, importance)

    def _is_significant_trade(self, trade: TradeMemory) -> bool:
        """Check if trade is significant enough for episodic memory"""
        return (
            abs(trade.pnl_pct) > 0.03 or  # >3% P&L
            trade.confidence > 0.9 or
            "circuit" in trade.lesson_learned.lower()
        )

    def _add_to_episodic(self, entry: MemoryEntry):
        """Add entry to episodic memory"""
        entry.memory_type = MemoryType.EPISODIC
        self._episodic.append(entry)

    def _generate_id(self, content: str) -> str:
        """Generate unique ID"""
        return hashlib.md5(content.encode()).hexdigest()[:12]


class ReflectionModule:
    """
    AI-Trader Reflection Module.

    Implements self-reflection for continuous improvement:
    1. Analyze past performance
    2. Identify mistakes and successes
    3. Generate improvement suggestions
    4. Update trading rules
    """

    def __init__(self, memory: TradingMemory):
        self.memory = memory
        self._reflections: List[Dict] = []
        self._improvement_history: List[Dict] = []

    def reflect_on_day(self, date: datetime = None) -> Dict[str, Any]:
        """
        Reflect on a day's trading performance.
        """
        date = date or datetime.now()

        # Get day's trades
        trades = self.memory.recall_recent_trades(count=50)
        day_trades = [t for t in trades if t.entry_time.date() == date.date()]

        if not day_trades:
            return {"status": "no_trades", "date": date.isoformat()}

        # Calculate metrics
        total_pnl = sum(t.pnl for t in day_trades)
        winners = [t for t in day_trades if t.was_successful]
        win_rate = len(winners) / len(day_trades)

        # Analyze patterns
        analysis = {
            "date": date.isoformat(),
            "total_trades": len(day_trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "winners": len(winners),
            "losers": len(day_trades) - len(winners),
            "insights": [],
            "improvements": []
        }

        # Find insights
        if win_rate < 0.4:
            analysis["insights"].append("Low win rate - consider stricter entry criteria")
            analysis["improvements"].append({
                "area": "entry_criteria",
                "suggestion": "Increase confidence threshold from 0.6 to 0.7"
            })

        if win_rate > 0.7 and total_pnl < 0:
            analysis["insights"].append("High win rate but negative P&L - winners too small")
            analysis["improvements"].append({
                "area": "take_profit",
                "suggestion": "Extend take-profit targets"
            })

        # Analyze by regime
        regime_performance = {}
        for trade in day_trades:
            regime = trade.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = {"pnl": 0, "count": 0}
            regime_performance[regime]["pnl"] += trade.pnl
            regime_performance[regime]["count"] += 1

        for regime, perf in regime_performance.items():
            if perf["count"] >= 3 and perf["pnl"] < 0:
                analysis["insights"].append(f"Poor performance in {regime} regime")
                analysis["improvements"].append({
                    "area": "regime_filter",
                    "suggestion": f"Consider avoiding trades in {regime} regime"
                })

        # Store reflection
        self._reflections.append(analysis)

        # Remember as event
        self.memory.remember_event(
            "daily_reflection",
            analysis,
            importance=0.7
        )

        return analysis

    def reflect_on_trade(self, trade: TradeMemory) -> Dict[str, Any]:
        """
        Reflect on a specific trade.
        """
        reflection = {
            "trade_id": trade.trade_id,
            "outcome": "success" if trade.was_successful else "failure",
            "pnl": trade.pnl,
            "analysis": [],
            "lessons": []
        }

        # Analyze the trade
        if trade.was_successful:
            if trade.confidence > 0.8:
                reflection["analysis"].append("High confidence trade paid off")
            if trade.pnl_pct > 0.02:
                reflection["lessons"].append("Let winners run in similar setups")
        else:
            if trade.confidence < 0.6:
                reflection["lessons"].append("Avoid low confidence trades")
            if trade.pnl_pct < -0.02:
                reflection["lessons"].append("Tighten stop-loss for this setup")

        # Compare to similar past trades
        similar = self.memory.recall_similar_situations({
            "rsi": trade.rsi,
            "regime": trade.market_regime,
            "sentiment": trade.sentiment
        }, top_k=5)

        if similar:
            similar_outcomes = [t.was_successful for t in similar]
            historical_win_rate = sum(similar_outcomes) / len(similar_outcomes)

            reflection["historical_context"] = {
                "similar_trades": len(similar),
                "historical_win_rate": historical_win_rate
            }

            if historical_win_rate < 0.3 and trade.was_successful:
                reflection["lessons"].append("This was an unusual win - don't rely on it")
            elif historical_win_rate > 0.7 and not trade.was_successful:
                reflection["lessons"].append("This was an unusual loss - setup still valid")

        return reflection

    def generate_improvement_plan(self) -> Dict[str, Any]:
        """
        Generate improvement plan based on recent reflections.
        """
        if len(self._reflections) < 5:
            return {"status": "insufficient_data"}

        recent = self._reflections[-10:]

        # Aggregate insights
        all_insights = []
        all_improvements = []

        for r in recent:
            all_insights.extend(r.get("insights", []))
            all_improvements.extend(r.get("improvements", []))

        # Count recurring issues
        from collections import Counter
        insight_counts = Counter(all_insights)

        plan = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period_days": len(recent),
            "recurring_issues": dict(insight_counts.most_common(5)),
            "priority_improvements": [],
            "suggested_parameter_changes": {}
        }

        # Prioritize improvements
        improvement_areas = Counter(i["area"] for i in all_improvements)

        for area, count in improvement_areas.most_common(3):
            suggestions = [i["suggestion"] for i in all_improvements if i["area"] == area]
            plan["priority_improvements"].append({
                "area": area,
                "frequency": count,
                "suggestions": list(set(suggestions))
            })

        # Suggest parameter changes
        if insight_counts.get("Low win rate", 0) >= 3:
            plan["suggested_parameter_changes"]["confidence_threshold"] = 0.7

        self._improvement_history.append(plan)

        return plan

    def get_lessons_learned(self, limit: int = 10) -> List[str]:
        """Get most important lessons learned"""
        all_lessons = []

        for reflection in self._reflections[-20:]:
            if "lessons" in reflection:
                all_lessons.extend(reflection["lessons"])

        # Deduplicate and count
        from collections import Counter
        lesson_counts = Counter(all_lessons)

        return [lesson for lesson, _ in lesson_counts.most_common(limit)]


# Singleton instances
_trading_memory: Optional[TradingMemory] = None
_reflection_module: Optional[ReflectionModule] = None


def get_trading_memory() -> TradingMemory:
    """Get global trading memory"""
    global _trading_memory
    if _trading_memory is None:
        _trading_memory = TradingMemory()
    return _trading_memory


def get_reflection_module() -> ReflectionModule:
    """Get global reflection module"""
    global _reflection_module
    if _reflection_module is None:
        _reflection_module = ReflectionModule(get_trading_memory())
    return _reflection_module
