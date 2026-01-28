"""
Unit tests for Memory Module
"""
import pytest
from datetime import datetime, timedelta


class TestMemoryType:
    """Test MemoryType enum"""

    def test_memory_types(self):
        from src.model.memory import MemoryType

        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.EPISODIC.value == "episodic"


class TestMemoryEntry:
    """Test MemoryEntry class"""

    def test_entry_creation(self):
        from src.model.memory import MemoryEntry, MemoryType

        entry = MemoryEntry(
            memory_id="test-1",
            memory_type=MemoryType.SHORT_TERM,
            timestamp=datetime.now(),
            content={"key": "value"},
            importance=0.7
        )

        assert entry.memory_id == "test-1"
        assert entry.importance == 0.7

    def test_access_tracking(self):
        from src.model.memory import MemoryEntry, MemoryType

        entry = MemoryEntry(
            memory_id="test-1",
            memory_type=MemoryType.SHORT_TERM,
            timestamp=datetime.now(),
            content={}
        )

        assert entry.access_count == 0
        entry.access()
        assert entry.access_count == 1


class TestTradeMemory:
    """Test TradeMemory class"""

    def test_trade_memory_creation(self):
        from src.model.memory import TradeMemory

        trade = TradeMemory(
            trade_id="trade-1",
            symbol="TQQQ",
            action="long",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=52.0,
            pnl=200.0,
            pnl_pct=0.04,
            market_regime="bullish",
            sentiment="greed",
            confidence=0.8,
            reasoning="Strong momentum"
        )

        assert trade.symbol == "TQQQ"
        assert trade.pnl == 200.0


class TestTradingMemory:
    """Test TradingMemory class"""

    @pytest.fixture
    def memory(self):
        from src.model.memory import TradingMemory
        return TradingMemory()

    @pytest.fixture
    def sample_trade(self):
        from src.model.memory import TradeMemory

        return TradeMemory(
            trade_id="trade-1",
            symbol="TQQQ",
            action="long",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=52.0,
            pnl=200.0,
            pnl_pct=0.04,
            market_regime="bullish",
            sentiment="greed",
            confidence=0.8,
            reasoning="Strong momentum",
            was_successful=True
        )

    def test_remember_trade(self, memory, sample_trade):
        memory.remember_trade(sample_trade)

        assert len(memory._trade_memories) == 1
        assert len(memory._short_term) == 1

    def test_recall_recent_trades(self, memory, sample_trade):
        memory.remember_trade(sample_trade)

        trades = memory.recall_recent_trades(count=10)

        assert len(trades) == 1
        assert trades[0].trade_id == "trade-1"

    def test_recall_by_symbol(self, memory, sample_trade):
        from src.model.memory import TradeMemory

        memory.remember_trade(sample_trade)

        other_trade = TradeMemory(
            trade_id="trade-2",
            symbol="QQQ",
            action="short",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=400.0,
            exit_price=398.0,
            pnl=200.0,
            pnl_pct=0.005,
            market_regime="bearish",
            sentiment="fear",
            confidence=0.7,
            reasoning="Weak momentum"
        )
        memory.remember_trade(other_trade)

        tqqq_trades = memory.recall_recent_trades(symbol="TQQQ")

        assert len(tqqq_trades) == 1
        assert tqqq_trades[0].symbol == "TQQQ"

    def test_remember_event(self, memory):
        memory.remember_event(
            "circuit_breaker",
            {"reason": "3% daily loss"},
            importance=0.9
        )

        assert len(memory._episodic) == 1

    def test_get_memory_stats(self, memory, sample_trade):
        memory.remember_trade(sample_trade)

        stats = memory.get_memory_stats()

        assert stats["trade_count"] == 1
        assert "TQQQ" in stats["symbols_tracked"]


class TestReflectionModule:
    """Test ReflectionModule class"""

    @pytest.fixture
    def reflection(self):
        from src.model.memory import TradingMemory, ReflectionModule
        memory = TradingMemory()
        return ReflectionModule(memory)

    def test_reflect_on_trade(self, reflection):
        from src.model.memory import TradeMemory

        trade = TradeMemory(
            trade_id="trade-1",
            symbol="TQQQ",
            action="long",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=52.0,
            pnl=200.0,
            pnl_pct=0.04,
            market_regime="bullish",
            sentiment="greed",
            confidence=0.9,
            reasoning="Strong momentum",
            was_successful=True
        )

        result = reflection.reflect_on_trade(trade)

        assert result["outcome"] == "success"
        assert "analysis" in result

    def test_get_lessons_learned(self, reflection):
        lessons = reflection.get_lessons_learned()

        assert isinstance(lessons, list)
