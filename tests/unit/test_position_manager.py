"""
Unit tests for position manager
"""
import pytest
from datetime import datetime, timedelta


class TestManagedPosition:
    """Test ManagedPosition class"""

    def test_position_creation(self):
        """Test position creation"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        assert position.symbol == "TQQQ"
        assert position.is_flat is True
        assert position.quantity == 0

    def test_add_long_entry(self):
        """Test adding a long entry"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(
            quantity=100,
            price=50.0,
            order_id="test-001"
        )

        assert position.is_long is True
        assert position.quantity == 100
        assert position.avg_cost == 50.0

    def test_add_short_entry(self):
        """Test adding a short entry"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(
            quantity=-100,  # Negative for short
            price=50.0,
            order_id="test-001"
        )

        assert position.is_short is True
        assert position.quantity == -100

    def test_reduce_long_position(self):
        """Test reducing a long position"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        pnl = position.reduce(50, 52.0, "exit-001")

        # PnL = (52 - 50) * 50 = 100
        assert pnl == 100.0
        assert position.quantity == 50
        assert position.realized_pnl == 100.0

    def test_close_full_position(self):
        """Test closing a full position"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        pnl = position.reduce(100, 55.0, "exit-001")

        # PnL = (55 - 50) * 100 = 500
        assert pnl == 500.0
        assert position.is_flat is True
        assert position.realized_pnl == 500.0

    def test_update_price(self):
        """Test price update and unrealized P&L"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        position.update_price(52.0)

        # Unrealized = (52 - 50) * 100 = 200
        assert position.unrealized_pnl == 200.0

    def test_stop_loss_trigger(self):
        """Test stop loss trigger detection"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        position.set_stop_loss(pct=0.02)  # 2% stop loss

        # Stop loss at 49.0
        assert position.stop_loss_price == pytest.approx(49.0, rel=0.01)

        # Not triggered at 49.5
        assert position.check_stop_loss(49.5) is False

        # Triggered at 48.5
        assert position.check_stop_loss(48.5) is True

    def test_take_profit_trigger(self):
        """Test take profit trigger detection"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        position.set_take_profit(pct=0.04)  # 4% take profit

        # Take profit at 52.0
        assert position.take_profit_price == pytest.approx(52.0, rel=0.01)

        # Not triggered at 51.5
        assert position.check_take_profit(51.5) is False

        # Triggered at 52.5
        assert position.check_take_profit(52.5) is True

    def test_trailing_stop(self):
        """Test trailing stop functionality"""
        from src.action.position_manager import ManagedPosition

        position = ManagedPosition(
            symbol="TQQQ",
            futu_code="US.TQQQ"
        )

        position.add_entry(100, 50.0, "entry-001")
        position.set_trailing_stop(0.02, current_price=50.0)  # 2% trailing

        # Initial stop at 49.0
        assert position.stop_loss_price == pytest.approx(49.0, rel=0.01)

        # Price moves up, stop should move up
        position.update_price(52.0)
        assert position.trailing_stop_high == 52.0
        assert position.stop_loss_price == pytest.approx(50.96, rel=0.01)


class TestPositionManager:
    """Test PositionManager class"""

    def test_manager_creation(self):
        """Test manager creation"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        assert manager.cash == 100000.0
        assert manager.position_count == 0

    def test_open_position(self):
        """Test opening a position"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        position = manager.open_position(
            futu_code="US.TQQQ",
            quantity=100,
            price=50.0,
            order_id="test-001"
        )

        assert position.quantity == 100
        assert manager.cash == 95000.0  # 100000 - 5000
        assert manager.position_count == 1

    def test_close_position(self):
        """Test closing a position"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        manager.open_position("US.TQQQ", 100, 50.0, "entry-001")
        pnl = manager.close_position("US.TQQQ", 100, 52.0, "exit-001")

        assert pnl == 200.0
        assert manager.cash == 100200.0  # Started with 95000, got 5200 back
        assert manager.position_count == 0

    def test_portfolio_value(self):
        """Test portfolio value calculation"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        manager.open_position("US.TQQQ", 100, 50.0, "entry-001")

        # Cash: 95000, Position: 5000
        assert manager.portfolio_value == 100000.0

    def test_update_prices(self):
        """Test price updates"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        manager.open_position("US.TQQQ", 100, 50.0, "entry-001")
        manager.update_prices({"US.TQQQ": 52.0})

        assert manager.total_unrealized_pnl == 200.0

    def test_check_risk_levels(self):
        """Test risk level checking"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)

        position = manager.open_position(
            "US.TQQQ", 100, 50.0, "entry-001",
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

        # Check at various prices
        triggers = manager.check_risk_levels({"US.TQQQ": 51.0})
        assert len(triggers) == 0  # No triggers

        triggers = manager.check_risk_levels({"US.TQQQ": 48.5})
        assert len(triggers) == 1  # Stop loss triggered
        assert triggers[0][1] == "stop_loss"

    def test_get_summary(self):
        """Test summary generation"""
        from src.action.position_manager import PositionManager

        manager = PositionManager(starting_cash=100000.0)
        manager.open_position("US.TQQQ", 100, 50.0, "entry-001")

        summary = manager.get_summary()

        assert "cash" in summary
        assert "portfolio_value" in summary
        assert "positions" in summary
        assert len(summary["positions"]) == 1
