"""
Unit tests for slippage controller
"""
import pytest
from datetime import datetime, timedelta


class TestSlippageRecord:
    """Test SlippageRecord class"""

    def test_record_creation(self):
        """Test record creation"""
        from src.risk.slippage_controller import SlippageRecord

        record = SlippageRecord(
            timestamp=datetime.now(),
            symbol="TQQQ",
            side="buy",
            expected_price=50.0,
            actual_price=50.05,
            quantity=100,
            slippage_pct=0.001,
            slippage_usd=5.0
        )

        assert record.symbol == "TQQQ"
        assert record.slippage_pct == 0.001

    def test_is_favorable_buy(self):
        """Test favorable slippage detection for buy"""
        from src.risk.slippage_controller import SlippageRecord

        # Favorable: bought lower than expected
        record = SlippageRecord(
            timestamp=datetime.now(),
            symbol="TQQQ",
            side="buy",
            expected_price=50.0,
            actual_price=49.95,
            quantity=100,
            slippage_pct=-0.001,
            slippage_usd=5.0
        )

        assert record.is_favorable is True

    def test_is_favorable_sell(self):
        """Test favorable slippage detection for sell"""
        from src.risk.slippage_controller import SlippageRecord

        # Favorable: sold higher than expected
        record = SlippageRecord(
            timestamp=datetime.now(),
            symbol="TQQQ",
            side="sell",
            expected_price=50.0,
            actual_price=50.05,
            quantity=100,
            slippage_pct=-0.001,
            slippage_usd=5.0
        )

        assert record.is_favorable is True


class TestSlippageController:
    """Test SlippageController class"""

    def test_controller_creation(self):
        """Test controller creation"""
        from src.risk.slippage_controller import SlippageController

        controller = SlippageController()
        assert controller.TARGET_SLIPPAGE_PCT == 0.002

    def test_record_slippage(self):
        """Test recording slippage"""
        from src.risk.slippage_controller import SlippageController

        controller = SlippageController()

        record = controller.record_slippage(
            symbol="TQQQ",
            side="buy",
            expected_price=50.0,
            actual_price=50.05,
            quantity=100
        )

        assert record.symbol == "TQQQ"
        assert record.slippage_pct == pytest.approx(0.001, rel=0.01)

    def test_get_stats(self):
        """Test getting statistics"""
        from src.risk.slippage_controller import SlippageController

        controller = SlippageController()

        # Record multiple slippages
        for i in range(10):
            controller.record_slippage(
                symbol="TQQQ",
                side="buy",
                expected_price=50.0,
                actual_price=50.0 + (i * 0.01),
                quantity=100
            )

        stats = controller.get_stats(24)

        assert stats.total_orders == 10
        assert stats.avg_slippage_pct >= 0

    def test_is_meeting_target(self):
        """Test target compliance check"""
        from src.risk.slippage_controller import SlippageController

        controller = SlippageController()

        # Record slippages within target
        for _ in range(100):
            controller.record_slippage(
                symbol="TQQQ",
                side="buy",
                expected_price=50.0,
                actual_price=50.05,  # 0.1% slippage
                quantity=100
            )

        assert controller.is_meeting_target(24) is True


class TestFillRateMonitor:
    """Test FillRateMonitor class"""

    def test_monitor_creation(self):
        """Test monitor creation"""
        from src.risk.slippage_controller import FillRateMonitor

        monitor = FillRateMonitor()
        assert monitor.TARGET_FILL_RATE == 0.95

    def test_record_order(self):
        """Test recording orders"""
        from src.risk.slippage_controller import FillRateMonitor

        monitor = FillRateMonitor()

        monitor.record_order(
            order_id="order-001",
            symbol="TQQQ",
            requested_qty=100,
            filled_qty=100,
            status="FILLED"
        )

        fill_rate = monitor.get_fill_rate(24)
        assert fill_rate == 1.0

    def test_partial_fill_stats(self):
        """Test partial fill statistics"""
        from src.risk.slippage_controller import FillRateMonitor

        monitor = FillRateMonitor()

        # Fully filled
        monitor.record_order("o1", "TQQQ", 100, 100, "FILLED")
        monitor.record_order("o2", "TQQQ", 100, 100, "FILLED")

        # Partial
        monitor.record_order("o3", "TQQQ", 100, 50, "PARTIAL")

        # Unfilled
        monitor.record_order("o4", "TQQQ", 100, 0, "CANCELLED")

        stats = monitor.get_partial_fill_stats(24)

        assert stats["total"] == 4
        assert stats["fully_filled"] == 2
        assert stats["partial"] == 1
        assert stats["unfilled"] == 1


class TestVolumeTracker:
    """Test VolumeTracker class"""

    def test_tracker_creation(self):
        """Test tracker creation"""
        from src.risk.slippage_controller import VolumeTracker

        tracker = VolumeTracker()
        assert tracker.TARGET_DAILY_VOLUME_USD == 50000.0

    def test_record_trade(self):
        """Test recording trades"""
        from src.risk.slippage_controller import VolumeTracker

        tracker = VolumeTracker()

        tracker.record_trade("TQQQ", 100, 50.0)
        tracker.record_trade("TQQQ", 200, 50.0)

        volume = tracker.get_today_volume()
        assert volume == 15000.0

    def test_is_meeting_target(self):
        """Test target compliance"""
        from src.risk.slippage_controller import VolumeTracker

        tracker = VolumeTracker()

        # Below target
        tracker.record_trade("TQQQ", 100, 50.0)
        assert tracker.is_meeting_target() is False

        # At target
        tracker.record_trade("TQQQ", 1000, 50.0)  # +50,000
        assert tracker.is_meeting_target() is True

    def test_get_progress(self):
        """Test progress calculation"""
        from src.risk.slippage_controller import VolumeTracker

        tracker = VolumeTracker()

        tracker.record_trade("TQQQ", 500, 50.0)  # 25,000

        progress = tracker.get_progress()
        assert progress == pytest.approx(50.0, rel=0.01)
