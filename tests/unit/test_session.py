"""
Unit tests for session manager
"""
import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch, MagicMock


class TestMarketSession:
    """Test MarketSession enum"""

    def test_session_values(self):
        """Test session enum values"""
        from src.core.session_manager import MarketSession

        assert MarketSession.CLOSED.value == "closed"
        assert MarketSession.PRE_MARKET.value == "pre_market"
        assert MarketSession.REGULAR.value == "regular"
        assert MarketSession.AFTER_HOURS.value == "after_hours"


class TestSessionManager:
    """Test SessionManager class"""

    def test_session_manager_creation(self):
        """Test SessionManager creation"""
        from src.core.session_manager import SessionManager

        manager = SessionManager()
        assert manager.timezone is not None

    def test_get_session_info(self):
        """Test getting session info"""
        from src.core.session_manager import SessionManager, SessionInfo

        manager = SessionManager()
        info = manager.get_session_info()

        assert isinstance(info, SessionInfo)
        assert info.session is not None
        assert info.seconds_to_close >= 0

    def test_next_trading_day(self):
        """Test next trading day calculation"""
        from src.core.session_manager import SessionManager
        from datetime import date

        manager = SessionManager()

        # Monday should return Monday (if not holiday)
        monday = date(2024, 1, 8)  # A Monday
        next_day = manager._get_next_trading_day(monday)
        assert next_day.weekday() < 5  # Should be weekday

    def test_get_session_for_order(self):
        """Test order session parameter"""
        from src.core.session_manager import SessionManager

        manager = SessionManager()
        session_param = manager.get_session_for_order(extended_hours=True)

        assert session_param in ["NORMAL", "EXTENDED"]


class TestSessionInfo:
    """Test SessionInfo class"""

    def test_session_info_progress(self):
        """Test session progress calculation"""
        from src.core.session_manager import SessionInfo, MarketSession
        from datetime import datetime
        import pytz

        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)

        info = SessionInfo(
            session=MarketSession.REGULAR,
            session_start=now - timedelta(hours=2),
            session_end=now + timedelta(hours=4),
            next_session=MarketSession.AFTER_HOURS,
            next_session_start=now + timedelta(hours=4),
            seconds_to_close=14400,
            seconds_to_next_open=14400,
            is_trading_allowed=True
        )

        # Progress should be ~33% (2 out of 6 hours)
        assert 20 <= info.progress_pct <= 50
