"""
Trading Session Manager - Manages pre-market, regular, and after-hours trading
Provides seamless session transitions for 24-hour trading capability
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional, Tuple
import pytz

from src.core.logger import get_logger

logger = get_logger(__name__)


class MarketSession(Enum):
    """Market trading sessions"""
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"


@dataclass
class SessionInfo:
    """Information about current trading session"""
    session: MarketSession
    session_start: datetime
    session_end: datetime
    next_session: MarketSession
    next_session_start: datetime
    seconds_to_close: int
    seconds_to_next_open: int
    is_trading_allowed: bool

    @property
    def progress_pct(self) -> float:
        """Session progress percentage"""
        if self.session == MarketSession.CLOSED:
            return 0.0
        total_seconds = (self.session_end - self.session_start).total_seconds()
        elapsed = (datetime.now(pytz.timezone('US/Eastern')) - self.session_start).total_seconds()
        return min(100.0, max(0.0, (elapsed / total_seconds) * 100))


class SessionManager:
    """
    Manages trading sessions for US markets.
    Handles pre-market, regular hours, and after-hours seamlessly.

    Session Times (Eastern Time):
    - Pre-market: 04:00 - 09:30
    - Regular: 09:30 - 16:00
    - After-hours: 16:00 - 20:00
    - Closed: 20:00 - 04:00 (next day)
    """

    # Session times in Eastern Time
    PRE_MARKET_START = time(4, 0)
    REGULAR_START = time(9, 30)
    REGULAR_END = time(16, 0)
    AFTER_HOURS_END = time(20, 0)

    # Holidays (add as needed)
    HOLIDAYS_2024 = [
        datetime(2024, 1, 1),   # New Year's Day
        datetime(2024, 1, 15),  # MLK Day
        datetime(2024, 2, 19),  # Presidents Day
        datetime(2024, 3, 29),  # Good Friday
        datetime(2024, 5, 27),  # Memorial Day
        datetime(2024, 6, 19),  # Juneteenth
        datetime(2024, 7, 4),   # Independence Day
        datetime(2024, 9, 2),   # Labor Day
        datetime(2024, 11, 28), # Thanksgiving
        datetime(2024, 12, 25), # Christmas
    ]

    def __init__(self):
        self.timezone = pytz.timezone('US/Eastern')
        self._last_session: Optional[MarketSession] = None

    def get_current_session(self) -> MarketSession:
        """Get the current trading session"""
        now = datetime.now(self.timezone)
        current_time = now.time()
        weekday = now.weekday()

        # Weekend check
        if weekday >= 5:  # Saturday=5, Sunday=6
            return MarketSession.CLOSED

        # Holiday check
        if self._is_holiday(now.date()):
            return MarketSession.CLOSED

        # Time-based session
        if current_time < self.PRE_MARKET_START:
            return MarketSession.CLOSED
        elif current_time < self.REGULAR_START:
            return MarketSession.PRE_MARKET
        elif current_time < self.REGULAR_END:
            return MarketSession.REGULAR
        elif current_time < self.AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    def get_session_info(self) -> SessionInfo:
        """Get detailed information about current session"""
        now = datetime.now(self.timezone)
        current_session = self.get_current_session()

        session_start, session_end = self._get_session_times(now, current_session)
        next_session, next_start = self._get_next_session(now, current_session)

        seconds_to_close = int((session_end - now).total_seconds()) if session_end > now else 0
        seconds_to_next = int((next_start - now).total_seconds()) if next_start > now else 0

        return SessionInfo(
            session=current_session,
            session_start=session_start,
            session_end=session_end,
            next_session=next_session,
            next_session_start=next_start,
            seconds_to_close=max(0, seconds_to_close),
            seconds_to_next_open=max(0, seconds_to_next),
            is_trading_allowed=current_session != MarketSession.CLOSED
        )

    def _get_session_times(
        self,
        now: datetime,
        session: MarketSession
    ) -> Tuple[datetime, datetime]:
        """Get start and end times for a session"""
        today = now.date()

        if session == MarketSession.PRE_MARKET:
            start = self.timezone.localize(datetime.combine(today, self.PRE_MARKET_START))
            end = self.timezone.localize(datetime.combine(today, self.REGULAR_START))
        elif session == MarketSession.REGULAR:
            start = self.timezone.localize(datetime.combine(today, self.REGULAR_START))
            end = self.timezone.localize(datetime.combine(today, self.REGULAR_END))
        elif session == MarketSession.AFTER_HOURS:
            start = self.timezone.localize(datetime.combine(today, self.REGULAR_END))
            end = self.timezone.localize(datetime.combine(today, self.AFTER_HOURS_END))
        else:  # CLOSED
            # During closed hours, return previous close and next open
            if now.time() < self.PRE_MARKET_START:
                # Before pre-market, previous day's after hours
                yesterday = today - timedelta(days=1)
                start = self.timezone.localize(datetime.combine(yesterday, self.AFTER_HOURS_END))
                end = self.timezone.localize(datetime.combine(today, self.PRE_MARKET_START))
            else:
                # After after-hours
                start = self.timezone.localize(datetime.combine(today, self.AFTER_HOURS_END))
                tomorrow = today + timedelta(days=1)
                end = self.timezone.localize(datetime.combine(tomorrow, self.PRE_MARKET_START))

        return start, end

    def _get_next_session(
        self,
        now: datetime,
        current: MarketSession
    ) -> Tuple[MarketSession, datetime]:
        """Get the next trading session and its start time"""
        today = now.date()

        if current == MarketSession.CLOSED:
            # Find next trading day
            next_day = self._get_next_trading_day(today)
            next_start = self.timezone.localize(
                datetime.combine(next_day, self.PRE_MARKET_START)
            )
            return MarketSession.PRE_MARKET, next_start

        elif current == MarketSession.PRE_MARKET:
            next_start = self.timezone.localize(
                datetime.combine(today, self.REGULAR_START)
            )
            return MarketSession.REGULAR, next_start

        elif current == MarketSession.REGULAR:
            next_start = self.timezone.localize(
                datetime.combine(today, self.REGULAR_END)
            )
            return MarketSession.AFTER_HOURS, next_start

        else:  # AFTER_HOURS
            next_day = self._get_next_trading_day(today + timedelta(days=1))
            next_start = self.timezone.localize(
                datetime.combine(next_day, self.PRE_MARKET_START)
            )
            return MarketSession.PRE_MARKET, next_start

    def _get_next_trading_day(self, from_date) -> datetime:
        """Find the next trading day (skipping weekends and holidays)"""
        check_date = from_date
        for _ in range(10):  # Check up to 10 days ahead
            if check_date.weekday() < 5 and not self._is_holiday(check_date):
                return check_date
            check_date += timedelta(days=1)
        return check_date

    def _is_holiday(self, date) -> bool:
        """Check if a date is a market holiday"""
        return date in [h.date() for h in self.HOLIDAYS_2024]

    def wait_for_session(self, target_session: MarketSession, timeout: int = None) -> bool:
        """
        Wait until target session begins.

        Args:
            target_session: Session to wait for
            timeout: Maximum seconds to wait (None = infinite)

        Returns:
            True if session started, False if timeout
        """
        import time as time_module

        start_time = time_module.time()

        while True:
            current = self.get_current_session()
            if current == target_session:
                return True

            if timeout and (time_module.time() - start_time) >= timeout:
                return False

            # Check every second
            time_module.sleep(1)

    def on_session_change(self, callback):
        """
        Register callback for session changes.
        Callback receives (old_session, new_session, session_info).
        """
        import threading

        def monitor():
            last_session = self.get_current_session()
            while True:
                current = self.get_current_session()
                if current != last_session:
                    info = self.get_session_info()
                    try:
                        callback(last_session, current, info)
                    except Exception as e:
                        logger.error(f"Session change callback error: {e}")
                    last_session = current

                import time as time_module
                time_module.sleep(1)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread

    def get_session_for_order(self, extended_hours: bool = True) -> str:
        """
        Get the appropriate session parameter for order placement.

        Returns Futu API session parameter.
        """
        session = self.get_current_session()

        if session == MarketSession.REGULAR:
            return "NORMAL"
        elif session in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS):
            if extended_hours:
                return "EXTENDED"
            else:
                return "NORMAL"
        else:
            return "NORMAL"


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
