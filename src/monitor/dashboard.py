"""
System Dashboard - Real-time system health and performance monitoring
Provides comprehensive view of all trading system metrics
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

from src.core.logger import get_logger
from src.core.session_manager import get_session_manager
from src.core.symbols import get_symbol_registry

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class TargetCompliance:
    """Compliance with prompt.txt targets"""
    # Latency targets
    order_latency_target_ms: float = 1.4
    order_latency_actual_ms: float = 0.0
    order_latency_meets: bool = False

    pipeline_latency_target_ms: float = 1000.0
    pipeline_latency_actual_ms: float = 0.0
    pipeline_latency_meets: bool = False

    # Slippage target
    slippage_target_pct: float = 0.002
    slippage_actual_pct: float = 0.0
    slippage_meets: bool = False

    # Volume target
    daily_volume_target_usd: float = 50000.0
    daily_volume_actual_usd: float = 0.0
    daily_volume_meets: bool = False

    # Fill rate target
    fill_rate_target_pct: float = 0.95
    fill_rate_actual_pct: float = 0.0
    fill_rate_meets: bool = False

    # Sharpe target
    sharpe_target: float = 2.0
    sharpe_actual: float = 0.0
    sharpe_meets: bool = False

    # Drawdown target
    max_drawdown_target_pct: float = 0.15
    max_drawdown_actual_pct: float = 0.0
    max_drawdown_meets: bool = False

    # Circuit breaker
    circuit_breaker_threshold_pct: float = 0.03
    circuit_breaker_triggered: bool = False

    @property
    def all_targets_met(self) -> bool:
        """Check if all targets are met"""
        return (
            self.order_latency_meets and
            self.pipeline_latency_meets and
            self.slippage_meets and
            self.daily_volume_meets and
            self.fill_rate_meets and
            self.sharpe_meets and
            self.max_drawdown_meets and
            not self.circuit_breaker_triggered
        )

    @property
    def targets_met_count(self) -> int:
        """Count of met targets"""
        count = 0
        if self.order_latency_meets:
            count += 1
        if self.pipeline_latency_meets:
            count += 1
        if self.slippage_meets:
            count += 1
        if self.daily_volume_meets:
            count += 1
        if self.fill_rate_meets:
            count += 1
        if self.sharpe_meets:
            count += 1
        if self.max_drawdown_meets:
            count += 1
        if not self.circuit_breaker_triggered:
            count += 1
        return count

    @property
    def total_targets(self) -> int:
        return 8


@dataclass
class DashboardData:
    """Complete dashboard data"""
    timestamp: datetime = field(default_factory=datetime.now)

    # System status
    health_status: HealthStatus = HealthStatus.UNKNOWN
    is_trading: bool = False

    # Session info
    current_session: str = "unknown"
    session_progress_pct: float = 0.0
    trading_allowed: bool = False

    # Portfolio
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0

    # Positions
    open_positions: int = 0

    # Trading activity
    trades_today: int = 0
    orders_today: int = 0

    # Target compliance
    compliance: TargetCompliance = field(default_factory=TargetCompliance)

    # Alerts
    active_alerts: int = 0

    # System resources
    cpu_percent: float = 0.0
    memory_percent: float = 0.0

    # Connection status
    quote_connected: bool = False
    trade_connected: bool = False

    # Errors
    errors_last_hour: int = 0


class SystemDashboard:
    """
    Comprehensive system dashboard.
    Aggregates data from all modules for monitoring.
    """

    def __init__(self):
        self._last_update: Optional[datetime] = None
        self._cached_data: Optional[DashboardData] = None

    def get_dashboard(self, force_refresh: bool = False) -> DashboardData:
        """Get complete dashboard data"""
        # Use cache if recent
        if (
            not force_refresh and
            self._cached_data and
            self._last_update and
            (datetime.now() - self._last_update).seconds < 5
        ):
            return self._cached_data

        data = DashboardData()

        # Get session info
        try:
            session_mgr = get_session_manager()
            info = session_mgr.get_session_info()
            data.current_session = info.session.value
            data.session_progress_pct = info.progress_pct
            data.trading_allowed = info.is_trading_allowed
        except Exception as e:
            logger.error(f"Error getting session info: {e}")

        # Get compliance data
        data.compliance = self._get_compliance()

        # Get system resources
        try:
            from src.monitor.performance import get_performance_monitor
            monitor = get_performance_monitor()
            metrics = monitor.get_current_metrics()
            data.cpu_percent = metrics.get("cpu_percent", 0)
            data.memory_percent = metrics.get("memory_percent", 0)
            data.errors_last_hour = metrics.get("errors_last_hour", 0)
        except Exception as e:
            logger.debug(f"Performance monitor not available: {e}")

        # Determine health status
        data.health_status = self._determine_health(data)

        # Cache and return
        self._cached_data = data
        self._last_update = datetime.now()

        return data

    def _get_compliance(self) -> TargetCompliance:
        """Get target compliance data"""
        compliance = TargetCompliance()

        # Get slippage data
        try:
            from src.risk.slippage_controller import get_slippage_controller
            controller = get_slippage_controller()
            stats = controller.get_stats(24)
            compliance.slippage_actual_pct = stats.avg_slippage_pct
            compliance.slippage_meets = stats.avg_slippage_pct <= compliance.slippage_target_pct
        except Exception:
            pass

        # Get fill rate
        try:
            from src.risk.slippage_controller import get_fill_rate_monitor
            monitor = get_fill_rate_monitor()
            compliance.fill_rate_actual_pct = monitor.get_fill_rate(24)
            compliance.fill_rate_meets = compliance.fill_rate_actual_pct >= compliance.fill_rate_target_pct
        except Exception:
            pass

        # Get volume
        try:
            from src.risk.slippage_controller import get_volume_tracker
            tracker = get_volume_tracker()
            compliance.daily_volume_actual_usd = tracker.get_today_volume()
            compliance.daily_volume_meets = compliance.daily_volume_actual_usd >= compliance.daily_volume_target_usd
        except Exception:
            pass

        # Get Sharpe and drawdown
        try:
            from src.risk.sharpe_calculator import get_sharpe_calculator
            calc = get_sharpe_calculator()
            compliance.sharpe_actual = calc.calculate_sharpe()
            compliance.sharpe_meets = compliance.sharpe_actual >= compliance.sharpe_target

            _, max_dd_pct = calc.calculate_max_drawdown()
            compliance.max_drawdown_actual_pct = max_dd_pct
            compliance.max_drawdown_meets = max_dd_pct <= compliance.max_drawdown_target_pct
        except Exception:
            pass

        # Get latency
        try:
            from src.action.order_optimizer import get_order_optimizer
            optimizer = get_order_optimizer()
            metrics = optimizer.get_latency_metrics()
            compliance.order_latency_actual_ms = metrics.p95_order_latency_ms
            compliance.order_latency_meets = compliance.order_latency_actual_ms <= compliance.order_latency_target_ms
        except Exception:
            pass

        return compliance

    def _determine_health(self, data: DashboardData) -> HealthStatus:
        """Determine overall health status"""
        # Critical conditions
        if data.compliance.circuit_breaker_triggered:
            return HealthStatus.CRITICAL
        if data.errors_last_hour > 50:
            return HealthStatus.CRITICAL
        if data.cpu_percent > 95 or data.memory_percent > 95:
            return HealthStatus.CRITICAL

        # Warning conditions
        if data.errors_last_hour > 10:
            return HealthStatus.WARNING
        if data.cpu_percent > 80 or data.memory_percent > 80:
            return HealthStatus.WARNING
        if data.compliance.targets_met_count < 5:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def get_compliance_summary(self) -> str:
        """Get human-readable compliance summary"""
        data = self.get_dashboard()
        c = data.compliance

        lines = []
        lines.append("=" * 60)
        lines.append("📊 TARGET COMPLIANCE SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Overall: {c.targets_met_count}/{c.total_targets} targets met")
        lines.append("")

        # Latency
        icon = "✅" if c.order_latency_meets else "❌"
        lines.append(f"{icon} Order Latency: {c.order_latency_actual_ms:.2f}ms (target: ≤{c.order_latency_target_ms}ms)")

        icon = "✅" if c.pipeline_latency_meets else "❌"
        lines.append(f"{icon} Pipeline Latency: {c.pipeline_latency_actual_ms:.0f}ms (target: ≤{c.pipeline_latency_target_ms:.0f}ms)")

        # Slippage
        icon = "✅" if c.slippage_meets else "❌"
        lines.append(f"{icon} Slippage: {c.slippage_actual_pct:.4%} (target: ≤{c.slippage_target_pct:.2%})")

        # Volume
        icon = "✅" if c.daily_volume_meets else "❌"
        lines.append(f"{icon} Daily Volume: ${c.daily_volume_actual_usd:,.2f} (target: ≥${c.daily_volume_target_usd:,.0f})")

        # Fill Rate
        icon = "✅" if c.fill_rate_meets else "❌"
        lines.append(f"{icon} Fill Rate: {c.fill_rate_actual_pct:.1%} (target: ≥{c.fill_rate_target_pct:.0%})")

        # Sharpe
        icon = "✅" if c.sharpe_meets else "❌"
        lines.append(f"{icon} Sharpe Ratio: {c.sharpe_actual:.2f} (target: ≥{c.sharpe_target})")

        # Drawdown
        icon = "✅" if c.max_drawdown_meets else "❌"
        lines.append(f"{icon} Max Drawdown: {c.max_drawdown_actual_pct:.2%} (target: ≤{c.max_drawdown_target_pct:.0%})")

        # Circuit Breaker
        icon = "✅" if not c.circuit_breaker_triggered else "🚨"
        status = "NOT TRIGGERED" if not c.circuit_breaker_triggered else "TRIGGERED"
        lines.append(f"{icon} Circuit Breaker: {status} (threshold: {c.circuit_breaker_threshold_pct:.0%})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def print_dashboard(self):
        """Print dashboard to console"""
        data = self.get_dashboard()

        print("\n" + "=" * 70)
        print("🤖 AI FUTU TRADER DASHBOARD")
        print("=" * 70)
        print(f"  Timestamp: {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Health: {data.health_status.value.upper()}")
        print("")
        print("📅 Session:")
        print(f"  Current: {data.current_session}")
        print(f"  Progress: {data.session_progress_pct:.1f}%")
        print(f"  Trading Allowed: {data.trading_allowed}")
        print("")
        print("💻 System:")
        print(f"  CPU: {data.cpu_percent:.1f}%")
        print(f"  Memory: {data.memory_percent:.1f}%")
        print(f"  Errors (1h): {data.errors_last_hour}")
        print("")
        print(self.get_compliance_summary())


# Singleton instance
_dashboard: Optional[SystemDashboard] = None


def get_dashboard() -> SystemDashboard:
    """Get global dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = SystemDashboard()
    return _dashboard
