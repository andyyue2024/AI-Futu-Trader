"""
Performance Monitor - Enhanced performance monitoring and profiling
Tracks system performance, memory usage, and provides diagnostics
"""
import os
import sys
import time
import threading
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import functools

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime = field(default_factory=datetime.now)

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0

    # Memory
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0

    # Process
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_threads: int = 0

    # Network
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0

    # Disk
    disk_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime

    # Latencies (ms)
    avg_order_latency: float = 0.0
    p95_order_latency: float = 0.0
    avg_quote_latency: float = 0.0
    avg_pipeline_latency: float = 0.0

    # Throughput
    orders_per_minute: float = 0.0
    quotes_per_second: float = 0.0

    # Errors
    error_rate: float = 0.0
    errors_last_hour: int = 0

    # System
    system_metrics: Optional[SystemMetrics] = None


@dataclass
class ErrorRecord:
    """Error record for tracking"""
    timestamp: datetime
    error_type: str
    message: str
    module: str
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class ErrorTracker:
    """
    Tracks and analyzes errors for monitoring.
    """

    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self._errors: deque = deque(maxlen=max_errors)
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record_error(
        self,
        error: Exception,
        module: str = "",
        context: Dict = None
    ):
        """Record an error"""
        import traceback

        with self._lock:
            record = ErrorRecord(
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                message=str(error),
                module=module,
                stack_trace=traceback.format_exc(),
                context=context or {}
            )

            self._errors.append(record)

            # Count by type
            error_key = f"{module}:{type(error).__name__}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

            logger.error(f"Error recorded: [{module}] {type(error).__name__}: {error}")

    def get_errors_since(self, since: datetime) -> List[ErrorRecord]:
        """Get errors since a timestamp"""
        with self._lock:
            return [e for e in self._errors if e.timestamp >= since]

    def get_error_rate(self, window_minutes: int = 60) -> float:
        """Get error rate (errors per minute)"""
        since = datetime.now() - timedelta(minutes=window_minutes)
        errors = self.get_errors_since(since)
        return len(errors) / window_minutes if window_minutes > 0 else 0

    def get_error_summary(self) -> Dict[str, int]:
        """Get error count by type"""
        with self._lock:
            return self._error_counts.copy()

    def get_recent_errors(self, limit: int = 20) -> List[ErrorRecord]:
        """Get most recent errors"""
        with self._lock:
            return list(self._errors)[-limit:]

    def clear(self):
        """Clear all errors"""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()


class FunctionProfiler:
    """
    Profiles function execution times.
    """

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._profiles: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def profile(self, func_name: str = None):
        """Decorator to profile a function"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._record(name, elapsed_ms)

            return wrapper
        return decorator

    def _record(self, func_name: str, elapsed_ms: float):
        """Record execution time"""
        with self._lock:
            if func_name not in self._profiles:
                self._profiles[func_name] = deque(maxlen=self.max_samples)
            self._profiles[func_name].append(elapsed_ms)

    def get_stats(self, func_name: str) -> Dict[str, float]:
        """Get statistics for a function"""
        with self._lock:
            if func_name not in self._profiles:
                return {}

            samples = list(self._profiles[func_name])
            if not samples:
                return {}

            samples.sort()
            n = len(samples)

            return {
                "count": n,
                "avg_ms": sum(samples) / n,
                "min_ms": samples[0],
                "max_ms": samples[-1],
                "p50_ms": samples[n // 2],
                "p95_ms": samples[int(n * 0.95)],
                "p99_ms": samples[int(n * 0.99)]
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all profiled functions"""
        with self._lock:
            return {name: self.get_stats(name) for name in self._profiles.keys()}


class PerformanceMonitor:
    """
    Comprehensive performance monitoring.
    """

    def __init__(
        self,
        sample_interval_seconds: int = 60,
        max_history: int = 1440  # 24 hours at 1-minute intervals
    ):
        self.sample_interval = sample_interval_seconds
        self.max_history = max_history

        self._snapshots: deque = deque(maxlen=max_history)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Components
        self.error_tracker = ErrorTracker()
        self.profiler = FunctionProfiler()

        # Process handle
        self._process = psutil.Process(os.getpid())

        # Baseline network/disk for delta calculation
        self._baseline_net = None
        self._baseline_disk = None

    def start(self):
        """Start monitoring"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        logger.info("Performance monitor started")

    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Performance monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                self._snapshots.append(snapshot)

                # Check for anomalies
                self._check_anomalies(snapshot)

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self.sample_interval)

    def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot"""
        system = self._collect_system_metrics()

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            error_rate=self.error_tracker.get_error_rate(60),
            errors_last_hour=len(self.error_tracker.get_errors_since(
                datetime.now() - timedelta(hours=1)
            )),
            system_metrics=system
        )

        return snapshot

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        metrics = SystemMetrics()

        try:
            # CPU
            metrics.cpu_percent = psutil.cpu_percent()
            metrics.cpu_count = psutil.cpu_count()

            # Memory
            mem = psutil.virtual_memory()
            metrics.memory_percent = mem.percent
            metrics.memory_used_mb = mem.used / (1024 * 1024)
            metrics.memory_available_mb = mem.available / (1024 * 1024)

            # Process
            metrics.process_cpu_percent = self._process.cpu_percent()
            metrics.process_memory_mb = self._process.memory_info().rss / (1024 * 1024)
            metrics.process_threads = self._process.num_threads()

            # Network
            net = psutil.net_io_counters()
            if self._baseline_net is None:
                self._baseline_net = net

            metrics.network_sent_mb = (net.bytes_sent - self._baseline_net.bytes_sent) / (1024 * 1024)
            metrics.network_recv_mb = (net.bytes_recv - self._baseline_net.bytes_recv) / (1024 * 1024)

            # Disk
            disk = psutil.disk_usage('/')
            metrics.disk_percent = disk.percent

            io = psutil.disk_io_counters()
            if self._baseline_disk is None:
                self._baseline_disk = io

            if io:
                metrics.disk_read_mb = (io.read_bytes - self._baseline_disk.read_bytes) / (1024 * 1024)
                metrics.disk_write_mb = (io.write_bytes - self._baseline_disk.write_bytes) / (1024 * 1024)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics

    def _check_anomalies(self, snapshot: PerformanceSnapshot):
        """Check for performance anomalies"""
        if not snapshot.system_metrics:
            return

        metrics = snapshot.system_metrics

        # High memory usage
        if metrics.memory_percent > 90:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")

        # High CPU usage
        if metrics.cpu_percent > 90:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        # High error rate
        if snapshot.error_rate > 1:
            logger.warning(f"High error rate: {snapshot.error_rate:.2f} errors/min")

    def get_current_metrics(self) -> Dict:
        """Get current metrics"""
        metrics = self._collect_system_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_mb": round(metrics.memory_used_mb, 1),
            "process_memory_mb": round(metrics.process_memory_mb, 1),
            "process_threads": metrics.process_threads,
            "error_rate": round(self.error_tracker.get_error_rate(), 2),
            "errors_last_hour": len(self.error_tracker.get_errors_since(
                datetime.now() - timedelta(hours=1)
            ))
        }

    def get_history(self, hours: int = 1) -> List[Dict]:
        """Get performance history"""
        cutoff = datetime.now() - timedelta(hours=hours)

        history = []
        for snapshot in self._snapshots:
            if snapshot.timestamp >= cutoff:
                history.append({
                    "timestamp": snapshot.timestamp.isoformat(),
                    "error_rate": snapshot.error_rate,
                    "cpu_percent": snapshot.system_metrics.cpu_percent if snapshot.system_metrics else 0,
                    "memory_percent": snapshot.system_metrics.memory_percent if snapshot.system_metrics else 0,
                })

        return history

    def get_health_status(self) -> Dict:
        """Get overall health status"""
        metrics = self._collect_system_metrics()
        error_rate = self.error_tracker.get_error_rate()

        # Determine status
        status = "healthy"
        issues = []

        if metrics.memory_percent > 90:
            status = "critical"
            issues.append("High memory usage")
        elif metrics.memory_percent > 75:
            status = "warning"
            issues.append("Elevated memory usage")

        if metrics.cpu_percent > 90:
            status = "critical"
            issues.append("High CPU usage")
        elif metrics.cpu_percent > 75:
            if status != "critical":
                status = "warning"
            issues.append("Elevated CPU usage")

        if error_rate > 5:
            status = "critical"
            issues.append("High error rate")
        elif error_rate > 1:
            if status != "critical":
                status = "warning"
            issues.append("Elevated error rate")

        return {
            "status": status,
            "issues": issues,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "error_rate": error_rate
            },
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
