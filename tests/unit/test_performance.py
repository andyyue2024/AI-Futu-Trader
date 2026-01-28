"""
Unit tests for Performance Monitor
"""
import pytest
import time
from datetime import datetime, timedelta


class TestSystemMetrics:
    """Test SystemMetrics class"""

    def test_metrics_creation(self):
        """Test metrics creation"""
        from src.monitor.performance import SystemMetrics

        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0
        )

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0


class TestErrorTracker:
    """Test ErrorTracker class"""

    def test_tracker_creation(self):
        """Test tracker creation"""
        from src.monitor.performance import ErrorTracker

        tracker = ErrorTracker(max_errors=100)
        assert tracker.max_errors == 100

    def test_record_error(self):
        """Test recording an error"""
        from src.monitor.performance import ErrorTracker

        tracker = ErrorTracker()

        try:
            raise ValueError("Test error")
        except Exception as e:
            tracker.record_error(e, module="test")

        errors = tracker.get_recent_errors(10)
        assert len(errors) == 1
        assert errors[0].error_type == "ValueError"
        assert errors[0].message == "Test error"

    def test_error_rate(self):
        """Test error rate calculation"""
        from src.monitor.performance import ErrorTracker

        tracker = ErrorTracker()

        # Record several errors
        for i in range(5):
            try:
                raise RuntimeError(f"Error {i}")
            except Exception as e:
                tracker.record_error(e)

        rate = tracker.get_error_rate(60)
        assert rate > 0

    def test_error_summary(self):
        """Test error summary"""
        from src.monitor.performance import ErrorTracker

        tracker = ErrorTracker()

        tracker.record_error(ValueError("test"), module="mod1")
        tracker.record_error(ValueError("test2"), module="mod1")
        tracker.record_error(TypeError("test3"), module="mod2")

        summary = tracker.get_error_summary()

        assert "mod1:ValueError" in summary
        assert summary["mod1:ValueError"] == 2


class TestFunctionProfiler:
    """Test FunctionProfiler class"""

    def test_profiler_creation(self):
        """Test profiler creation"""
        from src.monitor.performance import FunctionProfiler

        profiler = FunctionProfiler(max_samples=100)
        assert profiler.max_samples == 100

    def test_profile_decorator(self):
        """Test profiling a function"""
        from src.monitor.performance import FunctionProfiler

        profiler = FunctionProfiler()

        @profiler.profile("test_func")
        def slow_function():
            time.sleep(0.01)
            return 42

        # Call function
        result = slow_function()
        assert result == 42

        # Check stats
        stats = profiler.get_stats("test_func")
        assert stats["count"] == 1
        assert stats["avg_ms"] >= 10  # At least 10ms

    def test_get_all_stats(self):
        """Test getting all function stats"""
        from src.monitor.performance import FunctionProfiler

        profiler = FunctionProfiler()

        @profiler.profile("func1")
        def func1():
            pass

        @profiler.profile("func2")
        def func2():
            pass

        func1()
        func2()
        func2()

        all_stats = profiler.get_all_stats()

        assert "func1" in all_stats
        assert "func2" in all_stats
        assert all_stats["func2"]["count"] == 2


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def test_monitor_creation(self):
        """Test monitor creation"""
        from src.monitor.performance import PerformanceMonitor

        monitor = PerformanceMonitor(sample_interval_seconds=60)
        assert monitor.sample_interval == 60

    def test_get_current_metrics(self):
        """Test getting current metrics"""
        from src.monitor.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        metrics = monitor.get_current_metrics()

        assert "timestamp" in metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "error_rate" in metrics

    def test_get_health_status(self):
        """Test health status check"""
        from src.monitor.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        status = monitor.get_health_status()

        assert "status" in status
        assert status["status"] in ["healthy", "warning", "critical"]
        assert "metrics" in status
        assert "issues" in status

    def test_error_tracking_integration(self):
        """Test error tracking through monitor"""
        from src.monitor.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        try:
            raise RuntimeError("Test error")
        except Exception as e:
            monitor.error_tracker.record_error(e, module="test")

        errors = monitor.error_tracker.get_recent_errors(10)
        assert len(errors) == 1


class TestPerformanceSnapshot:
    """Test PerformanceSnapshot class"""

    def test_snapshot_creation(self):
        """Test snapshot creation"""
        from src.monitor.performance import PerformanceSnapshot

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            avg_order_latency=1.2,
            p95_order_latency=2.0,
            error_rate=0.1
        )

        assert snapshot.avg_order_latency == 1.2
        assert snapshot.p95_order_latency == 2.0
        assert snapshot.error_rate == 0.1
