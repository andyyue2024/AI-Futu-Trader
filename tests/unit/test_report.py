"""
Unit tests for Report Generator
"""
import pytest
import tempfile
import os
from datetime import date, datetime, timedelta


class TestReportConfig:
    """Test ReportConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        from src.report.generator import ReportConfig

        config = ReportConfig()

        assert config.title == "AI Futu Trader Report"
        assert config.include_summary is True
        assert config.include_trades is True

    def test_custom_config(self):
        """Test custom configuration"""
        from src.report.generator import ReportConfig

        config = ReportConfig(
            title="Custom Report",
            include_charts=False
        )

        assert config.title == "Custom Report"
        assert config.include_charts is False


class TestReportData:
    """Test ReportData class"""

    def test_data_creation(self):
        """Test report data creation"""
        from src.report.generator import ReportData

        data = ReportData(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            starting_equity=100000.0,
            ending_equity=105000.0,
            total_pnl=5000.0
        )

        assert data.total_pnl == 5000.0
        assert data.ending_equity == 105000.0


class TestReportGenerator:
    """Test ReportGenerator class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_generator_creation(self, temp_dir):
        """Test generator creation"""
        from src.report.generator import ReportGenerator, ReportConfig

        config = ReportConfig(output_dir=temp_dir)
        generator = ReportGenerator(config)

        assert generator.config.output_dir == temp_dir

    def test_collect_data(self, temp_dir):
        """Test data collection"""
        from src.report.generator import ReportGenerator, ReportConfig

        config = ReportConfig(output_dir=temp_dir)
        generator = ReportGenerator(config)

        start = date.today() - timedelta(days=7)
        end = date.today()

        data = generator.collect_data(start, end)

        assert data.start_date == start
        assert data.end_date == end

    def test_generate_html(self, temp_dir):
        """Test HTML report generation"""
        from src.report.generator import ReportGenerator, ReportConfig

        config = ReportConfig(output_dir=temp_dir)
        generator = ReportGenerator(config)

        start = date.today() - timedelta(days=7)
        end = date.today()

        filepath = generator.generate_html(start, end)

        assert os.path.exists(filepath)
        assert filepath.endswith(".html")

        with open(filepath) as f:
            content = f.read()
            assert "AI Futu Trader Report" in content
