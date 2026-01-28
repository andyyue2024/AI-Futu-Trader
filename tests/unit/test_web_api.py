"""
Unit tests for Web API
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestWebAPI:
    """Test Web API endpoints"""

    def test_create_app(self):
        """Test app creation"""
        from src.web.api import create_app

        app = create_app()
        assert app is not None
        assert app.title == "AI Futu Trader API"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status endpoint"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "current_session" in data

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test session endpoint"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/session")

        assert response.status_code == 200
        data = response.json()
        assert "current" in data
        assert "next" in data
        assert "progress" in data

    @pytest.mark.asyncio
    async def test_get_symbols(self):
        """Test symbols endpoint"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/symbols")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_config(self):
        """Test config endpoint"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "futu_host" in data
        assert "llm_provider" in data

    @pytest.mark.asyncio
    async def test_root_returns_html(self):
        """Test root returns HTML dashboard"""
        from src.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AI Futu Trader" in response.text
