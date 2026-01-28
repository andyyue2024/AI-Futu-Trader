"""
Web module - FastAPI-based Web interface
"""
from .api import app, create_app, run_server, set_engine, get_engine

__all__ = [
    "app",
    "create_app",
    "run_server",
    "set_engine",
    "get_engine",
]
