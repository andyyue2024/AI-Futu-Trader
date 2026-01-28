"""
Web API - FastAPI-based Web interface for AI Futu Trader
Provides REST API endpoints for monitoring and control
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.symbols import get_symbol_registry
from src.core.session_manager import get_session_manager

logger = get_logger(__name__)

# Pydantic models for API
class SymbolInfo(BaseModel):
    symbol: str
    futu_code: str
    name: Optional[str]
    instrument_type: str
    is_active: bool


class PositionInfo(BaseModel):
    symbol: str
    futu_code: str
    direction: str
    quantity: int
    avg_cost: float
    unrealized_pnl: float
    realized_pnl: float


class TradeInfo(BaseModel):
    trade_id: str
    symbol: str
    entry_time: str
    entry_price: float
    entry_side: str
    quantity: int
    exit_time: Optional[str]
    exit_price: Optional[float]
    pnl: float
    status: str


class PerformanceMetrics(BaseModel):
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    fill_rate: float
    avg_latency_ms: float


class SystemStatus(BaseModel):
    is_running: bool
    current_session: str
    session_progress: float
    quote_connected: bool
    trade_connected: bool
    circuit_breaker_active: bool
    last_update: str


class TradingCommand(BaseModel):
    action: str  # start, stop, pause
    symbols: Optional[List[str]] = None


class OrderRequest(BaseModel):
    symbol: str
    action: str  # long, short, flat
    quantity: Optional[int] = None


# Global state (will be replaced with actual engine reference)
_engine = None
_app_state = {
    "is_running": False,
    "start_time": None,
}


def get_engine():
    """Get trading engine instance"""
    global _engine
    return _engine


def set_engine(engine):
    """Set trading engine instance"""
    global _engine
    _engine = engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Web API starting up...")
    yield
    logger.info("Web API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Futu Trader API",
    description="REST API for AI Futu Trader monitoring and control",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# System Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Futu Trader</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    </head>
    <body class="bg-gray-900 text-white">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8">🤖 AI Futu Trader Dashboard</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <!-- Status Card -->
                <div class="bg-gray-800 rounded-lg p-4" hx-get="/api/status" hx-trigger="every 5s" hx-swap="innerHTML">
                    <h3 class="text-lg font-semibold mb-2">System Status</h3>
                    <p class="text-gray-400">Loading...</p>
                </div>
                
                <!-- P&L Card -->
                <div class="bg-gray-800 rounded-lg p-4" hx-get="/api/metrics/summary" hx-trigger="every 10s" hx-swap="innerHTML">
                    <h3 class="text-lg font-semibold mb-2">P&L</h3>
                    <p class="text-gray-400">Loading...</p>
                </div>
                
                <!-- Session Card -->
                <div class="bg-gray-800 rounded-lg p-4" hx-get="/api/session" hx-trigger="every 5s" hx-swap="innerHTML">
                    <h3 class="text-lg font-semibold mb-2">Market Session</h3>
                    <p class="text-gray-400">Loading...</p>
                </div>
                
                <!-- Quick Actions -->
                <div class="bg-gray-800 rounded-lg p-4">
                    <h3 class="text-lg font-semibold mb-2">Quick Actions</h3>
                    <div class="space-y-2">
                        <button onclick="startTrading()" class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded">Start</button>
                        <button onclick="stopTrading()" class="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded">Stop</button>
                    </div>
                </div>
            </div>
            
            <!-- Positions Table -->
            <div class="bg-gray-800 rounded-lg p-4 mb-8">
                <h3 class="text-lg font-semibold mb-4">Open Positions</h3>
                <div hx-get="/api/positions" hx-trigger="every 5s" hx-swap="innerHTML">
                    <p class="text-gray-400">Loading...</p>
                </div>
            </div>
            
            <!-- Recent Trades -->
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-lg font-semibold mb-4">Recent Trades</h3>
                <div hx-get="/api/trades/recent" hx-trigger="every 10s" hx-swap="innerHTML">
                    <p class="text-gray-400">Loading...</p>
                </div>
            </div>
        </div>
        
        <script>
            async function startTrading() {
                await fetch('/api/trading/start', {method: 'POST'});
                alert('Trading started');
            }
            async function stopTrading() {
                await fetch('/api/trading/stop', {method: 'POST'});
                alert('Trading stopped');
            }
        </script>
    </body>
    </html>
    """


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    session_mgr = get_session_manager()
    session_info = session_mgr.get_session_info()

    engine = get_engine()

    return {
        "is_running": _app_state.get("is_running", False),
        "current_session": session_info.session.value,
        "session_progress": round(session_info.progress_pct, 1),
        "trading_allowed": session_info.is_trading_allowed,
        "seconds_to_close": session_info.seconds_to_close,
        "circuit_breaker_active": False,  # Will be from engine
        "last_update": datetime.now().isoformat()
    }


@app.get("/api/session")
async def get_session():
    """Get market session info"""
    session_mgr = get_session_manager()
    info = session_mgr.get_session_info()

    return {
        "current": info.session.value,
        "next": info.next_session.value,
        "progress": round(info.progress_pct, 1),
        "is_trading_allowed": info.is_trading_allowed,
        "seconds_to_close": info.seconds_to_close,
        "seconds_to_next_open": info.seconds_to_next_open
    }


# ==========================================
# Trading Control Endpoints
# ==========================================

@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start trading"""
    _app_state["is_running"] = True
    _app_state["start_time"] = datetime.now()

    engine = get_engine()
    if engine:
        background_tasks.add_task(engine.start)

    return {"status": "started", "timestamp": datetime.now().isoformat()}


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading"""
    _app_state["is_running"] = False

    engine = get_engine()
    if engine:
        await engine.stop()

    return {"status": "stopped", "timestamp": datetime.now().isoformat()}


@app.post("/api/order")
async def place_order(order: OrderRequest):
    """Place a manual order"""
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    # Validate action
    if order.action not in ["long", "short", "flat"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    # Place order through engine
    # result = await engine.execute_action(order.symbol, order.action, order.quantity)

    return {
        "status": "submitted",
        "symbol": order.symbol,
        "action": order.action,
        "timestamp": datetime.now().isoformat()
    }


# ==========================================
# Symbols Endpoints
# ==========================================

@app.get("/api/symbols", response_model=List[SymbolInfo])
async def get_symbols():
    """Get all registered symbols"""
    registry = get_symbol_registry()
    symbols = []

    for sym in registry.all_symbols():
        symbols.append(SymbolInfo(
            symbol=sym.symbol,
            futu_code=sym.futu_code,
            name=sym.name,
            instrument_type=sym.instrument_type.value,
            is_active=sym.futu_code in registry.active_symbols
        ))

    return symbols


@app.post("/api/symbols/{futu_code}/activate")
async def activate_symbol(futu_code: str):
    """Activate a symbol for trading"""
    registry = get_symbol_registry()
    registry.activate(futu_code)
    return {"status": "activated", "symbol": futu_code}


@app.post("/api/symbols/{futu_code}/deactivate")
async def deactivate_symbol(futu_code: str):
    """Deactivate a symbol from trading"""
    registry = get_symbol_registry()
    registry.deactivate(futu_code)
    return {"status": "deactivated", "symbol": futu_code}


# ==========================================
# Positions Endpoints
# ==========================================

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    engine = get_engine()

    # Mock data for demo
    positions = [
        {
            "symbol": "TQQQ",
            "futu_code": "US.TQQQ",
            "direction": "LONG",
            "quantity": 100,
            "avg_cost": 50.25,
            "current_price": 51.00,
            "unrealized_pnl": 75.00,
            "pnl_pct": 1.49
        }
    ]

    return positions


# ==========================================
# Trades Endpoints
# ==========================================

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = Query(default=20, le=100)):
    """Get recent trades"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        trades = db.get_recent_trades(limit)

        return [
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "entry_side": t.entry_side,
                "quantity": t.quantity,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "status": t.status
            }
            for t in trades
        ]
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []


@app.get("/api/trades/date/{trade_date}")
async def get_trades_by_date(trade_date: str):
    """Get trades for a specific date"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()

        dt = date.fromisoformat(trade_date)
        trades = db.get_trades_by_date(dt)

        return [t.to_dict() for t in trades]
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
# Metrics Endpoints
# ==========================================

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()
        stats = db.get_trading_stats(30)

        return {
            "total_pnl": round(stats.get("total_pnl", 0), 2),
            "total_trades": stats.get("total_trades", 0),
            "win_rate": round(stats.get("win_rate", 0) * 100, 1),
            "avg_pnl": round(stats.get("avg_pnl", 0), 2),
            "avg_slippage_pct": round(stats.get("avg_slippage", 0) * 100, 4),
            "avg_latency_ms": round(stats.get("avg_latency_ms", 0), 2)
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "total_pnl": 0,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0
        }


@app.get("/api/metrics/daily")
async def get_daily_metrics(days: int = Query(default=30, le=365)):
    """Get daily performance metrics"""
    try:
        from src.data.persistence import get_trade_database
        db = get_trade_database()

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        records = db.get_daily_performance(start_date, end_date)

        return [r.to_dict() for r in records]
    except Exception as e:
        logger.error(f"Error getting daily metrics: {e}")
        return []


# ==========================================
# Reports Endpoints
# ==========================================

@app.get("/api/reports/generate")
async def generate_report(
    format: str = Query(default="pdf", regex="^(pdf|excel)$"),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
    background_tasks: BackgroundTasks = None
):
    """Generate trading report"""
    try:
        from src.report.generator import ReportGenerator

        generator = ReportGenerator()

        start = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=30)
        end = date.fromisoformat(end_date) if end_date else date.today()

        if format == "pdf":
            filepath = generator.generate_pdf(start, end)
        else:
            filepath = generator.generate_excel(start, end)

        return {
            "status": "generated",
            "format": format,
            "filepath": filepath,
            "download_url": f"/api/reports/download/{filepath.split('/')[-1]}"
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Report generation not available")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    import os
    filepath = f"reports/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(filepath)


# ==========================================
# Configuration Endpoints
# ==========================================

@app.get("/api/config")
async def get_config():
    """Get current configuration (safe subset)"""
    settings = get_settings()

    return {
        "futu_host": settings.futu_host,
        "futu_port": settings.futu_port,
        "trade_env": settings.futu_trade_env,
        "llm_provider": settings.llm_provider,
        "trading_symbols": settings.trading_symbols,
        "max_daily_drawdown": settings.max_daily_drawdown,
        "max_total_drawdown": settings.max_total_drawdown,
        "slippage_tolerance": settings.slippage_tolerance
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
