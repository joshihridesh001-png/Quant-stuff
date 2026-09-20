"""FastAPI application factory and entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from sqlalchemy import text

from quant.api.middleware import CorrelationAndTimingMiddleware, register_exception_handlers
from quant.api.v1.endpoints import (
    auth,
    autonomous,
    events,
    gateways,
    genotypes,
    market_data,
    mcp,
    news,
    orders,
    pre_trade,
    providers,
    risk,
    streaming,
)
from quant.core.config import get_settings
from quant.infrastructure.database.session import engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan manager."""
    # Startup verification
    yield
    # Graceful shutdown: stop autonomous trader daemon if running and dispose connection pool
    from quant.api.dependencies import _autonomous_trader

    if _autonomous_trader is not None:
        await _autonomous_trader.stop()
    await engine.dispose()


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI ASGI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="Institutional-grade, multi-algorithmic market prediction and alpha generation engine.",
        lifespan=lifespan,
    )

    # 1. Register RFC 7807 Exception Handlers
    register_exception_handlers(app)

    # 2. Add ASGI Middleware (Timing, Request ID, CORS)
    app.add_middleware(CorrelationAndTimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Mount API Routers
    app.include_router(events.router, prefix=settings.API_V1_PREFIX)
    app.include_router(genotypes.router, prefix=settings.API_V1_PREFIX)
    app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
    app.include_router(market_data.router, prefix=settings.API_V1_PREFIX)
    app.include_router(orders.router, prefix=settings.API_V1_PREFIX)
    app.include_router(risk.router, prefix=settings.API_V1_PREFIX)
    app.include_router(gateways.router, prefix=settings.API_V1_PREFIX)
    app.include_router(streaming.router, prefix=settings.API_V1_PREFIX)
    app.include_router(autonomous.router, prefix=settings.API_V1_PREFIX)
    app.include_router(news.router, prefix=settings.API_V1_PREFIX)
    app.include_router(providers.router, prefix=settings.API_V1_PREFIX)
    app.include_router(pre_trade.router, prefix=settings.API_V1_PREFIX)
    app.include_router(mcp.router, prefix=settings.API_V1_PREFIX)

    # 4. System Health Check Endpoint
    @app.get("/healthz", tags=["System Health"], summary="Liveness & Readiness Probe")
    async def health_check() -> dict[str, str]:
        # Verify database connection readiness
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            db_status = "healthy"
        except Exception as exc:
            db_status = f"unhealthy: {str(exc)}"

        return {
            "status": "ok" if db_status == "healthy" else "degraded",
            "database": db_status,
            "environment": settings.ENVIRONMENT,
        }

    # 5. OpenMetrics / Prometheus Telemetry Endpoint
    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        tags=["System Health"],
        summary="Prometheus & OpenMetrics Telemetry",
    )
    def prometheus_metrics() -> PlainTextResponse:
        """Expose institutional quantitative telemetry in standard OpenMetrics format."""
        from quant.api.dependencies import (
            _autonomous_trader,
            _risk_orchestrator,
            get_pre_trade_gate,
        )

        gate = get_pre_trade_gate()
        history = gate.history
        evaluations_total = len(history)
        allowed_count = sum(1 for d in history if d.allowed)
        blocked_count = sum(1 for d in history if not d.allowed)

        kill_switch_val = (
            1 if (_risk_orchestrator and _risk_orchestrator.kill_switch.is_active) else 0
        )
        swarm_val = 1 if (_autonomous_trader and _autonomous_trader.state.value == "RUNNING") else 0
        nav_val = _risk_orchestrator.state.current_equity if _risk_orchestrator else 10_000.0
        cash_val = _risk_orchestrator.state.cash if _risk_orchestrator else 10_000.0

        lines = [
            "# HELP quant_up Heartbeat gauge of the quantitative engine application",
            "# TYPE quant_up gauge",
            f'quant_up{{environment="{settings.ENVIRONMENT}"}} 1',
            "# HELP quant_kill_switch_active Status of emergency panic kill switch lockout (1=locked, 0=armed)",
            "# TYPE quant_kill_switch_active gauge",
            f"quant_kill_switch_active {kill_switch_val}",
            "# HELP quant_autonomous_swarm_active Status of autonomous swarm multi-algorithmic trader (1=active, 0=idle)",
            "# TYPE quant_autonomous_swarm_active gauge",
            f"quant_autonomous_swarm_active {swarm_val}",
            "# HELP quant_portfolio_nav Current portfolio net asset value in base currency",
            "# TYPE quant_portfolio_nav gauge",
            f"quant_portfolio_nav {nav_val:.2f}",
            "# HELP quant_portfolio_cash Current portfolio unencumbered cash balance in base currency",
            "# TYPE quant_portfolio_cash gauge",
            f"quant_portfolio_cash {cash_val:.2f}",
            "# HELP quant_pre_trade_evaluations_total Total orders processed through Bayesian Pre-Trade Gate",
            "# TYPE quant_pre_trade_evaluations_total counter",
            f'quant_pre_trade_evaluations_total{{verdict="allowed"}} {allowed_count}',
            f'quant_pre_trade_evaluations_total{{verdict="blocked"}} {blocked_count}',
            f"quant_pre_trade_evaluations_total_count {evaluations_total}",
            "",
        ]
        return PlainTextResponse(content="\n".join(lines), media_type="text/plain; version=0.0.4")

    # 6. Interactive Trading Terminal Dashboard
    @app.get(
        "/dashboard",
        response_class=HTMLResponse,
        tags=["Dashboard"],
        summary="Institutional Trading Terminal UI",
    )
    @app.get("/terminal", response_class=HTMLResponse, include_in_schema=False)
    async def get_trading_terminal() -> HTMLResponse:
        """Serve the interactive institutional execution trading terminal dashboard."""
        terminal_path = Path(__file__).parent / "templates" / "trading_terminal.html"
        if terminal_path.exists():
            return HTMLResponse(content=terminal_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Trading Terminal Template Not Found</h1>", status_code=404)

    # 6. Root Redirect to Interactive Documentation
    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        """Redirect root requests to interactive Swagger UI documentation."""
        return RedirectResponse(url="/docs")

    return app


app = create_application()
