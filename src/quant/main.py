"""FastAPI application factory and entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text

from quant.api.middleware import CorrelationAndTimingMiddleware, register_exception_handlers
from quant.api.v1.endpoints import (
    auth,
    autonomous,
    events,
    gateways,
    genotypes,
    market_data,
    orders,
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

    # 5. Interactive Trading Terminal Dashboard
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
