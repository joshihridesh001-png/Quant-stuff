"""ASGI Middleware for request correlation tracing, latency metrics, and RFC 7807 error formatting."""

import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class CorrelationAndTimingMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID and measures request latency in X-Process-Time-Ms."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register RFC 7807 Problem Details compliant exception handlers."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "type": "https://datatracker.ietf.org/doc/html/rfc7807#section-3",
                "title": "Bad Request",
                "status": status.HTTP_400_BAD_REQUEST,
                "detail": str(exc),
                "instance": str(request.url),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "type": "https://datatracker.ietf.org/doc/html/rfc7807#section-3",
                "title": "Internal Server Error",
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "detail": "An unexpected error occurred. Consult server logs.",
                "instance": str(request.url),
                "request_id": request_id,
            },
        )
