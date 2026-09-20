# ==============================================================================
# Multi-Stage Hardened Production Dockerfile for Quant Alpha Engine
# Python 3.13-slim runtime with unprivileged user (UID 10001) & signal trapping
# ==============================================================================

# --- Stage 1: Build & Dependency Wheel Cache ---
FROM python:3.13-slim AS builder

WORKDIR /build

# Install system compilation toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies into wheel cache
COPY pyproject.toml .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -e .

# --- Stage 2: Hardened Runtime Container ---
FROM python:3.13-slim AS runtime

# System runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Hardened unprivileged execution user
RUN groupadd -g 10001 quant && \
    useradd -u 10001 -g quant -s /bin/bash -m quant

# Working directory & data volume setup
WORKDIR /app
RUN mkdir -p /app/data /app/src && \
    chown -R quant:quant /app

# Install wheels from builder stage
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/* && rm -rf /tmp/wheels

# Copy application source and configuration
COPY --chown=quant:quant pyproject.toml .
COPY --chown=quant:quant alembic.ini* .
COPY --chown=quant:quant src/ /app/src/

# Install application in editable/link mode for clean module discovery
RUN pip install --no-cache-dir --no-deps -e .

# Security & runtime execution environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    APP_DATA_DIR=/app/data

# Switch to non-root UID 10001
USER 10001:10001

# Liveness and readiness health probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Expose HTTP ASGI service port
EXPOSE 8000

# PID 1 direct signal trapping for graceful lifespan shutdown on SIGTERM
CMD ["uvicorn", "quant.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
