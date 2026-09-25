# ==============================================================================
# Multi-Stage Hardened Production Dockerfile for Quant Alpha Engine
# React 19 Frontend + Python 3.13 Backend + Non-Root Hardened Security (UID 10001)
# ==============================================================================

# --- Stage 1: Build Frontend Single-Page Application (SPA) ---
FROM node:22-alpine AS frontend-builder

WORKDIR /web

# Install build dependencies
COPY web/package*.json ./
RUN npm ci

# Copy web source and compile production bundle
COPY web/ ./
RUN npm run build

# --- Stage 2: Build Python Dependencies & Wheel Cache ---
FROM python:3.13-slim AS python-builder

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

# --- Stage 3: Hardened Production Runtime Container ---
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
RUN mkdir -p /app/data /app/src /app/web/dist && \
    chown -R quant:quant /app

# Install wheels from Python builder stage
COPY --from=python-builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/* && rm -rf /tmp/wheels

# Copy application source and configuration
COPY --chown=quant:quant pyproject.toml .
COPY --chown=quant:quant alembic.ini* .
COPY --chown=quant:quant src/ /app/src/

# Copy compiled frontend SPA from frontend builder stage
COPY --from=frontend-builder --chown=quant:quant /web/dist /app/web/dist

# Install application in link mode for clean module discovery
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
