"""Integration tests for Autonomous Live Trading Swarm REST endpoints.

Purpose:
    Validates REST API contracts across /api/v1/autonomous routes, verifying authentication
    gates, status query responses, single-step execution, and background loop lifecycle controls.

Dependencies:
    - pytest, pytest-asyncio, httpx.AsyncClient.
    - tests.conftest: client, researcher_jwt_headers.

Structural Relationship:
    - Tests presentation layer integration against FastAPI application instance.
    - Drives underlying AutonomousTradingEngine daemon.

Invariants Enforced:
    - Rule 1: Four-tier line annotations on every test routine.
    - Rule 2: HTTP status code verification and schema compliance.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_autonomous_endpoints_rejected(client: AsyncClient) -> None:
    """Verify all autonomous trading endpoints return 401 when unauthenticated.

    Args:
        client: Test HTTP client.
    """
    # Functional Purpose: Verify authentication gate across autonomous trading endpoints.
    # Explicit Dependency Tracking: FastAPI security dependency get_current_user.
    # Structural Relationship: Presentation security baseline verification.
    # Defensive Invariant: Zero anonymous access to autonomous trading controls.
    routes = [
        ("GET", "/api/v1/autonomous/status"),
        ("POST", "/api/v1/autonomous/start"),
        ("POST", "/api/v1/autonomous/stop"),
        ("POST", "/api/v1/autonomous/pause"),
        ("POST", "/api/v1/autonomous/resume"),
        ("POST", "/api/v1/autonomous/step"),
    ]
    for method, path in routes:
        resp = await client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


@pytest.mark.asyncio
async def test_autonomous_status_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify GET /api/v1/autonomous/status returns operational status and universe.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    # Functional Purpose: Verify status endpoint returns valid telemetry.
    # Explicit Dependency Tracking: GET /api/v1/autonomous/status.
    # Structural Relationship: Polled by trading terminal HUD.
    # Defensive Invariant: Status fields must conform to AutonomousStatusDTO.
    resp = await client.get("/api/v1/autonomous/status", headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "iteration" in data
    assert "universe" in data
    assert isinstance(data["universe"], list)
    assert len(data["universe"]) > 0


@pytest.mark.asyncio
async def test_autonomous_step_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/autonomous/step executes a discrete rebalancing cycle.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    # Functional Purpose: Validate discrete rebalance cycle execution via REST.
    # Explicit Dependency Tracking: POST /api/v1/autonomous/step.
    # Structural Relationship: Single-step testing and manual rebalance trigger.
    # Defensive Invariant: Advances iteration and returns complete AutonomousStepReportDTO.
    resp = await client.post("/api/v1/autonomous/step", headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["iteration"] >= 1
    assert "timestamp_ns" in data
    assert "universe" in data
    assert "target_allocations" in data
    assert "current_positions" in data
    assert "orders_dispatched" in data
    assert data["duration_ms"] >= 0.0
    assert data["haircut"] >= 0.0
    assert isinstance(data["is_kill_switch_active"], bool)


@pytest.mark.asyncio
async def test_autonomous_lifecycle_endpoints(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify start, pause, resume, and stop lifecycle transitions via REST API.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    # Functional Purpose: Test operational lifecycle transitions via REST routes.
    # Explicit Dependency Tracking: POST /start, /pause, /resume, /stop.
    # Structural Relationship: Presentation interface for HUD engine toggle buttons.
    # Defensive Invariant: Verifies state machine transitions across each command.
    # 1. Start
    start_resp = await client.post("/api/v1/autonomous/start", headers=researcher_jwt_headers)
    assert start_resp.status_code == 200
    assert start_resp.json()["state"] == "RUNNING"

    # 2. Pause
    pause_resp = await client.post("/api/v1/autonomous/pause", headers=researcher_jwt_headers)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["state"] == "PAUSED"

    # 3. Resume
    resume_resp = await client.post("/api/v1/autonomous/resume", headers=researcher_jwt_headers)
    assert resume_resp.status_code == 200
    assert resume_resp.json()["state"] == "RUNNING"

    # 4. Stop
    stop_resp = await client.post("/api/v1/autonomous/stop", headers=researcher_jwt_headers)
    assert stop_resp.status_code == 200
    assert stop_resp.json()["state"] == "STOPPED"
