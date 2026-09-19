"""Integration tests for WebSocket streaming endpoints.

Purpose:
    Validates full-duplex WebSocket channels across /api/v1/ws/executions and /api/v1/ws/risk,
    verifying initial state snapshot delivery, bi-directional PING/PONG heartbeats, reactive
    order submission/cancellation broadcasts, and emergency kill switch event broadcasting.

Dependencies:
    - pytest, starlette.testclient.TestClient.
    - quant.api.dependencies: get_execution_service, get_risk_service.
    - quant.api.v1.schemas: ParentOrderCreateRequest.
    - quant.main: app.

Structural Relationship:
    - Validates presentation layer WebSocket transport against live application state.
    - Verifies real-time event pipeline feeding frontend trading terminal dashboard.

Invariants Enforced:
    - Immediate snapshot delivery upon handshake.
    - Bi-directional PING/PONG response.
    - Reactive broadcast on order lifecycle and emergency panic triggers.
    - Rule 1: Four-tier line annotations on all test routines.
    - Rule 2: Structured message envelope verification.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from quant.api.dependencies import get_execution_service, get_risk_service
from quant.api.v1.schemas import ParentOrderCreateRequest
from quant.main import app


def test_ws_executions_connection_snapshot_and_ping() -> None:
    """Verify connecting to /api/v1/ws/executions yields snapshot and handles PING/PONG."""
    # Functional Purpose: Verify execution WebSocket connection handshake and protocol.
    # Explicit Dependency Tracking: /api/v1/ws/executions endpoint.
    # Structural Relationship: Primary live order stream contract.
    # Defensive Invariant: Immediate SNAPSHOT delivery on handshake; PING receives PONG.
    with TestClient(app) as client, client.websocket_connect("/api/v1/ws/executions") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "SNAPSHOT"
        assert "orders" in snapshot
        assert isinstance(snapshot["orders"], list)

        # Test PING/PONG
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong["type"] == "PONG"
        assert "timestamp_ns" in pong


def test_ws_executions_live_broadcast() -> None:
    """Verify parent order submission and cancellation broadcast events over /api/v1/ws/executions."""
    # Functional Purpose: Verify reactive event delivery on order lifecycle changes.
    # Explicit Dependency Tracking: ExecutionService order listener.
    # Structural Relationship: Live order blotter update pipeline.
    # Defensive Invariant: Submissions and cancellations emitted as ORDER_UPDATE frames.
    service = get_execution_service()

    with TestClient(app) as client, client.websocket_connect("/api/v1/ws/executions") as ws:
        # 1. Read initial snapshot
        snapshot = ws.receive_json()
        assert snapshot["type"] == "SNAPSHOT"

        # 2. Submit parent order
        req = ParentOrderCreateRequest(
            symbol="NVDA",
            side="BUY",
            quantity=50.0,
            order_type="MARKET",
            algorithm="DIRECT_MARKET",
        )
        created_order = service.submit_parent_order(req)

        # 3. Verify ORDER_UPDATE CREATED received over WebSocket
        msg = ws.receive_json()
        assert msg["type"] == "ORDER_UPDATE"
        assert msg["event"] == "CREATED"
        assert msg["order"]["order_id"] == created_order.parent_id
        assert msg["order"]["symbol"] == "NVDA"

        # 4. Cancel order and verify ORDER_UPDATE CANCELLED received
        service.cancel_order(created_order.parent_id)
        cancel_msg = ws.receive_json()
        assert cancel_msg["type"] == "ORDER_UPDATE"
        assert cancel_msg["event"] == "CANCELLED"
        assert cancel_msg["order"]["order_id"] == created_order.parent_id


def test_ws_risk_connection_snapshot_and_ping() -> None:
    """Verify connecting to /api/v1/ws/risk yields snapshot and handles PING/PONG."""
    # Functional Purpose: Verify risk WebSocket connection handshake and telemetry snapshot.
    # Explicit Dependency Tracking: /api/v1/ws/risk endpoint.
    # Structural Relationship: Real-time risk HUD and gateway watchdog pipeline.
    # Defensive Invariant: Immediate SNAPSHOT with valid NAV and gateway list; PING receives PONG.
    with TestClient(app) as client, client.websocket_connect("/api/v1/ws/risk") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "SNAPSHOT"
        assert "risk" in snapshot
        assert snapshot["risk"]["nav"] > 0.0
        assert "gateways" in snapshot
        assert isinstance(snapshot["gateways"], list)

        # Test PING/PONG
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong["type"] == "PONG"
        assert "timestamp_ns" in pong


def test_ws_risk_panic_and_reset_broadcast() -> None:
    """Verify emergency panic and kill switch reset broadcast events over /api/v1/ws/risk."""
    # Functional Purpose: Verify immediate dissemination of kill switch events.
    # Explicit Dependency Tracking: RiskService risk listener.
    # Structural Relationship: Hot circuit breaker alert broadcast for terminal UI.
    # Defensive Invariant: Panic and reset transitions broadcast KILL_SWITCH_EVENT frames immediately.
    service = get_risk_service()

    with TestClient(app) as client, client.websocket_connect("/api/v1/ws/risk") as ws:
        # 1. Read initial snapshot
        snapshot = ws.receive_json()
        assert snapshot["type"] == "SNAPSHOT"

        # 2. Trigger panic via HTTP or service
        resp = client.post(
            "/api/v1/risk/panic",
            json={"reason": "WebSocket test panic trigger", "details": {"source": "unit_test"}},
            headers={"Authorization": "Bearer TEST"},
        )
        # Note: If unauthenticated, triggers directly via service to test broadcast
        if resp.status_code != 200:
            import asyncio

            asyncio.run(service.trigger_panic("WebSocket test panic trigger"))

        # 3. Verify KILL_SWITCH_EVENT frame received over WebSocket
        panic_frame = ws.receive_json()
        assert panic_frame["type"] == "KILL_SWITCH_EVENT"
        assert panic_frame["status"] == "PANIC_TRIGGERED"

        # 4. Reset kill switch
        reset_ok = service.reset_kill_switch("DEFAULT_ADMIN_TOKEN")
        assert reset_ok is True

        # 5. Verify reset event frame received
        reset_frame = ws.receive_json()
        assert reset_frame["type"] == "KILL_SWITCH_EVENT"
        assert reset_frame["status"] == "ARMED_STANDBY"
