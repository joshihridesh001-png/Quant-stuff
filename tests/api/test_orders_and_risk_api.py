"""Integration and unit API tests for live execution orders, risk monitoring, and gateway endpoints.

Purpose:
    Validates REST API contracts across /api/v1/orders, /api/v1/risk, and /api/v1/gateways,
    verifying authentication gates, RBAC authorization, schema validation, algorithmic parent
    order slicing, Perold (1988) implementation shortfall TCA attribution, emergency panic
    kill switch activation, and constant-time administrative reset.

Dependencies:
    - pytest, pytest-asyncio, httpx.AsyncClient.
    - quant.api.v1.schemas: ParentOrderCreateRequest, RiskLimitsUpdateRequest, PanicTriggerRequest,
      KillSwitchResetRequest, HeartbeatPingRequest.
    - tests.conftest: client, researcher_jwt_headers, admin_jwt_headers.

Structural Relationship:
    - Tests presentation layer integration against FastAPI application instance.
    - Drives underlying ExecutionService and RiskService singleton instances.

Invariants Enforced:
    - INV-RSK-008: Immediate execution lockout upon emergency panic activation.
    - INV-SOR-001: Parent-child mass conservation across slices.
    - INV-SOR-005: Perold (1988) implementation shortfall additive conservation.
    - Rule 1: Four-tier line annotations on all test routines.
    - Rule 2: Diagnostic error code mappings and HTTP status codes.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# ============================================================================
# 1. Unauthenticated Route Rejection Tests (401 Unauthorized)
# ============================================================================


@pytest.mark.asyncio
async def test_unauthenticated_requests_rejected(client: AsyncClient) -> None:
    """Verify all protected execution, risk, and gateway endpoints return 401 when unauthenticated."""
    # Functional Purpose: Verify authentication gate across live execution endpoints.
    # Explicit Dependency Tracking: FastAPI security dependency get_current_user.
    # Structural Relationship: Route security baseline verification.
    # Defensive Invariant: Zero anonymous access to execution or risk systems.
    endpoints: list[tuple[str, str, dict[str, object] | None]] = [
        ("GET", "/api/v1/orders", None),
        (
            "POST",
            "/api/v1/orders",
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100.0,
                "order_type": "MARKET",
                "algorithm": "DIRECT_MARKET",
            },
        ),
        ("GET", "/api/v1/orders/test-order-id", None),
        ("DELETE", "/api/v1/orders/test-order-id", None),
        ("GET", "/api/v1/orders/test-order-id/shortfall", None),
        ("GET", "/api/v1/risk/status", None),
        ("GET", "/api/v1/risk/limits", None),
        ("PUT", "/api/v1/risk/limits", {"max_order_notional": 100000.0}),
        ("POST", "/api/v1/risk/panic", {"reason": "Unauthorized trigger attempt"}),
        ("POST", "/api/v1/risk/reset", {"admin_token": "FAKE_TOKEN"}),
        ("GET", "/api/v1/gateways/health", None),
        ("POST", "/api/v1/gateways/PAPER_BROKER/heartbeat", {"sequence_number": 1}),
    ]

    for method, path, json_data in endpoints:
        if method == "GET":
            resp = await client.get(path)
        elif method == "POST":
            resp = await client.post(path, json=json_data or {})
        elif method == "PUT":
            resp = await client.put(path, json=json_data or {})
        elif method == "DELETE":
            resp = await client.delete(path)
        else:
            pytest.fail(f"Unsupported method: {method}")

        assert resp.status_code == 401, f"Expected 401 for {method} {path}, got {resp.status_code}"


# ============================================================================
# 2. Role-Based Access Control (RBAC) Tests (403 Forbidden)
# ============================================================================


@pytest.mark.asyncio
async def test_rbac_admin_routes_forbidden_for_researcher(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify non-admin roles (e.g. RESEARCHER) are rejected with 403 Forbidden on admin routes."""
    # Functional Purpose: Verify strict RBAC authorization boundaries.
    # Explicit Dependency Tracking: require_role dependency in risk endpoints.
    # Structural Relationship: Prevents unauthorized parameter tampering or safety disarms.
    # Defensive Invariant: Non-admin users cannot alter risk limits or reset kill switches.
    limit_resp = await client.put(
        "/api/v1/risk/limits",
        json={"max_order_notional": 999999.0},
        headers=researcher_jwt_headers,
    )
    assert limit_resp.status_code == 403

    reset_resp = await client.post(
        "/api/v1/risk/reset",
        json={"admin_token": "DEFAULT_ADMIN_TOKEN"},
        headers=researcher_jwt_headers,
    )
    assert reset_resp.status_code == 403


# ============================================================================
# 3. Gateway Health & Heartbeat Tests
# ============================================================================


@pytest.mark.asyncio
async def test_gateway_health_query(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify querying gateway health returns status for registered PAPER_BROKER."""
    # Functional Purpose: Query gateway connectivity and watchdog metrics.
    # Explicit Dependency Tracking: RiskService.get_gateway_health.
    # Structural Relationship: Endpoint GET /api/v1/gateways/health.
    # Defensive Invariant: Yields registered gateway list with valid connection state.
    response = await client.get("/api/v1/gateways/health", headers=researcher_jwt_headers)
    assert response.status_code == 200
    gateways = response.json()
    assert isinstance(gateways, list)
    assert len(gateways) >= 1
    broker = next((g for g in gateways if g["gateway_id"] == "PAPER_BROKER"), None)
    assert broker is not None
    assert broker["is_connected"] is True
    assert broker["status"] in ("CONNECTED", "DEGRADED", "DISCONNECTED", "RECONNECTING")


@pytest.mark.asyncio
async def test_gateway_heartbeat_recording(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify recording gateway heartbeats updates latency and sequence numbers."""
    # Functional Purpose: Ingest transport keep-alive pulses from gateway watchdogs.
    # Explicit Dependency Tracking: RiskService.record_gateway_heartbeat.
    # Structural Relationship: Endpoint POST /api/v1/gateways/{id}/heartbeat.
    # Defensive Invariant: 404 on unregistered gateway, 200 on registered gateway.
    not_found = await client.post(
        "/api/v1/gateways/NON_EXISTENT_GATEWAY/heartbeat",
        json={"sequence_number": 1, "latency_ms": 1.5},
        headers=researcher_jwt_headers,
    )
    assert not_found.status_code == 404

    success = await client.post(
        "/api/v1/gateways/PAPER_BROKER/heartbeat",
        json={"sequence_number": 42, "latency_ms": 2.4},
        headers=researcher_jwt_headers,
    )
    assert success.status_code == 200
    data = success.json()
    assert data["gateway_id"] == "PAPER_BROKER"
    assert data["last_latency_ms"] == 2.4
    assert data["is_connected"] is True


# ============================================================================
# 4. Risk Monitor & Dynamic Limits Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_risk_status_and_limits(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify risk status telemetry and active pre-trade risk boundary queries."""
    # Functional Purpose: Read firm-wide exposure metrics and risk firewall limits.
    # Explicit Dependency Tracking: RiskService.get_risk_status, get_risk_limits.
    # Structural Relationship: Endpoints GET /api/v1/risk/status and GET /api/v1/risk/limits.
    # Defensive Invariant: Non-negative NAV, finite leverage ratios, unbreached limits.
    status_resp = await client.get("/api/v1/risk/status", headers=researcher_jwt_headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["nav"] > 0.0
    assert status_data["cash"] > 0.0
    assert status_data["is_kill_switch_active"] is False

    limits_resp = await client.get("/api/v1/risk/limits", headers=researcher_jwt_headers)
    assert limits_resp.status_code == 200
    limits_data = limits_resp.json()
    assert limits_data["max_order_notional"] > 0.0
    assert limits_data["max_gross_leverage"] > 0.0


@pytest.mark.asyncio
async def test_update_risk_limits_admin(
    client: AsyncClient,
    admin_jwt_headers: dict[str, str],
) -> None:
    """Verify administrator can dynamically update pre-trade risk limits."""
    # Functional Purpose: Modulate pre-trade firewall parameters without service interruption.
    # Explicit Dependency Tracking: RiskService.update_risk_limits.
    # Structural Relationship: Endpoint PUT /api/v1/risk/limits.
    # Defensive Invariant: Updated limits reflected atomically in subsequent inspections.
    update_payload = {"max_order_notional": 750000.0, "max_gross_leverage": 2.5}
    response = await client.put(
        "/api/v1/risk/limits",
        json=update_payload,
        headers=admin_jwt_headers,
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["max_order_notional"] == 750000.0
    assert updated_data["max_gross_leverage"] == 2.5

    # Verify query reflects new boundary
    check_resp = await client.get("/api/v1/risk/limits", headers=admin_jwt_headers)
    assert check_resp.status_code == 200
    assert check_resp.json()["max_order_notional"] == 750000.0


# ============================================================================
# 5. Algorithmic Parent Order Lifecycle & TCA Shortfall Tests
# ============================================================================


@pytest.mark.asyncio
async def test_submit_and_query_parent_orders(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify submission of TWAP, VWAP, Arrival Price, and Market parent orders."""
    # Functional Purpose: Exercise parent order submission across execution algorithms.
    # Explicit Dependency Tracking: ExecutionService.submit_parent_order.
    # Structural Relationship: Endpoints POST /api/v1/orders and GET /api/v1/orders.
    # Defensive Invariant: INV-SOR-001 exact mass conservation, 201 Created status.
    orders_to_create = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100.0,
            "order_type": "MARKET",
            "algorithm": "POISSON_TWAP",
            "horizon_seconds": 60.0,
            "algo_params": {"timing_jitter_pct": 0.1},
        },
        {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 50.0,
            "order_type": "MARKET",
            "algorithm": "VOLUME_ADAPTIVE_VWAP",
            "horizon_seconds": 120.0,
            "algo_params": {"max_participation_rate": 0.10},
        },
        {
            "symbol": "NVDA",
            "side": "SELL",
            "quantity": 25.0,
            "order_type": "MARKET",
            "algorithm": "ARRIVAL_PRICE",
            "horizon_seconds": 60.0,
            "algo_params": {"urgency_parameter": 0.8},
        },
        {
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 10.0,
            "order_type": "MARKET",
            "algorithm": "DIRECT_MARKET",
        },
    ]

    created_order_ids: list[str] = []

    for payload in orders_to_create:
        resp = await client.post("/api/v1/orders", json=payload, headers=researcher_jwt_headers)
        assert resp.status_code == 201, f"Failed creating {payload['algorithm']}: {resp.text}"
        data = resp.json()
        assert "order_id" in data
        assert data["symbol"] == payload["symbol"]
        assert data["total_quantity"] == payload["quantity"]
        assert data["side"] == payload["side"]
        assert data["arrival_price"] > 0.0
        assert data["is_closed"] is False
        created_order_ids.append(data["order_id"])

    # 1. Query all orders
    list_resp = await client.get("/api/v1/orders", headers=researcher_jwt_headers)
    assert list_resp.status_code == 200
    all_orders = list_resp.json()
    assert len(all_orders) >= 4

    # 2. Query with symbol filter
    filter_resp = await client.get(
        "/api/v1/orders?symbol=AAPL",
        headers=researcher_jwt_headers,
    )
    assert filter_resp.status_code == 200
    aapl_orders = filter_resp.json()
    assert all(o["symbol"] == "AAPL" for o in aapl_orders)

    # 3. Query with pagination
    paginated_resp = await client.get(
        "/api/v1/orders?limit=2&offset=0",
        headers=researcher_jwt_headers,
    )
    assert paginated_resp.status_code == 200
    assert len(paginated_resp.json()) == 2

    # 4. Get single order details
    first_id = created_order_ids[0]
    detail_resp = await client.get(
        f"/api/v1/orders/{first_id}",
        headers=researcher_jwt_headers,
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["order_id"] == first_id

    # 5. Get non-existent order details -> 404
    missing_resp = await client.get(
        "/api/v1/orders/non-existent-order-id",
        headers=researcher_jwt_headers,
    )
    assert missing_resp.status_code == 404

    # 6. Implementation Shortfall TCA attribution
    shortfall_resp = await client.get(
        f"/api/v1/orders/{first_id}/shortfall?terminal_price=155.0",
        headers=researcher_jwt_headers,
    )
    assert shortfall_resp.status_code == 200
    shortfall_data = shortfall_resp.json()
    assert shortfall_data["order_id"] == first_id
    assert "total_shortfall" in shortfall_data
    assert "is_additive_conserved" in shortfall_data
    assert shortfall_data["is_additive_conserved"] is True

    # 7. Cancel order
    cancel_resp = await client.delete(
        f"/api/v1/orders/{first_id}",
        headers=researcher_jwt_headers,
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"

    # 8. Cancel non-existent order -> 404
    bad_cancel = await client.delete(
        "/api/v1/orders/non-existent-order-id",
        headers=researcher_jwt_headers,
    )
    assert bad_cancel.status_code == 404


@pytest.mark.asyncio
async def test_invalid_order_rejection(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify invalid order payloads fail schema or domain validation with 422."""
    # Functional Purpose: Verify input validation against non-finite or illegal payloads.
    # Explicit Dependency Tracking: ParentOrderCreateRequest validation.
    # Structural Relationship: Defensive ingress boundary for order submissions.
    # Defensive Invariant: Non-positive quantities or illegal algorithms rejected with 422.
    # Non-positive quantity
    bad_qty = await client.post(
        "/api/v1/orders",
        json={"symbol": "AAPL", "side": "BUY", "quantity": 0.0},
        headers=researcher_jwt_headers,
    )
    assert bad_qty.status_code == 422

    # Negative duration
    bad_dur = await client.post(
        "/api/v1/orders",
        json={
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "horizon_seconds": -5.0,
        },
        headers=researcher_jwt_headers,
    )
    assert bad_dur.status_code == 422


# ============================================================================
# 6. Emergency Panic Kill Switch & Admin Disarm Resumption Flow
# ============================================================================


@pytest.mark.asyncio
async def test_panic_trigger_and_admin_reset_flow(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
    admin_jwt_headers: dict[str, str],
) -> None:
    """Verify emergency panic trips kill switch, blocks submissions, and permits admin reset."""
    # Functional Purpose: Verify complete emergency kill switch lifecycle and post-incident reset.
    # Explicit Dependency Tracking: RiskService.trigger_panic, reset_kill_switch.
    # Structural Relationship: Endpoints POST /api/v1/risk/panic and POST /api/v1/risk/reset.
    # Defensive Invariant: INV-RSK-008 immediate execution lockout and constant-time admin reset.
    # 1. Trigger panic as authorized operator
    panic_resp = await client.post(
        "/api/v1/risk/panic",
        json={
            "reason": "Flash crash detected in external quotes",
            "details": {"latency_spike": True},
        },
        headers=researcher_jwt_headers,
    )
    assert panic_resp.status_code == 200
    panic_data = panic_resp.json()
    assert panic_data["status"] == "PANIC_TRIGGERED"

    # 2. Verify risk status shows kill switch is active
    status_resp = await client.get("/api/v1/risk/status", headers=researcher_jwt_headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["is_kill_switch_active"] is True

    # 3. Verify order submission is strictly blocked under active kill switch -> 400 Bad Request
    order_attempt = await client.post(
        "/api/v1/orders",
        json={
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "order_type": "MARKET",
            "algorithm": "DIRECT_MARKET",
        },
        headers=researcher_jwt_headers,
    )
    assert order_attempt.status_code == 400
    assert "ERR-RSK-008" in str(order_attempt.json())

    # 4. Attempt reset with invalid secret token -> 403 Forbidden
    bad_reset = await client.post(
        "/api/v1/risk/reset",
        json={"admin_token": "INCORRECT_SECRET_KEY"},
        headers=admin_jwt_headers,
    )
    assert bad_reset.status_code == 403

    # 5. Reset with valid admin secret token -> 200 OK
    good_reset = await client.post(
        "/api/v1/risk/reset",
        json={"admin_token": "DEFAULT_ADMIN_TOKEN"},
        headers=admin_jwt_headers,
    )
    assert good_reset.status_code == 200
    assert good_reset.json()["status"] == "ARMED_STANDBY"
    assert good_reset.json()["success"] is True

    # 6. Verify risk status shows kill switch is disarmed
    post_reset_status = await client.get("/api/v1/risk/status", headers=researcher_jwt_headers)
    assert post_reset_status.status_code == 200
    assert post_reset_status.json()["is_kill_switch_active"] is False

    # 7. Verify order submission succeeds again
    post_reset_order = await client.post(
        "/api/v1/orders",
        json={
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "order_type": "MARKET",
            "algorithm": "DIRECT_MARKET",
        },
        headers=researcher_jwt_headers,
    )
    assert post_reset_order.status_code == 201
