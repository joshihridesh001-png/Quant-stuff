"""Unit and integration tests for Institutional Model Context Protocol (MCP) Client.

Verifies:
    - Protocol handshakes (initialize, version negotiation, capabilities).
    - Dynamic tool discovery (tools/list with schema verification).
    - Tool dispatching and typed helper methods (telemetry, macro regimes, order book, swarm).
    - Pre-trade Bayesian risk simulation gate evaluation.
    - Client-side schema validation and bounds checking (ERR-MCP-005).
    - Tool execution error trapping and isError flags (ERR-MCP-004).
    - Protocol error handling for invalid envelopes and unknown methods (ERR-MCP-002).
    - HTTP Bearer authentication, token acquisition, and auto-renewal on 401 (ERR-MCP-001).
    - Transport failure handling (ERR-MCP-003).
    - Emergency kill switch trip and disarm cycle over MCP.
"""

from __future__ import annotations

import math

import httpx
import pytest

from quant.core.security import create_access_token
from quant.mcp.client import (
    ERR_MCP_AUTH_FAILURE,
    ERR_MCP_PROTOCOL_VIOLATION,
    ERR_MCP_SCHEMA_VALIDATION_FAILURE,
    ERR_MCP_TOOL_EXECUTION_FAILURE,
    ERR_MCP_TRANSPORT_FAILURE,
    MCPAuthFailureException,
    MCPClient,
    MCPProtocolException,
    MCPSchemaValidationException,
    MCPToolExecutionException,
    MCPTransportException,
)


@pytest.mark.asyncio
async def test_mcp_client_stdio_handshake() -> None:
    """Verify stdio transport initialization handshake negotiates protocol version."""
    # Functional Purpose: Confirm protocol negotiation over stdio child process pipes.
    # Explicit Dependency Tracking: MCPClient(transport="stdio").
    # Structural Relationship: First step in desktop agent integration session.
    # Defensive Invariant: Server info name must be 'quant-engine-mcp'.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        init_res = await client.initialize(client_name="test-agent", client_version="1.0.0")
        assert init_res["protocolVersion"] == "2024-11-05"
        assert init_res["serverInfo"]["name"] == "quant-engine-mcp"
        assert "tools" in init_res["capabilities"]
        assert client.is_connected is True


@pytest.mark.asyncio
async def test_mcp_client_stdio_list_tools() -> None:
    """Verify stdio transport discovers all 7 registered institutional quantitative tools."""
    # Functional Purpose: Verify dynamic discovery of all quantitative tool schemas.
    # Explicit Dependency Tracking: list_tools().
    # Structural Relationship: Populates client tool schema cache.
    # Defensive Invariant: Exactly 7 tools registered with valid inputSchema.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools) == 7
        tool_names = {t["name"] for t in tools}
        expected = {
            "quant_portfolio_telemetry",
            "quant_market_orderbook",
            "quant_macro_regimes",
            "quant_evaluate_pre_trade",
            "quant_swarm_status",
            "quant_emergency_panic",
            "quant_kill_switch_reset",
        }
        assert tool_names == expected
        for tool in tools:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_mcp_client_stdio_portfolio_telemetry() -> None:
    """Verify telemetry tool execution and typed helper method."""
    # Functional Purpose: Confirm real-time risk metrics retrieval over stdio.
    # Explicit Dependency Tracking: get_portfolio_telemetry().
    # Structural Relationship: Agent observation phase.
    # Defensive Invariant: NAV and Cash must be strictly positive floats.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        telemetry = await client.get_portfolio_telemetry()
        assert "nav" in telemetry
        assert "cash" in telemetry
        assert "is_kill_switch_active" in telemetry
        assert float(telemetry["nav"]) > 0.0


@pytest.mark.asyncio
async def test_mcp_client_stdio_macro_regimes() -> None:
    """Verify macro regimes tool returns Federal Reserve indicators."""
    # Functional Purpose: Query FRED macro observations.
    # Explicit Dependency Tracking: get_macro_regimes().
    # Structural Relationship: Regime conditioning for trading decisions.
    # Defensive Invariant: Yield spread and effective Fed funds rate present.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        macro = await client.get_macro_regimes()
        assert "t10y2y_yield_spread" in macro
        assert "dff_effective_rate" in macro
        assert "is_yield_curve_inverted" in macro
        assert isinstance(macro["is_yield_curve_inverted"], bool)


@pytest.mark.asyncio
async def test_mcp_client_stdio_market_orderbook() -> None:
    """Verify market order book query returns pricing and volume."""
    # Functional Purpose: Query consolidated NBBO quote for candidate asset.
    # Explicit Dependency Tracking: get_market_orderbook('SPY').
    # Structural Relationship: Pricing reference for order generation.
    # Defensive Invariant: Symbol in response matches query symbol.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        book = await client.get_market_orderbook("SPY")
        assert book["symbol"] == "SPY"
        assert "close" in book or book.get("status") == "no_data_available"


@pytest.mark.asyncio
async def test_mcp_client_stdio_evaluate_pre_trade() -> None:
    """Verify 5-dimensional Bayesian pre-trade gate simulation over MCP."""
    # Functional Purpose: Simulate candidate order evaluation through Bayesian risk gate.
    # Explicit Dependency Tracking: evaluate_pre_trade().
    # Structural Relationship: Pre-execution risk checkpoint.
    # Defensive Invariant: Decision must be PASS or REJECT; latency must be positive.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        decision = await client.evaluate_pre_trade(
            symbol="SPY",
            action="BUY",
            quantity=10.0,
            reference_price=500.0,
            market_spread_bps=2.0,
            order_book_imbalance=0.05,
            macro_yield_spread=0.18,
        )
        assert "allowed" in decision
        assert "decision" in decision
        assert "toxicity_probability" in decision
        assert "checks" in decision
        assert len(decision["checks"]) == 5
        assert decision["allowed"] is True
        assert decision["decision"] == "PASS"


@pytest.mark.asyncio
async def test_mcp_client_stdio_swarm_status() -> None:
    """Verify swarm status tool returns autonomous engine state."""
    # Functional Purpose: Inspect background autonomous trading engine status.
    # Explicit Dependency Tracking: get_swarm_status().
    # Structural Relationship: Agent daemon supervision.
    # Defensive Invariant: Engine state and universe must be present.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()
        status = await client.get_swarm_status()
        assert "state" in status
        assert "universe" in status
        assert isinstance(status["universe"], list)


@pytest.mark.asyncio
async def test_mcp_client_schema_validation_rejections() -> None:
    """Verify client-side schema validation traps invalid arguments with ERR-MCP-005."""
    # Functional Purpose: Prove client-side pre-flight validation prevents corrupt inputs.
    # Explicit Dependency Tracking: _validate_tool_call, MCPSchemaValidationException.
    # Structural Relationship: Defense-in-depth gatekeeper before network transmission.
    # Defensive Invariant: Prohibits non-finite numbers, empty strings, and negative values.
    client = MCPClient(transport="stdio")

    # 1. Empty tool name
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool("", {})
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE

    # 2. Non-finite quantity (NaN)
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool(
            "quant_evaluate_pre_trade",
            {"symbol": "SPY", "action": "BUY", "quantity": float("nan"), "reference_price": 500.0},
        )
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE
    assert "non-finite" in str(exc_info.value)

    # 3. Non-finite reference price (Infinity)
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool(
            "quant_evaluate_pre_trade",
            {"symbol": "SPY", "action": "BUY", "quantity": 10.0, "reference_price": math.inf},
        )
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE

    # 4. Negative quantity
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool(
            "quant_evaluate_pre_trade",
            {"symbol": "SPY", "action": "BUY", "quantity": -5.0, "reference_price": 500.0},
        )
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE

    # 5. Invalid action enum
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool(
            "quant_evaluate_pre_trade",
            {
                "symbol": "SPY",
                "action": "INVALID_ACTION",
                "quantity": 10.0,
                "reference_price": 500.0,
            },
        )
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE

    # 6. Order book imbalance out of [-1.0, 1.0] bounds
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool(
            "quant_evaluate_pre_trade",
            {
                "symbol": "SPY",
                "action": "BUY",
                "quantity": 10.0,
                "reference_price": 500.0,
                "order_book_imbalance": 2.5,
            },
        )
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE

    # 7. Missing required symbol on market orderbook
    with pytest.raises(MCPSchemaValidationException) as exc_info:
        await client.call_tool("quant_market_orderbook", {})
    assert exc_info.value.code == ERR_MCP_SCHEMA_VALIDATION_FAILURE


@pytest.mark.asyncio
async def test_mcp_client_tool_execution_error_handling() -> None:
    """Verify tool execution errors are properly captured with isError and ERR-MCP-004."""
    # Functional Purpose: Verify error payload capture and exception raising on tool failures.
    # Explicit Dependency Tracking: call_tool(raise_on_error=True).
    # Structural Relationship: Error propagation for agent supervisor.
    # Defensive Invariant: Server-side tool failure reflected in isError flag and exception code.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        await client.initialize()

        # Unknown tool name returns isError=True in content
        resp = await client.call_tool("quant_non_existent_tool", {}, raise_on_error=False)
        assert resp.is_error is True
        assert resp["isError"] is True
        assert "Unknown tool" in resp.raw_text

        # With raise_on_error=True, raises MCPToolExecutionException (ERR-MCP-004)
        with pytest.raises(MCPToolExecutionException) as exc_info:
            await client.call_tool("quant_non_existent_tool", {}, raise_on_error=True)
        assert exc_info.value.code == ERR_MCP_TOOL_EXECUTION_FAILURE


@pytest.mark.asyncio
async def test_mcp_client_protocol_violation_error() -> None:
    """Verify protocol errors (-32601 Method not found) raise MCPProtocolException (ERR-MCP-002)."""
    # Functional Purpose: Verify JSON-RPC error translation to MCPProtocolException.
    # Explicit Dependency Tracking: _send_rpc.
    # Structural Relationship: Protocol adherence validation.
    # Defensive Invariant: JSON-RPC errors map to ERR_MCP_PROTOCOL_VIOLATION.
    async with MCPClient(transport="stdio", timeout_seconds=10.0) as client:
        with pytest.raises(MCPProtocolException) as exc_info:
            await client._send_rpc("invalid/method/name", {})
        assert exc_info.value.code == ERR_MCP_PROTOCOL_VIOLATION
        assert "-32601" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_client_http_transport_authenticated(
    client: httpx.AsyncClient,
    admin_jwt_headers: dict[str, str],
) -> None:
    """Verify HTTP JSON-RPC gateway execution with authenticated JWT Bearer token."""
    # Functional Purpose: Validate HTTP transport integration against live FastAPI ASGI app.
    # Explicit Dependency Tracking: MCPClient(transport="http", http_client=client).
    # Structural Relationship: Production HTTP gateway testing.
    # Defensive Invariant: Authenticated HTTP requests execute identically to stdio.
    token = admin_jwt_headers["Authorization"].split(" ")[1]
    mcp_client = MCPClient(
        transport="http",
        base_url="http://testserver",
        token=token,
        http_client=client,
    )

    # 1. Initialize
    init_res = await mcp_client.initialize()
    assert init_res["protocolVersion"] == "2024-11-05"

    # 2. List tools
    tools = await mcp_client.list_tools()
    assert len(tools) == 7

    # 3. Telemetry
    telemetry = await mcp_client.get_portfolio_telemetry()
    assert "nav" in telemetry


@pytest.mark.asyncio
async def test_mcp_client_http_token_acquisition(client: httpx.AsyncClient) -> None:
    """Verify MCPClient.login() acquires JWT access token from /api/v1/auth/token."""
    # Functional Purpose: Validate automatic credential negotiation.
    # Explicit Dependency Tracking: MCPClient.login().
    # Structural Relationship: Initial session setup for HTTP agents.
    # Defensive Invariant: Stores unexpired JWT string on success.
    mcp_client = MCPClient(
        transport="http",
        base_url="http://testserver",
        username="admin",
        password="quant-secret-pass",
        role="ADMIN",
        http_client=client,
    )
    token = await mcp_client.login()
    assert isinstance(token, str)
    assert len(token) > 20
    assert mcp_client.token == token


@pytest.mark.asyncio
async def test_mcp_client_http_auth_failure_invalid_credentials(client: httpx.AsyncClient) -> None:
    """Verify invalid credentials raise MCPAuthFailureException (ERR-MCP-001)."""
    # Functional Purpose: Confirm failed login throws ERR-MCP-001.
    # Explicit Dependency Tracking: login() with bad password.
    # Structural Relationship: Security boundary verification.
    # Defensive Invariant: Rejects bad credentials with HTTP 401.
    mcp_client = MCPClient(
        transport="http",
        base_url="http://testserver",
        username="admin",
        password="wrong-password",
        role="ADMIN",
        http_client=client,
    )
    with pytest.raises(MCPAuthFailureException) as exc_info:
        await mcp_client.login()
    assert exc_info.value.code == ERR_MCP_AUTH_FAILURE


@pytest.mark.asyncio
async def test_mcp_client_http_auto_token_renewal_on_401(client: httpx.AsyncClient) -> None:
    """Verify client automatically negotiates fresh token when gateway returns 401."""
    # Functional Purpose: Test transparent JWT token renewal on 401 Unauthorized.
    # Explicit Dependency Tracking: _send_rpc_http automatic retry loop.
    # Structural Relationship: Guarantees long-running agent loops do not break upon token expiry.
    # Defensive Invariant: Re-authenticates and retries RPC dispatch.
    from datetime import timedelta

    expired_token = create_access_token(
        payload={"sub": "admin", "role": "ADMIN"},
        expires_delta=timedelta(minutes=-30),
    )
    mcp_client = MCPClient(
        transport="http",
        base_url="http://testserver",
        token=expired_token,
        username="admin",
        password="quant-secret-pass",
        role="ADMIN",
        http_client=client,
    )
    # Even though token starts expired, client should catch 401, call login(), and succeed!
    init_res = await mcp_client.initialize()
    assert init_res["protocolVersion"] == "2024-11-05"
    assert mcp_client.token != expired_token


@pytest.mark.asyncio
async def test_mcp_client_emergency_panic_and_reset(
    client: httpx.AsyncClient,
    admin_jwt_headers: dict[str, str],
) -> None:
    """Verify emergency panic kill switch trigger and reset over MCP client."""
    # Functional Purpose: Validate emergency circuit breaker trip and recovery commands.
    # Explicit Dependency Tracking: trigger_emergency_panic, reset_kill_switch.
    # Structural Relationship: Emergency risk controls for institutional operators.
    # Defensive Invariant: Only ADMIN role can disarm back to ARMED_STANDBY.
    token = admin_jwt_headers["Authorization"].split(" ")[1]
    mcp_client = MCPClient(
        transport="http",
        base_url="http://testserver",
        token=token,
        http_client=client,
    )

    # 1. Trigger Emergency Panic
    panic_res = await mcp_client.trigger_emergency_panic(
        reason="Test Automated Circuit Breaker Audit",
        details="Triggered in test_mcp_client.py",
    )
    assert panic_res.get("status") == "PANIC_TRIGGERED"

    # 2. Verify Kill Switch is active in telemetry
    telemetry = await mcp_client.get_portfolio_telemetry()
    assert telemetry.get("is_kill_switch_active") is True

    # 3. Disarm Kill Switch back to Standby
    reset_res = await mcp_client.reset_kill_switch(admin_token="DEFAULT_ADMIN_TOKEN")
    assert reset_res.get("success") is True
    assert reset_res.get("status") == "ARMED_STANDBY"

    # 4. Verify Kill Switch is restored to inactive
    telemetry_post = await mcp_client.get_portfolio_telemetry()
    assert telemetry_post.get("is_kill_switch_active") is False


@pytest.mark.asyncio
async def test_mcp_client_transport_failure_offline_port() -> None:
    """Verify connecting to an unreachable port raises MCPTransportException (ERR-MCP-003)."""
    # Functional Purpose: Verify transport level connection exceptions are mapped to ERR-MCP-003.
    # Explicit Dependency Tracking: MCPClient(transport="http").
    # Structural Relationship: Network partition / dead gateway error handling.
    # Defensive Invariant: Connect error raises MCPTransportException with ERR_MCP_TRANSPORT_FAILURE.
    mcp_client = MCPClient(
        transport="http",
        base_url="http://127.0.0.1:59999",
        timeout_seconds=0.5,
    )
    with pytest.raises(MCPTransportException) as exc_info:
        await mcp_client.initialize()
    assert exc_info.value.code == ERR_MCP_TRANSPORT_FAILURE
