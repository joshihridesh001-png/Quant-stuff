"""Integration tests for Model Context Protocol (MCP) HTTP RPC endpoint.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoint: POST /api/v1/mcp/rpc
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_mcp_rpc_rejected(client: AsyncClient) -> None:
    """Verify POST /api/v1/mcp/rpc returns 401 when unauthenticated."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    }
    resp = await client.post("/api/v1/mcp/rpc", json=req)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_mcp_rpc_tools_list_authenticated(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/mcp/rpc returns list of tools when authenticated."""
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }
    resp = await client.post("/api/v1/mcp/rpc", json=req, headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 2
    assert "result" in data
    assert "tools" in data["result"]
    assert len(data["result"]["tools"]) == 7


@pytest.mark.asyncio
async def test_mcp_rpc_tool_call_authenticated(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/mcp/rpc executes tool call."""
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "quant_portfolio_telemetry",
            "arguments": {},
        },
    }
    resp = await client.post("/api/v1/mcp/rpc", json=req, headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data
    assert "content" in data["result"]


@pytest.mark.asyncio
async def test_mcp_rpc_emergency_panic_rbac_enforcement(
    client: AsyncClient,
    researcher_jwt_headers: dict[str, str],
    admin_jwt_headers: dict[str, str],
) -> None:
    """Verify non-admin cannot trip panic via MCP, while admin is authorized."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "quant_emergency_panic",
            "arguments": {"reason": "Test Panic"},
        },
    }
    # 1. Researcher role -> Forbidden in tool execution
    res_forbidden = await client.post("/api/v1/mcp/rpc", json=req, headers=researcher_jwt_headers)
    assert res_forbidden.status_code == 200
    res_data = res_forbidden.json()
    assert res_data["result"]["isError"] is True
    assert "Permission denied" in res_data["result"]["content"][0]["text"]

    # 2. Admin role -> Authorized execution
    res_admin = await client.post("/api/v1/mcp/rpc", json=req, headers=admin_jwt_headers)
    assert res_admin.status_code == 200
    admin_data = res_admin.json()
    assert admin_data["result"]["isError"] is False
    assert "PANIC_TRIGGERED" in admin_data["result"]["content"][0]["text"]

    # 3. Disarm kill switch back to ARMED_STANDBY for test state isolation
    reset_req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "quant_kill_switch_reset",
            "arguments": {"admin_token": "DEFAULT_ADMIN_TOKEN"},
        },
    }
    res_reset = await client.post("/api/v1/mcp/rpc", json=reset_req, headers=admin_jwt_headers)
    assert res_reset.status_code == 200
    reset_data = res_reset.json()
    assert reset_data["result"]["isError"] is False
    assert "ARMED_STANDBY" in reset_data["result"]["content"][0]["text"]
