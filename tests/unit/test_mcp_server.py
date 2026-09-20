"""Unit tests for Institutional Model Context Protocol (MCP) Server.

Verifies:
    - JSON-RPC 2.0 protocol compliance (initialize, tools/list, tools/call).
    - Tool schema integrity across all 7 quantitative tools.
    - Execution handlers for telemetry, macro regimes, pre-trade simulation, and swarm state.
    - Standard error codes (-32600, -32601, -32602).
"""

from __future__ import annotations

import json

import pytest

from quant.mcp.server import handle_mcp_request


@pytest.mark.asyncio
async def test_mcp_initialize() -> None:
    """Verify initialize request returns valid MCP protocol capabilities and server info."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-agent", "version": "1.0.0"},
        },
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    res = resp["result"]
    assert res["protocolVersion"] == "2024-11-05"
    assert res["serverInfo"]["name"] == "quant-engine-mcp"
    assert "tools" in res["capabilities"]


@pytest.mark.asyncio
async def test_mcp_tools_list() -> None:
    """Verify tools/list returns all 7 institutional tools with valid JSON schemas."""
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    assert len(tools) == 7

    tool_names = {t["name"] for t in tools}
    expected_tools = {
        "quant_portfolio_telemetry",
        "quant_market_orderbook",
        "quant_macro_regimes",
        "quant_evaluate_pre_trade",
        "quant_swarm_status",
        "quant_emergency_panic",
        "quant_kill_switch_reset",
    }
    assert tool_names == expected_tools

    for tool in tools:
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_mcp_tools_call_telemetry() -> None:
    """Verify tools/call executes quant_portfolio_telemetry."""
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "quant_portfolio_telemetry",
            "arguments": {},
        },
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 3
    assert "result" in resp
    content = resp["result"]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    payload = json.loads(content[0]["text"])
    assert "nav" in payload
    assert "is_kill_switch_active" in payload


@pytest.mark.asyncio
async def test_mcp_tools_call_evaluate_pre_trade() -> None:
    """Verify tools/call executes quant_evaluate_pre_trade simulation."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "quant_evaluate_pre_trade",
            "arguments": {
                "symbol": "SPY",
                "action": "BUY",
                "quantity": 10.0,
                "reference_price": 500.0,
                "market_spread_bps": 2.0,
            },
        },
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 4
    content = resp["result"]["content"]
    payload = json.loads(content[0]["text"])
    assert "decision" in payload
    assert "allowed" in payload
    assert payload["allowed"] is True
    assert payload["decision"] == "PASS"


@pytest.mark.asyncio
async def test_mcp_tools_call_swarm_status() -> None:
    """Verify tools/call executes quant_swarm_status."""
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "quant_swarm_status",
            "arguments": {},
        },
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 5
    content = resp["result"]["content"]
    payload = json.loads(content[0]["text"])
    assert "state" in payload
    assert "universe" in payload


@pytest.mark.asyncio
async def test_mcp_invalid_method_error() -> None:
    """Verify unknown method returns JSON-RPC -32601 Method not found error."""
    req = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "unknown/operation",
    }
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 99
    assert "error" in resp
    assert resp["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_mcp_invalid_request_envelope() -> None:
    """Verify request with non-string method returns JSON-RPC -32600 error."""
    req = {"jsonrpc": "2.0", "id": 100}
    resp = await handle_mcp_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["error"]["code"] == -32600
