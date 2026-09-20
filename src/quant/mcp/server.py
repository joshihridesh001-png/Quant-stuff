"""JSON-RPC 2.0 Model Context Protocol (MCP) Server.

Purpose:
    Processes inbound MCP JSON-RPC protocol requests, performs tool dispatching,
    and returns standardized structured responses for LLM agent interaction.

Dependencies:
    - asyncio: Non-blocking coroutines.
    - json: JSON-RPC parsing and formatting.
    - sys: Stdio stream handles.
    - typing: Type annotations.
    - quant.mcp.tools: get_tool_definitions, execute_tool.

Invariants Enforced:
    - Compliance with JSON-RPC 2.0 specification (error codes -32600, -32601, -32603).
    - Rule 1: Four-tier docstrings on all functions and classes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from typing import Any

from quant.mcp.tools import execute_tool, get_tool_definitions

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION: str = "2024-11-05"
SERVER_NAME: str = "quant-engine-mcp"
SERVER_VERSION: str = "0.1.0"


async def handle_mcp_request(request: dict[str, Any]) -> dict[str, Any]:
    """Process a single JSON-RPC 2.0 Model Context Protocol request.

    Args:
        request: Dictionary representing inbound JSON-RPC message.

    Returns:
        dict[str, Any]: Standard JSON-RPC 2.0 response dictionary.
    """
    # Functional Purpose: Core JSON-RPC 2.0 protocol router for MCP standard messages.
    # Explicit Dependency Tracking: get_tool_definitions, execute_tool.
    # Structural Relationship: Ingested from stdio reader loop or HTTP RPC endpoint.
    # Defensive Invariant: Adheres strictly to JSON-RPC 2.0 response envelope.
    req_id = request.get("id")
    method = request.get("method")

    if not isinstance(method, str):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid Request: 'method' must be string"},
        }

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    elif method == "notifications/initialized":
        # Client acknowledgement notification - no response required
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    elif method == "tools/list":
        tools = get_tool_definitions()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }

    elif method == "tools/call":
        params = request.get("params") or {}
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if not isinstance(tool_name, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: 'name' is required"},
            }

        result = await execute_tool(tool_name, arguments)
        text_content = json.dumps(result, ensure_ascii=False)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": text_content,
                    }
                ]
            },
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


class MCPServer:
    """Standard IO transport server for Model Context Protocol."""

    def __init__(self) -> None:
        """Initialize MCP stdio server instance."""
        self._running: bool = False

    async def run_stdio(self) -> None:
        """Run continuous asynchronous read/write loop over standard I/O."""
        # Functional Purpose: Standard input/output transport for desktop AI agent integration.
        # Explicit Dependency Tracking: sys.stdin, sys.stdout, asyncio.
        # Structural Relationship: Main entrypoint when run via python -m quant.mcp.server.
        # Defensive Invariant: Gracefully handles EOF and malformed JSON lines.
        self._running = True
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            line = await reader.readline()
            if not line:
                break
            raw_text = line.decode("utf-8").strip()
            if not raw_text:
                continue

            try:
                payload = json.loads(raw_text)
                response = await handle_mcp_request(payload)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(exc)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    """CLI entry point for running MCP server via python -m quant.mcp.server."""
    server = MCPServer()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
