"""Institutional Model Context Protocol (MCP) Server for External AI Agents.

Purpose:
    Exposes standardized JSON-RPC 2.0 Model Context Protocol tools allowing autonomous
    AI assistants (Antigravity, Claude, Gemini, Cursor) to inspect portfolio state,
    query consolidated order books, simulate Bayesian pre-trade decisions, and trigger
    emergency circuit breakers.
"""

from quant.mcp.server import MCPServer, handle_mcp_request
from quant.mcp.tools import get_tool_definitions

__all__ = [
    "MCPServer",
    "get_tool_definitions",
    "handle_mcp_request",
]
