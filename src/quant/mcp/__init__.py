"""Institutional Model Context Protocol (MCP) Architecture for AI Agents.

Purpose:
    Exposes standardized JSON-RPC 2.0 Model Context Protocol tools and clients allowing
    autonomous AI assistants (Antigravity, Claude, Cursor) to inspect portfolio state,
    query consolidated order books, simulate Bayesian pre-trade decisions, and trigger
    emergency circuit breakers.
"""

from quant.mcp.client import (
    ERR_MCP_AUTH_FAILURE,
    ERR_MCP_PROTOCOL_VIOLATION,
    ERR_MCP_SCHEMA_VALIDATION_FAILURE,
    ERR_MCP_TOOL_EXECUTION_FAILURE,
    ERR_MCP_TRANSPORT_FAILURE,
    MCPAuthFailureException,
    MCPBaseException,
    MCPClient,
    MCPClientConfig,
    MCPProtocolException,
    MCPSchemaValidationException,
    MCPToolCallResponse,
    MCPToolExecutionException,
)
from quant.mcp.server import MCPServer, handle_mcp_request
from quant.mcp.tools import get_tool_definitions

__all__ = [
    "ERR_MCP_AUTH_FAILURE",
    "ERR_MCP_PROTOCOL_VIOLATION",
    "ERR_MCP_SCHEMA_VALIDATION_FAILURE",
    "ERR_MCP_TOOL_EXECUTION_FAILURE",
    "ERR_MCP_TRANSPORT_FAILURE",
    "MCPAuthFailureException",
    "MCPBaseException",
    "MCPClient",
    "MCPClientConfig",
    "MCPProtocolException",
    "MCPSchemaValidationException",
    "MCPServer",
    "MCPToolCallResponse",
    "MCPToolExecutionException",
    "get_tool_definitions",
    "handle_mcp_request",
]
