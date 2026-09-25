"""Model Context Protocol (MCP) HTTP JSON-RPC 2.0 gateway endpoint.

Purpose:
    Allows web-based AI agents, Copilots, and HTTP clients to invoke MCP tools
    over standard HTTPS POST /api/v1/mcp/rpc with authentication.

Dependencies:
    - FastAPI APIRouter, Depends, Body.
    - quant.api.dependencies: get_current_user.
    - quant.mcp.server: handle_mcp_request.

Structural Relationship:
    - Mounted under /api/v1/mcp in quant.main.
    - Standard bridge between MCP JSON-RPC protocol and FastAPI web services.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends

from quant.api.dependencies import get_current_user
from quant.mcp.server import handle_mcp_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["Model Context Protocol (MCP)"])


@router.post(
    "/rpc",
    summary="Execute a JSON-RPC 2.0 Model Context Protocol request",
)
async def mcp_rpc_endpoint(
    request: dict[str, Any] = Body(..., description="Standard JSON-RPC 2.0 request payload"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Process an inbound MCP JSON-RPC protocol message and return protocol response.

    Args:
        request: Standard JSON-RPC 2.0 dictionary.
        user: Authenticated user claims context.

    Returns:
        dict[str, Any]: Standard JSON-RPC 2.0 response payload.
    """
    # Functional Purpose: Provide secure, authenticated HTTP gateway for MCP agent tool calls.
    # Explicit Dependency Tracking: handle_mcp_request.
    # Structural Relationship: Remote agent gateway endpoint.
    # Defensive Invariant: Adheres to JSON-RPC 2.0 specification envelope with RBAC enforcement.
    user_role = str(user.get("role", "GUEST"))
    return await handle_mcp_request(request, user_role=user_role)
