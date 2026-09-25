"""Model Context Protocol (MCP) Client for Institutional AI Agents.

Purpose:
    Provides an institutional-grade, type-safe client for the Model Context Protocol (MCP 2024-11-05),
    supporting both HTTP JSON-RPC gateway transport (POST /api/v1/mcp/rpc) with automatic JWT
    lifecycle management and local standard I/O (stdio) transport. Enables autonomous AI agents
    (Antigravity, Claude, Cursor) to discover registered quantitative tools, inspect telemetry,
    and simulate pre-trade decisions through multi-dimensional risk gates before execution.

Dependencies:
    - asyncio: Asynchronous event loop and subprocess pipe management.
    - json: JSON-RPC 2.0 serialization and deserialization.
    - math: Numerical finiteness verification (math.isfinite).
    - sys: Python interpreter executable resolution.
    - typing: Static type definitions and protocol declarations.
    - httpx: Asynchronous HTTP transport and connection pooling.

Invariants Enforced:
    - INV-MCP-001 (Schema Validation): Client-side pre-flight validation rejects malformed,
      missing, or non-finite arguments prior to network transmission.
    - INV-MCP-002 (Cryptographic Authentication): All HTTP tool invocations require valid,
      unexpired JWT bearer tokens with automatic renewal on 401 Unauthorized.
    - INV-MCP-003 (Deterministic Diagnostics): Standardized fault vectors (ERR-MCP-001 to ERR-MCP-005).
    - Rule 1: Four-tier docstrings and line-by-line annotations across all components.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import httpx

logger = logging.getLogger(__name__)

# Protocol Constants
MCP_PROTOCOL_VERSION: Final[str] = "2024-11-05"
DEFAULT_CLIENT_NAME: Final[str] = "quant-institutional-mcp-client"
DEFAULT_CLIENT_VERSION: Final[str] = "0.1.0"

# Diagnostic Fault Vector IDs (Rule 2)
ERR_MCP_AUTH_FAILURE: Final[str] = "ERR-MCP-001"
ERR_MCP_PROTOCOL_VIOLATION: Final[str] = "ERR-MCP-002"
ERR_MCP_TRANSPORT_FAILURE: Final[str] = "ERR-MCP-003"
ERR_MCP_TOOL_EXECUTION_FAILURE: Final[str] = "ERR-MCP-004"
ERR_MCP_SCHEMA_VALIDATION_FAILURE: Final[str] = "ERR-MCP-005"


class MCPBaseException(Exception):
    """Base exception for Model Context Protocol client operations."""

    def __init__(self, message: str, code: str) -> None:
        """Initialize MCP base exception with diagnostic fault vector code.

        Args:
            message: Descriptive etiology of the failure.
            code: Unique hierarchical diagnostic code (e.g. ERR-MCP-001).
        """
        # Functional Purpose: Standardize error representation across the MCP client lifecycle.
        # Explicit Dependency Tracking: Built-in Exception class.
        # Structural Relationship: Root class for all domain-specific MCP exceptions.
        # Defensive Invariant: Message and diagnostic code must be non-empty strings.
        super().__init__(f"[{code}] {message}")
        self.message: str = message
        self.code: str = code


class MCPAuthFailureException(MCPBaseException):
    """Raised when authentication fails or JWT renewal is rejected (ERR-MCP-001)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ERR_MCP_AUTH_FAILURE)


class MCPProtocolException(MCPBaseException):
    """Raised when JSON-RPC 2.0 or MCP envelope invariants are violated (ERR-MCP-002)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ERR_MCP_PROTOCOL_VIOLATION)


class MCPTransportException(MCPBaseException):
    """Raised when communication transport drops, hangs, or fails (ERR-MCP-003)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ERR_MCP_TRANSPORT_FAILURE)


class MCPToolExecutionException(MCPBaseException):
    """Raised when server tool execution reports failure or isError is true (ERR-MCP-004)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ERR_MCP_TOOL_EXECUTION_FAILURE)


class MCPSchemaValidationException(MCPBaseException):
    """Raised when tool arguments fail client-side schema verification (ERR-MCP-005)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ERR_MCP_SCHEMA_VALIDATION_FAILURE)


class MCPToolCallResponse(dict[str, Any]):
    """Structured response container for MCP tool execution results.

    Inherits from dict[str, Any] to guarantee backward compatibility with standard
    JSON-RPC result dicts while providing strongly-typed convenience accessors.
    """

    def __init__(
        self,
        content: list[dict[str, Any]],
        is_error: bool,
        data: Any = None,
        raw_text: str = "",
    ) -> None:
        """Initialize MCP tool call response envelope.

        Args:
            content: List of MCP content items (type/text).
            is_error: Boolean flag indicating if tool failed.
            data: Deserialized JSON data dictionary or scalar.
            raw_text: Raw unparsed text string from primary content item.
        """
        # Functional Purpose: Encapsulate tool outputs with dual dict and attribute access.
        # Explicit Dependency Tracking: Built-in dict.
        # Structural Relationship: Returned by call_tool() and helper methods.
        # Defensive Invariant: content must be a list; is_error must be a boolean.
        super().__init__(
            content=content,
            isError=is_error,
            data=data,
            raw_text=raw_text,
        )
        self._content: list[dict[str, Any]] = content
        self._is_error: bool = is_error
        self._data: Any = data
        self._raw_text: str = raw_text

    @property
    def content(self) -> list[dict[str, Any]]:
        """Return raw MCP content items list."""
        return self._content

    @property
    def is_error(self) -> bool:
        """Return True if tool execution reported an error condition."""
        return self._is_error

    @property
    def data(self) -> Any:
        """Return deserialized tool payload object."""
        return self._data

    @property
    def raw_text(self) -> str:
        """Return raw textual content payload."""
        return self._raw_text


@dataclass(frozen=True, slots=True)
class MCPClientConfig:
    """Configuration parameters for MCPClient transport and authentication."""

    base_url: str = "http://127.0.0.1:8000"
    rpc_endpoint: str = "/api/v1/mcp/rpc"
    auth_endpoint: str = "/api/v1/auth/token"
    transport: Literal["http", "stdio"] = "http"
    username: str | None = None
    password: str | None = None
    role: str = "RESEARCHER"
    token: str | None = None
    timeout_seconds: float = 15.0
    server_command: Sequence[str] | None = None


class MCPClient:
    """Institutional Client for Model Context Protocol (MCP 2024-11-05).

    Supports dual transport modes:
    1. HTTP JSON-RPC Gateway (`transport="http"`): Interacts with POST `/api/v1/mcp/rpc`
       with automatic JWT token acquisition and renewal on 401 Unauthorized.
    2. Stdio Subprocess (`transport="stdio"`): Interacts with `python -m quant.mcp.server`
       via asynchronous standard input/output pipes.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        rpc_endpoint: str = "/api/v1/mcp/rpc",
        auth_endpoint: str = "/api/v1/auth/token",
        transport: Literal["http", "stdio"] = "http",
        username: str | None = None,
        password: str | None = None,
        role: str = "RESEARCHER",
        token: str | None = None,
        timeout_seconds: float = 15.0,
        server_command: Sequence[str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize MCP client instance.

        Args:
            base_url: Base HTTP URL for the quant API server.
            rpc_endpoint: Relative path to JSON-RPC 2.0 MCP endpoint.
            auth_endpoint: Relative path to token issuance endpoint.
            transport: Communication transport mode ('http' or 'stdio').
            username: Username for JWT authentication.
            password: Password for JWT authentication.
            role: Requested RBAC role ('RESEARCHER' or 'ADMIN').
            token: Pre-existing JWT access token if already acquired.
            timeout_seconds: Timeout budget in seconds for RPC operations.
            server_command: Command line tuple for stdio subprocess.
            http_client: Optional injected AsyncClient instance (e.g. for testing with ASGI).
        """
        # Functional Purpose: Configure client credentials, transport endpoints, and timeouts.
        # Explicit Dependency Tracking: httpx.AsyncClient, asyncio.subprocess.
        # Structural Relationship: Upstream client for autonomous trading agents and CLI scripts.
        # Defensive Invariant: timeout_seconds > 0.0, transport in {'http', 'stdio'}.
        if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
            raise MCPSchemaValidationException("timeout_seconds must be a strictly positive float")
        if transport not in {"http", "stdio"}:
            raise MCPSchemaValidationException("transport must be either 'http' or 'stdio'")

        self._base_url: str = base_url.rstrip("/")
        self._rpc_endpoint: str = rpc_endpoint
        self._auth_endpoint: str = auth_endpoint
        self._transport_mode: Literal["http", "stdio"] = transport
        self._username: str | None = username
        self._password: str | None = password
        self._role: str = role.upper()
        self._token: str | None = token
        self._timeout: float = float(timeout_seconds)
        self._server_command: list[str] = (
            list(server_command)
            if server_command is not None
            else [sys.executable, "-m", "quant.mcp.server"]
        )

        self._injected_client: httpx.AsyncClient | None = http_client
        self._owned_client: httpx.AsyncClient | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._initialized: bool = False
        self._cached_tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> MCPClient:
        """Asynchronous context manager entrypoint."""
        # Functional Purpose: Establish transport connections upon entering context block.
        # Explicit Dependency Tracking: connect().
        # Structural Relationship: Used in 'async with MCPClient(...) as client:' constructs.
        # Defensive Invariant: Returns fully connected and authenticated client instance.
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Asynchronous context manager exit point."""
        # Functional Purpose: Gracefully drain and tear down active transports.
        # Explicit Dependency Tracking: close().
        # Structural Relationship: Invoked on exit of context block.
        # Defensive Invariant: Closes HTTP connections and terminates stdio subprocesses.
        await self.close()

    @property
    def is_connected(self) -> bool:
        """Return True if the underlying transport is active and ready."""
        if self._transport_mode == "http":
            return (self._injected_client is not None and not self._injected_client.is_closed) or (
                self._owned_client is not None and not self._owned_client.is_closed
            )
        elif self._transport_mode == "stdio":
            return self._process is not None and self._process.returncode is None
        return False

    @property
    def token(self) -> str | None:
        """Return active JWT Bearer token if present."""
        return self._token

    async def connect(self) -> None:
        """Establish transport connection and authenticate if credentials are provided."""
        # Functional Purpose: Initialize HTTP connection pool or spawn stdio child process.
        # Explicit Dependency Tracking: httpx.AsyncClient, asyncio.create_subprocess_exec.
        # Structural Relationship: Called prior to sending RPC messages.
        # Defensive Invariant: Idempotent connection setup; does not re-open existing connections.
        if self._transport_mode == "http":
            if self._injected_client is None and self._owned_client is None:
                self._owned_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
            # Authenticate if credentials are provided and token is not set
            if self._token is None and self._username and self._password:
                await self.login()
        elif self._transport_mode == "stdio":
            if self._process is None or self._process.returncode is not None:
                try:
                    self._process = await asyncio.create_subprocess_exec(
                        *self._server_command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except Exception as exc:
                    raise MCPTransportException(
                        f"Failed to spawn stdio server process '{' '.join(self._server_command)}': {exc}"
                    ) from exc

    async def close(self) -> None:
        """Close active network sessions and terminate stdio subprocesses."""
        # Functional Purpose: Release OS resources, sockets, and child processes cleanly.
        # Explicit Dependency Tracking: AsyncClient.aclose(), Process.terminate().
        # Structural Relationship: Called during teardown or context manager exit.
        # Defensive Invariant: Does not leak orphaned processes or unclosed sockets.
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

        if self._process is not None:
            if self._process.returncode is None:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._process = None

    async def login(
        self,
        username: str | None = None,
        password: str | None = None,
        role: str | None = None,
    ) -> str:
        """Authenticate with the quant engine and acquire an HS256 JWT access token.

        Args:
            username: Override username (defaults to instance username).
            password: Override password (defaults to instance password).
            role: Override role (defaults to instance role).

        Returns:
            str: Newly issued JWT access token string.

        Raises:
            MCPAuthFailureException: If authentication credentials are rejected or invalid.
        """
        # Functional Purpose: Obtain cryptographic JWT claim token for HTTP JSON-RPC gateway.
        # Explicit Dependency Tracking: POST /api/v1/auth/token endpoint.
        # Structural Relationship: Invoked on connect or when refreshing expired tokens on 401.
        # Defensive Invariant: Sets internal token on success; raises ERR-MCP-001 on failure.
        user = username or self._username
        pwd = password or self._password
        target_role = (role or self._role).upper()

        if not user or not pwd:
            raise MCPAuthFailureException(
                "Cannot authenticate: username and password must be provided"
            )

        client = self._injected_client or self._owned_client
        if client is None:
            self._owned_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
            client = self._owned_client

        auth_payload = {
            "username": user,
            "password": pwd,
            "role": target_role,
        }

        try:
            resp = await client.post(self._auth_endpoint, json=auth_payload)
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if not isinstance(token, str) or not token:
                    raise MCPAuthFailureException("Auth endpoint response missing 'access_token'")
                self._token = token
                self._username = user
                self._password = pwd
                self._role = target_role
                return token
            elif resp.status_code in {401, 403}:
                detail = resp.text
                with contextlib.suppress(Exception):
                    detail = str(resp.json().get("detail", detail))
                raise MCPAuthFailureException(
                    f"Authentication rejected ({resp.status_code}): {detail}"
                )
            else:
                raise MCPTransportException(
                    f"Authentication endpoint returned unexpected HTTP status {resp.status_code}: {resp.text}"
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise MCPTransportException(
                f"Transport error connecting to auth endpoint '{self._auth_endpoint}': {exc}"
            ) from exc

    async def initialize(
        self,
        client_name: str = DEFAULT_CLIENT_NAME,
        client_version: str = DEFAULT_CLIENT_VERSION,
    ) -> dict[str, Any]:
        """Perform formal MCP protocol initialization handshake.

        Args:
            client_name: Identifier for calling agent or platform.
            client_version: Semantic version of client integration.

        Returns:
            dict[str, Any]: Server capability and identification metadata.

        Raises:
            MCPProtocolException: If server does not adhere to MCP protocol standards.
        """
        # Functional Purpose: Execute bidirectional protocol negotiation handshake.
        # Explicit Dependency Tracking: 'initialize' JSON-RPC method.
        # Structural Relationship: First message exchange in standard MCP session.
        # Defensive Invariant: Validates protocolVersion field in server handshake response.
        init_params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": client_name,
                "version": client_version,
            },
        }
        res = await self._send_rpc("initialize", init_params)
        protocol_ver = res.get("protocolVersion")
        if not protocol_ver:
            raise MCPProtocolException(
                "MCP handshake failed: server response missing mandatory 'protocolVersion'"
            )

        self._initialized = True
        return res

    async def list_tools(self) -> list[dict[str, Any]]:
        """Query and return all tools registered on the MCP server.

        Returns:
            list[dict[str, Any]]: List of tool definitions with JSON schemas.

        Raises:
            MCPProtocolException: If tools list response is corrupt or missing.
        """
        # Functional Purpose: Dynamic discovery of institutional quant tools.
        # Explicit Dependency Tracking: 'tools/list' JSON-RPC method.
        # Structural Relationship: Populates cached tool schemas used for client-side validation.
        # Defensive Invariant: Ensures response contains a valid list of tool schemas.
        res = await self._send_rpc("tools/list", {})
        tools = res.get("tools")
        if not isinstance(tools, list):
            raise MCPProtocolException("Invalid tools/list response: 'tools' must be a list")

        self._cached_tools = tools
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        raise_on_error: bool = False,
    ) -> MCPToolCallResponse:
        """Invoke an MCP tool by name with validated JSON arguments.

        Args:
            name: Target tool identifier (e.g. 'quant_portfolio_telemetry').
            arguments: Dictionary of arguments conforming to tool schema.
            raise_on_error: If True, raises MCPToolExecutionException when isError is True.

        Returns:
            MCPToolCallResponse: Structured response containing content, error status, and parsed data.

        Raises:
            MCPSchemaValidationException: If arguments fail pre-flight validation (ERR-MCP-005).
            MCPToolExecutionException: If raise_on_error=True and tool reports an error (ERR-MCP-004).
            MCPProtocolException: If JSON-RPC envelope is violated (ERR-MCP-002).
            MCPTransportException: If network or pipe transport fails (ERR-MCP-003).
        """
        # Functional Purpose: Standardized, validated dispatch of quantitative tool calls.
        # Explicit Dependency Tracking: 'tools/call' JSON-RPC method, _validate_tool_call.
        # Structural Relationship: Primary action interface for autonomous agent decision loop.
        # Defensive Invariant: Client-side validation prevents corrupt/non-finite inputs (INV-MCP-001).
        args = arguments if arguments is not None else {}
        self._validate_tool_call(name, args)

        payload_params = {"name": name, "arguments": args}
        raw_res = await self._send_rpc("tools/call", payload_params)

        content = raw_res.get("content")
        if not isinstance(content, list):
            raise MCPProtocolException(
                "Invalid tools/call result: 'content' must be an array of content items"
            )

        is_error = bool(raw_res.get("isError", False))
        raw_text = ""
        parsed_data: Any = None

        if content and isinstance(content[0], dict):
            raw_text = str(content[0].get("text", ""))
            if raw_text:
                try:
                    parsed_data = json.loads(raw_text)
                except json.JSONDecodeError:
                    parsed_data = raw_text

        if is_error and raise_on_error:
            err_msg = raw_text or f"Tool '{name}' failed with isError=True"
            if isinstance(parsed_data, dict) and "error" in parsed_data:
                err_msg = str(parsed_data["error"])
            raise MCPToolExecutionException(f"Tool '{name}' execution failed: {err_msg}")

        return MCPToolCallResponse(
            content=content,
            is_error=is_error,
            data=parsed_data,
            raw_text=raw_text,
        )

    # -------------------------------------------------------------------------
    # Type-Safe High-Level Quantitative Helper Methods
    # -------------------------------------------------------------------------

    async def get_portfolio_telemetry(self) -> dict[str, Any]:
        """Fetch real-time firm-wide portfolio valuation, cash, leverage, and kill switch status.

        Returns:
            dict[str, Any]: Portfolio risk telemetry metrics dictionary.
        """
        # Functional Purpose: Retrieve high-frequency portfolio risk and cash telemetry.
        # Explicit Dependency Tracking: quant_portfolio_telemetry MCP tool.
        # Structural Relationship: Feeds agent observation phase before sizing new allocations.
        # Defensive Invariant: Requires is_error is False; raises on failure.
        resp = await self.call_tool("quant_portfolio_telemetry", {}, raise_on_error=True)
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_telemetry": resp.raw_text}

    async def get_macro_regimes(self) -> dict[str, Any]:
        """Inspect macroeconomic indicators from the Federal Reserve (FRED).

        Returns:
            dict[str, Any]: Dictionary containing T10Y2Y spread, DFF rate, and inversion status.
        """
        # Functional Purpose: Inspect macro rates and yield curve slope for regime conditioning.
        # Explicit Dependency Tracking: quant_macro_regimes MCP tool.
        # Structural Relationship: Ingested by agent to adjust risk appetite under inversion.
        # Defensive Invariant: Guarantees parsed regime dictionary is returned.
        resp = await self.call_tool("quant_macro_regimes", {}, raise_on_error=True)
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_macro": resp.raw_text}

    async def get_market_orderbook(self, symbol: str) -> dict[str, Any]:
        """Query consolidated quote, VWAP, and volume for a ticker symbol.

        Args:
            symbol: Target ticker symbol (e.g. 'SPY', 'AAPL').

        Returns:
            dict[str, Any]: Pricing and order book metrics dictionary.
        """
        # Functional Purpose: Inspect NBBO quote and recent volume for candidate asset.
        # Explicit Dependency Tracking: quant_market_orderbook MCP tool.
        # Structural Relationship: Consulted by agent prior to formulating order prices.
        # Defensive Invariant: symbol must be a non-empty alphanumeric string.
        resp = await self.call_tool(
            "quant_market_orderbook",
            {"symbol": symbol.strip().upper()},
            raise_on_error=True,
        )
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_orderbook": resp.raw_text}

    async def evaluate_pre_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        reference_price: float,
        *,
        is_position_exit: bool = False,
        market_spread_bps: float = 5.0,
        order_book_imbalance: float = 0.0,
        macro_yield_spread: float = 0.18,
    ) -> dict[str, Any]:
        """Evaluate a proposed order through the 5-dimensional Bayesian log-odds decision gate.

        Args:
            symbol: Target ticker symbol.
            action: Trade action ('BUY', 'SELL', 'CLOSE', 'LIQUIDATE').
            quantity: Number of shares (strictly positive float).
            reference_price: Reference price (strictly positive float).
            is_position_exit: Whether order is an exit/de-risking trade.
            market_spread_bps: Bid-ask spread in basis points.
            order_book_imbalance: Order book queue imbalance ratio in [-1.0, 1.0].
            macro_yield_spread: 10Y-2Y yield curve spread in percentage points.

        Returns:
            dict[str, Any]: Pre-trade gate evaluation result with decision, allowed, and checks.
        """
        # Functional Purpose: Run mandatory pre-trade simulation prior to execution.
        # Explicit Dependency Tracking: quant_evaluate_pre_trade MCP tool.
        # Structural Relationship: Gatekeeper preventing toxic or out-of-budget order routing.
        # Defensive Invariant: Finite positive scalars enforced on quantity and reference_price.
        args = {
            "symbol": symbol.strip().upper(),
            "action": action.strip().upper(),
            "quantity": float(quantity),
            "reference_price": float(reference_price),
            "is_position_exit": bool(is_position_exit),
            "market_spread_bps": float(market_spread_bps),
            "order_book_imbalance": float(order_book_imbalance),
            "macro_yield_spread": float(macro_yield_spread),
        }
        resp = await self.call_tool("quant_evaluate_pre_trade", args, raise_on_error=True)
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_decision": resp.raw_text}

    async def get_swarm_status(self) -> dict[str, Any]:
        """Inspect autonomous trading swarm daemon lifecycle state and allocations.

        Returns:
            dict[str, Any]: Swarm engine state, current iteration, and universe list.
        """
        # Functional Purpose: Query health and execution status of the autonomous background swarm.
        # Explicit Dependency Tracking: quant_swarm_status MCP tool.
        # Structural Relationship: Enables supervising agent to monitor background trader daemons.
        # Defensive Invariant: Returns structured swarm state dictionary.
        resp = await self.call_tool("quant_swarm_status", {}, raise_on_error=True)
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_status": resp.raw_text}

    async def trigger_emergency_panic(
        self,
        reason: str,
        details: str = "Triggered via MCP client",
    ) -> dict[str, Any]:
        """EMERGENCY KILL SWITCH: Immediately lock order entry and cancel resting orders.

        Requires ADMIN or SYSTEM role.

        Args:
            reason: Justification or incident diagnosis for panic trigger.
            details: Contextual details regarding the failure.

        Returns:
            dict[str, Any]: Panic cancellation event summary.
        """
        # Functional Purpose: Provide emergency circuit-breaker trip capability to AI agents.
        # Explicit Dependency Tracking: quant_emergency_panic MCP tool.
        # Structural Relationship: Emergency halt interface for risk guardians.
        # Defensive Invariant: Requires ADMIN role authorization on backend gateway.
        resp = await self.call_tool(
            "quant_emergency_panic",
            {"reason": reason, "details": details},
            raise_on_error=True,
        )
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_panic_res": resp.raw_text}

    async def reset_kill_switch(self, admin_token: str) -> dict[str, Any]:
        """Disarm emergency kill switch back to standby using administrator secret token.

        Requires ADMIN role.

        Args:
            admin_token: Cryptographic administrator secret token.

        Returns:
            dict[str, Any]: Reset confirmation status payload.
        """
        # Functional Purpose: Re-arm emergency kill switch to standby following resolution.
        # Explicit Dependency Tracking: quant_kill_switch_reset MCP tool.
        # Structural Relationship: Admin-only recovery command.
        # Defensive Invariant: Validates admin_token parameter is non-empty.
        resp = await self.call_tool(
            "quant_kill_switch_reset",
            {"admin_token": admin_token},
            raise_on_error=True,
        )
        if isinstance(resp.data, dict):
            return resp.data
        return {"raw_reset_res": resp.raw_text}

    # -------------------------------------------------------------------------
    # Internal Protocol Routing & Validation Engine
    # -------------------------------------------------------------------------

    async def _send_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch JSON-RPC 2.0 message over configured transport and return 'result'.

        Args:
            method: JSON-RPC method name (e.g. 'initialize', 'tools/list', 'tools/call').
            params: Parameters dictionary.

        Returns:
            dict[str, Any]: Unwrapped JSON-RPC 'result' payload.

        Raises:
            MCPProtocolException: If server returns JSON-RPC 'error' or envelope violation.
            MCPTransportException: If transport fails, drops, or times out.
        """
        # Functional Purpose: Multiplex JSON-RPC 2.0 messages over HTTP or stdio transports.
        # Explicit Dependency Tracking: _send_rpc_http, _send_rpc_stdio.
        # Structural Relationship: Core communication channel for all MCP methods.
        # Defensive Invariant: Generates strictly monotonic request IDs and enforces JSON-RPC 2.0.
        self._request_id += 1
        req_id = self._request_id
        req_payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            req_payload["params"] = params

        if not self.is_connected:
            await self.connect()

        if self._transport_mode == "http":
            response_dict = await self._send_rpc_http(req_payload)
        elif self._transport_mode == "stdio":
            response_dict = await self._send_rpc_stdio(req_payload)
        else:
            raise MCPTransportException(f"Unsupported transport mode: {self._transport_mode}")

        # Enforce JSON-RPC 2.0 envelope invariants
        if not isinstance(response_dict, dict):
            raise MCPProtocolException(
                f"Protocol violation: Server response is not a JSON object ({type(response_dict).__name__})"
            )

        if "error" in response_dict:
            err = response_dict["error"]
            err_code = err.get("code", -32603) if isinstance(err, dict) else -32603
            err_msg = (
                err.get("message", "Unknown JSON-RPC error") if isinstance(err, dict) else str(err)
            )
            raise MCPProtocolException(f"Server returned error code {err_code}: {err_msg}")

        if "result" not in response_dict:
            raise MCPProtocolException(
                "Protocol violation: Server response missing mandatory 'result' field"
            )

        result_val = response_dict["result"]
        if not isinstance(result_val, dict):
            raise MCPProtocolException(
                f"Protocol violation: 'result' field must be an object ({type(result_val).__name__})"
            )

        return result_val

    async def _send_rpc_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch JSON-RPC message over HTTP with automatic JWT renewal on 401."""
        # Functional Purpose: Send HTTP POST to /api/v1/mcp/rpc with Bearer authorization.
        # Explicit Dependency Tracking: httpx.AsyncClient.post, login().
        # Structural Relationship: Handles HTTP transport layer for _send_rpc.
        # Defensive Invariant: Renews token once if 401 Unauthorized is encountered (INV-MCP-002).
        client = self._injected_client or self._owned_client
        if client is None:
            raise MCPTransportException("HTTP client connection not established")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            resp = await client.post(self._rpc_endpoint, json=payload, headers=headers)

            # Auto-renewal on 401 Unauthorized
            if resp.status_code == 401 and self._username and self._password:
                logger.info(
                    "MCP gateway returned 401 Unauthorized. Attempting automatic JWT renewal..."
                )
                await self.login()
                headers["Authorization"] = f"Bearer {self._token}"
                resp = await client.post(self._rpc_endpoint, json=payload, headers=headers)

            if resp.status_code == 401:
                raise MCPAuthFailureException(
                    "Unauthorized (HTTP 401): Valid Bearer token required for MCP gateway"
                )

            if resp.status_code >= 400:
                raise MCPTransportException(
                    f"HTTP transport failed with status {resp.status_code}: {resp.text}"
                )

            res_json = resp.json()
            if not isinstance(res_json, dict):
                raise MCPProtocolException("HTTP response payload is not a valid JSON dictionary")
            return res_json

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise MCPTransportException(
                f"Transport failure connecting to MCP RPC endpoint '{self._rpc_endpoint}': {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise MCPProtocolException(f"Failed to decode server JSON response: {exc}") from exc

    async def _send_rpc_stdio(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch JSON-RPC message over standard I/O pipes to child server process."""
        # Functional Purpose: Transmit newline-delimited JSON-RPC to stdio server process.
        # Explicit Dependency Tracking: asyncio.subprocess.Process pipes.
        # Structural Relationship: Handles stdio transport layer for _send_rpc.
        # Defensive Invariant: Traps process exits, EOF, and malformed JSON lines.
        if (
            self._process is None
            or self._process.stdin is None
            or self._process.stdout is None
            or self._process.returncode is not None
        ):
            raise MCPTransportException("Stdio server process is not running or pipes unavailable")

        raw_req = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._process.stdin.write(raw_req.encode("utf-8"))
            await self._process.stdin.drain()

            line_bytes = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self._timeout,
            )
            if not line_bytes:
                raise MCPTransportException("Stdio server process closed output stream (EOF)")

            raw_resp = line_bytes.decode("utf-8").strip()
            res_json = json.loads(raw_resp)
            if not isinstance(res_json, dict):
                raise MCPProtocolException(
                    "Stdio server emitted response that is not a JSON object"
                )
            return res_json

        except TimeoutError as exc:
            raise MCPTransportException(
                f"Timeout waiting {self._timeout}s for stdio server response"
            ) from exc
        except (OSError, BrokenPipeError) as exc:
            raise MCPTransportException(
                f"Pipe error communicating with stdio server: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise MCPProtocolException(
                f"Failed to parse JSON response from stdio server: {exc}"
            ) from exc

    def _validate_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        """Enforce client-side parameter schema and domain bounds validation (INV-MCP-001).

        Args:
            name: Target tool identifier.
            arguments: Tool parameters dictionary.

        Raises:
            MCPSchemaValidationException: If arguments fail structural or numerical checks.
        """
        # Functional Purpose: Catch hallucinated, missing, or non-finite parameters client-side.
        # Explicit Dependency Tracking: math.isfinite.
        # Structural Relationship: Pre-flight validator executed before call_tool network dispatch.
        # Defensive Invariant: Prohibits NaNs, Infs, missing required keys, or negative quantities.
        if not isinstance(name, str) or not name.strip():
            raise MCPSchemaValidationException("Tool name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise MCPSchemaValidationException("Tool arguments must be a dictionary")

        # Generic defensive inspection: Prohibit non-finite numbers and unexpected boolean numbers
        for key, val in arguments.items():
            if (
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and not math.isfinite(val)
            ):
                raise MCPSchemaValidationException(
                    f"Argument '{key}' has non-finite value '{val}' for tool '{name}'"
                )

        # Domain-specific constraints for core quantitative tools
        if name == "quant_market_orderbook":
            symbol = arguments.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                raise MCPSchemaValidationException(
                    "Tool 'quant_market_orderbook' requires non-empty string argument 'symbol'"
                )

        elif name == "quant_evaluate_pre_trade":
            symbol = arguments.get("symbol")
            action = arguments.get("action")
            quantity = arguments.get("quantity")
            reference_price = arguments.get("reference_price")

            if not isinstance(symbol, str) or not symbol.strip():
                raise MCPSchemaValidationException(
                    "Pre-trade gate requires non-empty string 'symbol'"
                )

            valid_actions = {"BUY", "SELL", "CLOSE", "LIQUIDATE"}
            if not isinstance(action, str) or action.upper() not in valid_actions:
                raise MCPSchemaValidationException(
                    f"Pre-trade gate action must be one of {valid_actions}, got '{action}'"
                )

            if (
                isinstance(quantity, bool)
                or not isinstance(quantity, (int, float))
                or quantity <= 0.0
            ):
                raise MCPSchemaValidationException(
                    f"Pre-trade gate quantity must be strictly positive float, got '{quantity}'"
                )

            if (
                isinstance(reference_price, bool)
                or not isinstance(reference_price, (int, float))
                or reference_price <= 0.0
            ):
                raise MCPSchemaValidationException(
                    f"Pre-trade gate reference_price must be strictly positive float, got '{reference_price}'"
                )

            obi = arguments.get("order_book_imbalance")
            if (
                obi is not None
                and not isinstance(obi, bool)
                and isinstance(obi, (int, float))
                and not (-1.0 <= float(obi) <= 1.0)
            ):
                raise MCPSchemaValidationException(
                    f"Order book imbalance must be bounded in [-1.0, 1.0], got '{obi}'"
                )

        elif name == "quant_emergency_panic":
            reason = arguments.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise MCPSchemaValidationException(
                    "Emergency panic requires a non-empty string 'reason'"
                )

        elif name == "quant_kill_switch_reset":
            admin_token = arguments.get("admin_token")
            if not isinstance(admin_token, str) or not admin_token.strip():
                raise MCPSchemaValidationException(
                    "Kill switch reset requires non-empty string 'admin_token'"
                )
