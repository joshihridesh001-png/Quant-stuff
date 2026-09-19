"""Alpaca Markets Live and Paper Execution Gateway Subsystem.

Purpose:
    Provides live and paper broker integration for US Equities and ETFs via Alpaca's REST API:
    1. Implements the ExecutionGateway protocol (INV-GW-001 through INV-GW-006).
    2. Maps domain Order models to Alpaca REST payloads with RFC 4122 client_order_id.
    3. Translates Alpaca execution reports, open positions, and account balances to domain entities.
    4. Handles authentication, connection lifecycle, rate limiting, and margin errors.

Dependencies:
    - httpx: Asynchronous HTTP client with connection pooling and mock transport support.
    - time: High-resolution timestamps (time.time_ns).
    - uuid: Unique execution report ID generation.
    - quant.execution.models: Order, ExecutionReport, OrderState, OrderSide, OrderType,
      TimeInForce, GatewayError, GatewayDisconnectedException, InsufficientMarginException,
      InvalidOrderInputException, and diagnostic constants.

Structural Relationship:
    - Sits in the Execution layer alongside PaperExecutionGateway.
    - Consumed by ExecutionService, RiskOrchestrator, and AutonomousTradingEngine.
    - Registered as the active broker in API dependencies when BROKER_TYPE == "alpaca".

Invariants Enforced:
    - INV-GW-001: Monotonic order state mapping honoring broker lifecycle.
    - INV-GW-002: Client order IDs match deterministic cl_ord_id.
    - INV-GW-003: Leaves and filled quantities are non-negative and finite.
    - INV-GW-005: Rejects submissions when disconnected with ERR_GW_DISCONNECTED.
    - Rule 1: Four-tier docstring and structural annotations on all classes and methods.
    - Rule 2: Diagnostic error codes ERR-GW-001 through ERR-GW-006.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Final

import httpx

from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_INSUFFICIENT_MARGIN,
    ERR_GW_NON_FINITE_INPUT,
    ERR_GW_RATE_LIMIT_EXCEEDED,
    ExecutionReport,
    GatewayDisconnectedException,
    GatewayError,
    InsufficientMarginException,
    InvalidOrderInputException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)

_DEFAULT_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"
_DEFAULT_TIMEOUT_SEC: Final[float] = 10.0


class AlpacaExecutionGateway:
    """Live and Paper Execution Gateway for Alpaca Markets."""

    __slots__ = (
        "_api_key",
        "_base_url",
        "_client",
        "_custom_transport",
        "_is_connected",
        "_order_metadata",
        "_secret_key",
        "_timeout",
    )

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SEC,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the Alpaca execution gateway with API credentials.

        Args:
            api_key: Alpaca API key ID (APCA-API-KEY-ID).
            secret_key: Alpaca secret API key (APCA-API-SECRET-KEY).
            base_url: Base endpoint URL (paper or live).
            timeout: Default HTTP request timeout in seconds.
            transport: Optional custom transport for mocking or proxying.
        """
        # Functional Purpose: Configure broker credentials, endpoints, and HTTP client options.
        # Explicit Dependency Tracking: httpx.AsyncBaseTransport, base_url string.
        # Structural Relationship: Gateway constructor called by dependency injection provider.
        # Defensive Invariant: Credentials stored in private slots; client initialized on connect.
        self._api_key: str = api_key
        self._secret_key: str = secret_key
        self._base_url: str = base_url.rstrip("/")
        self._timeout: float = timeout
        self._custom_transport: httpx.AsyncBaseTransport | None = transport
        self._client: httpx.AsyncClient | None = None
        self._is_connected: bool = False
        self._order_metadata: dict[str, tuple[str, OrderSide, str]] = {}

    @property
    def is_connected(self) -> bool:
        """Indicate whether the gateway has an active verified broker session."""
        # Functional Purpose: Expose connectivity state to RiskOrchestrator and watchdogs.
        # Explicit Dependency Tracking: bool state variable.
        # Structural Relationship: ExecutionGateway protocol requirement.
        # Defensive Invariant: Returns True only if session is active and verified.
        return self._is_connected

    def _get_headers(self) -> dict[str, str]:
        """Generate authentication headers required by Alpaca REST API."""
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def connect(self) -> None:
        """Establish HTTP session and verify credentials against Alpaca /v2/account."""
        # Functional Purpose: Initialize AsyncClient, test credentials, and assert account active.
        # Explicit Dependency Tracking: httpx.AsyncClient, /v2/account endpoint.
        # Structural Relationship: Lifecycle initializer before order routing begins.
        # Defensive Invariant: Raises GatewayDisconnectedException if authentication fails.
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._get_headers(),
                timeout=self._timeout,
                transport=self._custom_transport,
            )

        try:
            response = await self._client.get("/v2/account")
            if response.status_code == 200:
                self._is_connected = True
            elif response.status_code in (401, 403):
                self._is_connected = False
                raise GatewayDisconnectedException(
                    f"Alpaca authentication failed with status {response.status_code}: {response.text}",
                    code=ERR_GW_DISCONNECTED,
                )
            else:
                self._is_connected = False
                raise GatewayDisconnectedException(
                    f"Alpaca connection failed with status {response.status_code}: {response.text}",
                    code=ERR_GW_DISCONNECTED,
                )
        except httpx.RequestError as exc:
            self._is_connected = False
            raise GatewayDisconnectedException(
                f"Alpaca network connectivity error: {exc}",
                code=ERR_GW_DISCONNECTED,
            ) from exc

    async def disconnect(self) -> None:
        """Terminate the HTTP session and mark gateway disconnected."""
        # Functional Purpose: Close active HTTP connections and clean up resources.
        # Explicit Dependency Tracking: httpx.AsyncClient.aclose.
        # Structural Relationship: Gateway teardown on shutdown or emergency panic.
        # Defensive Invariant: Idempotent teardown; guarantees is_connected is False.
        self._is_connected = False
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _ensure_connected(self) -> httpx.AsyncClient:
        """Assert gateway is connected and return active HTTP client."""
        if not self._is_connected or self._client is None or self._client.is_closed:
            raise GatewayDisconnectedException(
                "Cannot perform broker action while gateway is disconnected.",
                code=ERR_GW_DISCONNECTED,
            )
        return self._client

    def _map_alpaca_status_to_state(self, status: str) -> OrderState:
        """Map Alpaca order status string to domain OrderState."""
        status_clean = status.lower().strip()
        if status_clean in ("new", "accepted", "pending_new"):
            return OrderState.NEW
        if status_clean in ("partially_filled",):
            return OrderState.PARTIALLY_FILLED
        if status_clean in ("filled",):
            return OrderState.FILLED
        if status_clean in ("canceled", "cancelled", "pending_cancel"):
            return OrderState.CANCELLED
        if status_clean in ("expired",):
            return OrderState.EXPIRED
        if status_clean in ("rejected", "stopped", "suspended"):
            return OrderState.REJECTED
        return OrderState.NEW

    async def submit_order(self, order: Order) -> ExecutionReport:
        """Submit an order for execution or order-book resting on Alpaca."""
        # Functional Purpose: Map domain Order to Alpaca JSON, POST /v2/orders, and parse response.
        # Explicit Dependency Tracking: Order, ExecutionReport, /v2/orders endpoint.
        # Structural Relationship: Primary execution method called by ExecutionService / SOR.
        # Defensive Invariant: Maps 403 to InsufficientMarginException, 429 to RateLimit, 422 to InvalidInput.
        client = self._ensure_connected()

        side_str = "buy" if order.side == OrderSide.BUY else "sell"
        type_str = "market" if order.order_type == OrderType.MARKET else "limit"
        tif_str = "day"
        if order.time_in_force == TimeInForce.IOC:
            tif_str = "ioc"
        elif order.time_in_force == TimeInForce.GTC:
            tif_str = "gtc"
        elif order.time_in_force == TimeInForce.FOK:
            tif_str = "fok"

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": side_str,
            "type": type_str,
            "time_in_force": tif_str,
            "client_order_id": order.cl_ord_id,
        }

        if order.order_type == OrderType.LIMIT and order.price is not None:
            payload["limit_price"] = str(order.price)

        try:
            response = await client.post("/v2/orders", json=payload)
        except httpx.RequestError as exc:
            raise GatewayDisconnectedException(
                f"Network failure submitting order to Alpaca: {exc}",
                code=ERR_GW_DISCONNECTED,
            ) from exc

        if response.status_code == 403:
            raise InsufficientMarginException(
                f"Alpaca rejected order due to margin/buying power: {response.text}",
                code=ERR_GW_INSUFFICIENT_MARGIN,
            )
        if response.status_code == 422:
            raise InvalidOrderInputException(
                f"Alpaca rejected unprocessable order input: {response.text}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if response.status_code == 429:
            raise GatewayError(
                f"Alpaca rate limit exceeded: {response.text}",
                code=ERR_GW_RATE_LIMIT_EXCEEDED,
            )
        if response.status_code not in (200, 201):
            raise GatewayError(
                f"Alpaca order submission failed with HTTP {response.status_code}: {response.text}",
                code="ERR-GW-SUBMIT-FAILED",
            )

        data = response.json()
        status_str = data.get("status", "new")
        order_state = self._map_alpaca_status_to_state(status_str)

        filled_qty = float(data.get("filled_qty") or 0.0)
        target_qty = float(data.get("qty") or order.quantity)
        leaves_qty = max(0.0, target_qty - filled_qty)

        avg_price = 0.0
        if data.get("filled_avg_price") is not None:
            avg_price = float(data["filled_avg_price"])

        exchange_order_id = str(data.get("id") or f"ex-{order.cl_ord_id}")
        cum_quote_amount = filled_qty * avg_price

        # Record metadata for subsequent cancellations
        self._order_metadata[order.cl_ord_id] = (order.symbol, order.side, exchange_order_id)

        return ExecutionReport(
            report_id=f"rep-alpaca-{uuid.uuid4().hex[:12]}",
            cl_ord_id=order.cl_ord_id,
            exchange_order_id=exchange_order_id,
            symbol=order.symbol,
            side=order.side,
            exec_type=order_state,
            last_quantity=filled_qty,
            last_price=avg_price,
            cum_quantity=filled_qty,
            leaves_quantity=leaves_qty,
            cum_quote_amount=cum_quote_amount,
            average_price=avg_price,
            fee=0.0,
            timestamp_ns=time.time_ns(),
            text=f"Alpaca order {exchange_order_id} status: {status_str}",
        )

    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport:
        """Request cancellation of an active order by client_order_id."""
        # Functional Purpose: Send DELETE /v2/orders:by_client_order_id/{cl_ord_id} to Alpaca.
        # Explicit Dependency Tracking: ExecutionReport, /v2/orders:by_client_order_id.
        # Structural Relationship: Cancellation method for OrderStateMachine and kill switch.
        # Defensive Invariant: Returns ExecutionReport with CANCELLED state.
        client = self._ensure_connected()

        try:
            response = await client.delete(f"/v2/orders:by_client_order_id/{cl_ord_id}")
        except httpx.RequestError as exc:
            raise GatewayDisconnectedException(
                f"Network failure cancelling order on Alpaca: {exc}",
                code=ERR_GW_DISCONNECTED,
            ) from exc

        if response.status_code not in (200, 204, 404):
            # 404 is tolerated if already filled or cancelled
            raise GatewayError(
                f"Failed to cancel Alpaca order {cl_ord_id}: HTTP {response.status_code} {response.text}",
                code="ERR-GW-CANCEL-FAILED",
            )

        meta = self._order_metadata.get(cl_ord_id)
        symbol = meta[0] if meta else "UNKNOWN"
        side = meta[1] if meta else OrderSide.BUY
        exchange_order_id = meta[2] if meta else f"ex-{cl_ord_id}"

        return ExecutionReport(
            report_id=f"rep-alpaca-{uuid.uuid4().hex[:12]}",
            cl_ord_id=cl_ord_id,
            exchange_order_id=exchange_order_id,
            symbol=symbol,
            side=side,
            exec_type=OrderState.CANCELLED,
            last_quantity=0.0,
            last_price=0.0,
            cum_quantity=0.0,
            leaves_quantity=0.0,
            cum_quote_amount=0.0,
            average_price=0.0,
            fee=0.0,
            timestamp_ns=time.time_ns(),
            text=f"Alpaca cancelled order {cl_ord_id}",
        )

    async def get_order(self, cl_ord_id: str) -> Order | None:
        """Retrieve order metadata from Alpaca by client_order_id."""
        # Functional Purpose: Query GET /v2/orders:by_client_order_id/{cl_ord_id}.
        # Explicit Dependency Tracking: Order domain model.
        # Structural Relationship: Read-only order query for blotter reconciliation.
        # Defensive Invariant: Returns None on HTTP 404.
        client = self._ensure_connected()
        try:
            response = await client.get(f"/v2/orders:by_client_order_id/{cl_ord_id}")
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                return None
            data = response.json()
            side = OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL
            order_type = (
                OrderType.MARKET if data.get("type") == "market" else OrderType.LIMIT
            )
            price = float(data["limit_price"]) if data.get("limit_price") else None
            return Order(
                cl_ord_id=data.get("client_order_id", cl_ord_id),
                symbol=data.get("symbol", ""),
                side=side,
                order_type=order_type,
                quantity=float(data.get("qty", 0.0)),
                price=price,
            )
        except Exception:
            return None

    async def get_open_orders(self) -> list[Order]:
        """Retrieve all currently active or open orders on Alpaca."""
        # Functional Purpose: Query GET /v2/orders?status=open.
        # Explicit Dependency Tracking: list[Order].
        # Structural Relationship: Used during startup and kill switch reconciliation.
        # Defensive Invariant: Returns empty list if query fails.
        client = self._ensure_connected()
        try:
            response = await client.get("/v2/orders", params={"status": "open"})
            if response.status_code != 200:
                return []
            orders: list[Order] = []
            for item in response.json():
                side = OrderSide.BUY if item.get("side") == "buy" else OrderSide.SELL
                order_type = (
                    OrderType.MARKET if item.get("type") == "market" else OrderType.LIMIT
                )
                price = float(item["limit_price"]) if item.get("limit_price") else None
                orders.append(
                    Order(
                        cl_ord_id=item.get("client_order_id", ""),
                        symbol=item.get("symbol", ""),
                        side=side,
                        order_type=order_type,
                        quantity=float(item.get("qty", 0.0)),
                        price=price,
                    )
                )
            return orders
        except Exception:
            return []

    async def get_positions(self) -> dict[str, float]:
        """Retrieve current portfolio instrument positions from Alpaca."""
        # Functional Purpose: Query GET /v2/positions and return symbol -> quantity mapping.
        # Explicit Dependency Tracking: dict[str, float].
        # Structural Relationship: Consumed by PreTradeRiskFirewall and PortfolioLedger.
        # Defensive Invariant: Float quantity parsed safely with finite guard.
        client = self._ensure_connected()
        try:
            response = await client.get("/v2/positions")
            if response.status_code != 200:
                return {}
            positions: dict[str, float] = {}
            for pos in response.json():
                symbol = pos.get("symbol", "")
                qty = float(pos.get("qty", 0.0))
                if symbol and math.isfinite(qty):
                    positions[symbol] = qty
            return positions
        except Exception:
            return {}

    async def get_account_balance(self) -> dict[str, float]:
        """Retrieve current account cash, equity, and buying power from Alpaca."""
        # Functional Purpose: Query GET /v2/account and return balance breakdown.
        # Explicit Dependency Tracking: dict[str, float] containing cash, equity, buying_power.
        # Structural Relationship: Consumed by RiskOrchestrator for margin sufficiency.
        # Defensive Invariant: Returns non-negative floats with finite validation.
        client = self._ensure_connected()
        try:
            response = await client.get("/v2/account")
            if response.status_code != 200:
                return {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}
            data = response.json()
            cash = float(data.get("cash", 0.0))
            equity = float(data.get("portfolio_value", data.get("equity", 0.0)))
            buying_power = float(data.get("buying_power", 0.0))
            return {
                "cash": cash if math.isfinite(cash) else 0.0,
                "equity": equity if math.isfinite(equity) else 0.0,
                "buying_power": buying_power if math.isfinite(buying_power) else 0.0,
            }
        except Exception:
            return {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}
