"""Pure domain entities, enums, exceptions, and diagnostic error codes for live order execution.

Purpose:
    Establishes core domain value objects and entities representing orders, execution reports,
    lifecycle states, and invariant protection for the Phase 6 Execution Gateway subsystem.

Dependencies:
    - dataclasses: High-performance dataclasses with slots and immutability controls.
    - enum: Python 3.11+ StrEnum for zero-overhead string-compatible enumerations.
    - math: Non-finite scalar verification (math.isfinite).
    - typing: Static typing annotations and Final constants.

Structural Relationship:
    - Root domain model package for:
        1. OrderStateMachine (fsm.py)
        2. IdempotencyRouter (idempotency.py)
        3. ExecutionGateway & PaperExecutionGateway (gateway.py)
        4. OrderAuditLogger (audit.py)
    - Emits: Order, ExecutionReport, OrderState, OrderSide, OrderType, TimeInForce,
      and diagnostic exception hierarchy.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity): Terminal states are terminal.
    - INV-GW-002 (Cryptographic Idempotency Token Uniqueness): Unique cl_ord_id tracking.
    - INV-GW-003 (Execution Mass Conservation): Leaves quantity non-negative and bounded.
    - INV-GW-005 (Strict Non-Finite Input Protection): Complete rejection of NaN, Inf, bool,
      and non-positive prices or lot sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_GW_INVALID_STATE_TRANSITION: Final[str] = "ERR-GW-001"
ERR_GW_DUPLICATE_ORDER_ID: Final[str] = "ERR-GW-002"
ERR_GW_NON_FINITE_INPUT: Final[str] = "ERR-GW-003"
ERR_GW_INSUFFICIENT_MARGIN: Final[str] = "ERR-GW-004"
ERR_GW_DISCONNECTED: Final[str] = "ERR-GW-005"
ERR_GW_RATE_LIMIT_EXCEEDED: Final[str] = "ERR-GW-006"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Exception Protocol)
# ============================================================================


class GatewayError(Exception):
    """Base exception for all live execution gateway and order state machine errors."""

    def __init__(self, message: str, code: str = "ERR-GW-000") -> None:
        # Functional Purpose: Initialize base execution gateway exception with diagnostic code.
        # Explicit Dependency Tracking: Exception base class.
        # Structural Relationship: Root of the execution error taxonomy.
        # Defensive Invariant: code must be a non-empty string diagnostic identifier.
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class InvalidStateTransitionException(GatewayError):
    """Raised when an order transition violates the deterministic transition graph (INV-GW-001)."""

    def __init__(self, message: str, code: str = ERR_GW_INVALID_STATE_TRANSITION) -> None:
        # Functional Purpose: Signal invalid, terminal, or forbidden order state transitions.
        # Explicit Dependency Tracking: ERR_GW_INVALID_STATE_TRANSITION diagnostic fault code.
        # Structural Relationship: Emitted by OrderStateMachine when an order transition is invalid.
        # Defensive Invariant: code defaults to ERR-GW-001.
        super().__init__(message=message, code=code)


class DuplicateOrderException(GatewayError):
    """Raised when a client_order_id collision or replay attempt is detected (INV-GW-002)."""

    def __init__(self, message: str, code: str = ERR_GW_DUPLICATE_ORDER_ID) -> None:
        # Functional Purpose: Signal duplicate or replayed client_order_id on active orders.
        # Explicit Dependency Tracking: ERR_GW_DUPLICATE_ORDER_ID diagnostic fault code.
        # Structural Relationship: Emitted by IdempotencyRouter on order registration collision.
        # Defensive Invariant: code defaults to ERR-GW-002.
        super().__init__(message=message, code=code)


class InvalidOrderInputException(GatewayError):
    """Raised when order parameters violate domain boundaries, types, or finiteness (INV-GW-005)."""

    def __init__(self, message: str, code: str = ERR_GW_NON_FINITE_INPUT) -> None:
        # Functional Purpose: Signal NaN, infinite, boolean, or negative lot sizes/prices.
        # Explicit Dependency Tracking: ERR_GW_NON_FINITE_INPUT diagnostic fault code.
        # Structural Relationship: Emitted by Order and ExecutionReport __post_init__ validators.
        # Defensive Invariant: code defaults to ERR-GW-003.
        super().__init__(message=message, code=code)


class InsufficientMarginException(GatewayError):
    """Raised when an order notional value exceeds available margin or balance (INV-GW-003)."""

    def __init__(self, message: str, code: str = ERR_GW_INSUFFICIENT_MARGIN) -> None:
        # Functional Purpose: Signal order rejection caused by insufficient balance or buying power.
        # Explicit Dependency Tracking: ERR_GW_INSUFFICIENT_MARGIN diagnostic fault code.
        # Structural Relationship: Emitted by ExecutionGateway risk filters and paper broker.
        # Defensive Invariant: code defaults to ERR-GW-004.
        super().__init__(message=message, code=code)


class GatewayDisconnectedException(GatewayError):
    """Raised when an order action is attempted while the gateway is disconnected (INV-GW-006)."""

    def __init__(self, message: str, code: str = ERR_GW_DISCONNECTED) -> None:
        # Functional Purpose: Signal order routing failure due to inactive or lost broker session.
        # Explicit Dependency Tracking: ERR_GW_DISCONNECTED diagnostic fault code.
        # Structural Relationship: Emitted by ExecutionGateway when submit/cancel is attempted offline.
        # Defensive Invariant: code defaults to ERR-GW-005.
        super().__init__(message=message, code=code)


class RateLimitExceededException(GatewayError):
    """Raised when outbound request frequency exceeds broker rate-limit capacity (INV-GW-006)."""

    def __init__(self, message: str, code: str = ERR_GW_RATE_LIMIT_EXCEEDED) -> None:
        # Functional Purpose: Signal broker request throttle or leaky-bucket exhaustion.
        # Explicit Dependency Tracking: ERR_GW_RATE_LIMIT_EXCEEDED diagnostic fault code.
        # Structural Relationship: Emitted by ExecutionGateway when exchange throttles requests.
        # Defensive Invariant: code defaults to ERR-GW-006.
        super().__init__(message=message, code=code)


# ============================================================================
# Domain Enums (StrEnum)
# ============================================================================


class OrderState(StrEnum):
    """Order lifecycle state enumeration adhering to FIX protocol semantics."""

    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(StrEnum):
    """Order trading direction."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Execution instruction and order book matching mode."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    PEGGED = "PEGGED"


class TimeInForce(StrEnum):
    """Order lifespan and expiration policy."""

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


# ============================================================================
# Domain Entities and Value Objects (Rule 1 & Rule 4: Best-of-the-Best Contracts)
# ============================================================================


@dataclass(slots=True)
class Order:
    """Stateful domain entity tracking an order through its execution lifecycle.

    Attributes:
        cl_ord_id: Client-assigned deterministic unique identifier.
        symbol: Market ticker or instrument identifier.
        side: Trading direction (BUY or SELL).
        order_type: Execution type (MARKET, LIMIT, STOP_LIMIT, PEGGED).
        quantity: Order quantity in lot units (must be finite, > 0.0).
        price: Optional limit price (finite, > 0.0 if specified).
        stop_price: Optional stop trigger price (finite, > 0.0 if specified).
        time_in_force: Lifespan policy (default: TimeInForce.GTC).
        state: Current order lifecycle state (default: OrderState.PENDING_NEW).
        exchange_order_id: Exchange-assigned order identifier once acknowledged.
        filled_quantity: Cumulative filled base units (finite, >= 0.0).
        filled_quote_amount: Cumulative quote consideration paid/received (finite, >= 0.0).
        average_price: Volume-weighted execution price (finite, >= 0.0).
        fees_paid: Cumulative transaction fees incurred (finite, >= 0.0).
        created_at_ns: Local submission timestamp in epoch nanoseconds (>= 0).
        updated_at_ns: Most recent state transition timestamp in epoch nanoseconds (>= 0).
    """

    cl_ord_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    state: OrderState = OrderState.PENDING_NEW
    exchange_order_id: str | None = None
    filled_quantity: float = 0.0
    filled_quote_amount: float = 0.0
    average_price: float = 0.0
    fees_paid: float = 0.0
    created_at_ns: int = 0
    updated_at_ns: int = 0

    def __post_init__(self) -> None:
        """Validate domain bounds, types, non-finite guards, and numeric invariants upon instantiation."""
        # Functional Purpose: Enforce INV-GW-005 non-finite input guards and boundary invariants.
        # Explicit Dependency Tracking: math.isfinite, InvalidOrderInputException, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Invoked automatically by dataclass __init__ on every Order creation.
        # Defensive Invariant: All quantities/prices must be strictly finite, non-boolean, and positive.

        # 1. Validate string identifiers
        if not isinstance(self.cl_ord_id, str) or not self.cl_ord_id:
            raise InvalidOrderInputException(
                f"Order cl_ord_id must be a non-empty string, got {self.cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.symbol, str) or not self.symbol:
            raise InvalidOrderInputException(
                f"Order symbol must be a non-empty string, got {self.symbol!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if self.exchange_order_id is not None and not isinstance(self.exchange_order_id, str):
            raise InvalidOrderInputException(
                f"Order exchange_order_id must be str or None, got {type(self.exchange_order_id).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 2. Validate enum instances
        if not isinstance(self.side, OrderSide):
            raise InvalidOrderInputException(
                f"Order side must be OrderSide enum instance, got {type(self.side).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.order_type, OrderType):
            raise InvalidOrderInputException(
                f"Order order_type must be OrderType enum instance, got {type(self.order_type).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.time_in_force, TimeInForce):
            raise InvalidOrderInputException(
                f"Order time_in_force must be TimeInForce enum instance, got {type(self.time_in_force).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.state, OrderState):
            raise InvalidOrderInputException(
                f"Order state must be OrderState enum instance, got {type(self.state).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 3. Validate primary quantity (> 0.0, finite, non-boolean)
        if (
            not isinstance(self.quantity, (int, float))
            or isinstance(self.quantity, bool)
            or not math.isfinite(self.quantity)
            or self.quantity <= 0.0
        ):
            raise InvalidOrderInputException(
                f"Order quantity must be finite scalar > 0.0, got {self.quantity!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        self.quantity = float(self.quantity)

        # 4. Validate limit price if specified (> 0.0, finite, non-boolean)
        if self.price is not None:
            if (
                not isinstance(self.price, (int, float))
                or isinstance(self.price, bool)
                or not math.isfinite(self.price)
                or self.price <= 0.0
            ):
                raise InvalidOrderInputException(
                    f"Order price must be finite scalar > 0.0, got {self.price!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            self.price = float(self.price)

        # 5. Validate stop trigger price if specified (> 0.0, finite, non-boolean)
        if self.stop_price is not None:
            if (
                not isinstance(self.stop_price, (int, float))
                or isinstance(self.stop_price, bool)
                or not math.isfinite(self.stop_price)
                or self.stop_price <= 0.0
            ):
                raise InvalidOrderInputException(
                    f"Order stop_price must be finite scalar > 0.0, got {self.stop_price!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            self.stop_price = float(self.stop_price)

        # 6. Validate non-negative auxiliary floats (>= 0.0, finite, non-boolean)
        aux_float_fields = (
            "filled_quantity",
            "filled_quote_amount",
            "average_price",
            "fees_paid",
        )
        for field_name in aux_float_fields:
            val = getattr(self, field_name)
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
                or val < 0.0
            ):
                raise InvalidOrderInputException(
                    f"Order {field_name} must be finite scalar >= 0.0, got {val!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            setattr(self, field_name, float(val))

        # 7. Validate non-negative timestamps (int, non-boolean, >= 0)
        timestamp_fields = ("created_at_ns", "updated_at_ns")
        for ts_field in timestamp_fields:
            ts_val = getattr(self, ts_field)
            if not isinstance(ts_val, int) or isinstance(ts_val, bool) or ts_val < 0:
                raise InvalidOrderInputException(
                    f"Order {ts_field} must be integer >= 0, got {ts_val!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )

    @property
    def leaves_quantity(self) -> float:
        """Remaining unfilled order quantity under mass conservation max(0.0, quantity - filled_quantity)."""
        # Functional Purpose: Compute remaining unexecuted quantity under mass conservation INV-GW-003.
        # Explicit Dependency Tracking: self.quantity, self.filled_quantity.
        # Structural Relationship: Queried by OrderStateMachine and ExecutionGateway to determine terminal state.
        # Defensive Invariant: leaves_quantity is clamped to [0.0, quantity] via max(0.0, ...).
        return max(0.0, float(self.quantity - self.filled_quantity))

    @property
    def is_terminal(self) -> bool:
        """Whether the order has transitioned into an immutable terminal state."""
        # Functional Purpose: Provide deterministic check for order terminality under INV-GW-001.
        # Explicit Dependency Tracking: self.state, OrderState.
        # Structural Relationship: Checked by OrderStateMachine and ExecutionGateway before cancellation/fill.
        # Defensive Invariant: True only when state in {FILLED, CANCELLED, REJECTED, EXPIRED}.
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }

    @property
    def is_active(self) -> bool:
        """Whether the order is actively resting, pending, or subject to exchange matching."""
        # Functional Purpose: Provide deterministic check for in-flight active order status.
        # Explicit Dependency Tracking: self.state, OrderState.
        # Structural Relationship: Queried by IdempotencyRouter, pre-trade risk filters, and gateways.
        # Defensive Invariant: True only when state in {PENDING_NEW, NEW, PARTIALLY_FILLED, PENDING_CANCEL}.
        return self.state in {
            OrderState.PENDING_NEW,
            OrderState.NEW,
            OrderState.PARTIALLY_FILLED,
            OrderState.PENDING_CANCEL,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Immutable execution report emitted upon order matching, acknowledgment, or rejection.

    Attributes:
        report_id: Unique event identifier for audit traceability.
        cl_ord_id: Target order client identifier.
        exchange_order_id: Exchange-assigned order identifier.
        symbol: Traded instrument ticker.
        side: Trading direction (BUY or SELL).
        exec_type: Order lifecycle state resulting from this execution report.
        last_quantity: Quantity matched in this specific fill event (>= 0.0).
        last_price: Price matched in this specific fill event (>= 0.0).
        cum_quantity: Total cumulative filled quantity across all fills (>= 0.0).
        leaves_quantity: Remaining order quantity available for execution (>= 0.0).
        cum_quote_amount: Total cumulative quote consideration matched (>= 0.0).
        average_price: Volume-weighted average price across all fills (>= 0.0).
        fee: Transaction fee charged on this execution event (>= 0.0).
        timestamp_ns: Exchange or gateway report generation timestamp in epoch nanoseconds (>= 0).
        text: Human-readable diagnostic, rejection reason, or wire message (default: "").
    """

    report_id: str
    cl_ord_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    exec_type: OrderState
    last_quantity: float
    last_price: float
    cum_quantity: float
    leaves_quantity: float
    cum_quote_amount: float
    average_price: float
    fee: float
    timestamp_ns: int
    text: str = ""

    def __post_init__(self) -> None:
        """Validate immutability contracts, domain bounds, and non-finite guards upon instantiation."""
        # Functional Purpose: Enforce INV-GW-005 non-finite protection on immutable ExecutionReport value objects.
        # Explicit Dependency Tracking: math.isfinite, InvalidOrderInputException, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Consumed by OrderStateMachine, OrderAuditLogger, and strategy callbacks.
        # Defensive Invariant: All numeric fields must be strictly non-negative, finite, and non-boolean.

        # 1. Validate string identifiers
        if not isinstance(self.report_id, str) or not self.report_id:
            raise InvalidOrderInputException(
                f"ExecutionReport report_id must be a non-empty string, got {self.report_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.cl_ord_id, str) or not self.cl_ord_id:
            raise InvalidOrderInputException(
                f"ExecutionReport cl_ord_id must be a non-empty string, got {self.cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.exchange_order_id, str):
            raise InvalidOrderInputException(
                f"ExecutionReport exchange_order_id must be string, got {type(self.exchange_order_id).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.symbol, str) or not self.symbol:
            raise InvalidOrderInputException(
                f"ExecutionReport symbol must be a non-empty string, got {self.symbol!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.text, str):
            raise InvalidOrderInputException(
                f"ExecutionReport text must be string, got {type(self.text).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 2. Validate enum instances
        if not isinstance(self.side, OrderSide):
            raise InvalidOrderInputException(
                f"ExecutionReport side must be OrderSide enum instance, got {type(self.side).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(self.exec_type, OrderState):
            raise InvalidOrderInputException(
                f"ExecutionReport exec_type must be OrderState enum instance, got {type(self.exec_type).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 3. Validate non-negative finite float fields
        float_fields = (
            "last_quantity",
            "last_price",
            "cum_quantity",
            "leaves_quantity",
            "cum_quote_amount",
            "average_price",
            "fee",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
                or val < 0.0
            ):
                raise InvalidOrderInputException(
                    f"ExecutionReport {field_name} must be finite scalar >= 0.0, got {val!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            object.__setattr__(self, field_name, float(val))

        # 4. Validate timestamp
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise InvalidOrderInputException(
                f"ExecutionReport timestamp_ns must be integer >= 0, got {self.timestamp_ns!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
