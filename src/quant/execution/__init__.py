"""Execution Gateway Subsystem for Live Market Order Routing and Lifecycle Management.

Purpose:
    Exposes domain models, enums, exceptions, and diagnostic error codes for live execution,
    providing deterministic order state transitions, idempotency routing, and audit persistence.

Dependencies:
    - quant.execution.models: Domain entities, enums, exceptions, and error code constants.
    - quant.execution.fsm: Deterministic order state machine with causal out-of-order reconciliation.

Structural Relationship:
    - Root public export boundary for quant.execution package.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity)
    - INV-GW-002 (Cryptographic Idempotency Token Uniqueness)
    - INV-GW-003 (Execution Mass Conservation)
    - INV-GW-004 (Causal Out-of-Order Packet Reconciliation)
    - INV-GW-005 (Strict Non-Finite Input Protection)
"""

from quant.execution.fsm import OrderStateMachine
from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_DUPLICATE_ORDER_ID,
    ERR_GW_INSUFFICIENT_MARGIN,
    ERR_GW_INVALID_STATE_TRANSITION,
    ERR_GW_NON_FINITE_INPUT,
    ERR_GW_RATE_LIMIT_EXCEEDED,
    DuplicateOrderException,
    ExecutionReport,
    GatewayDisconnectedException,
    GatewayError,
    InsufficientMarginException,
    InvalidOrderInputException,
    InvalidStateTransitionException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    RateLimitExceededException,
    TimeInForce,
)

__all__ = [
    "ERR_GW_DISCONNECTED",
    "ERR_GW_DUPLICATE_ORDER_ID",
    "ERR_GW_INSUFFICIENT_MARGIN",
    "ERR_GW_INVALID_STATE_TRANSITION",
    "ERR_GW_NON_FINITE_INPUT",
    "ERR_GW_RATE_LIMIT_EXCEEDED",
    "DuplicateOrderException",
    "ExecutionReport",
    "GatewayDisconnectedException",
    "GatewayError",
    "InsufficientMarginException",
    "InvalidOrderInputException",
    "InvalidStateTransitionException",
    "Order",
    "OrderSide",
    "OrderStateMachine",
    "OrderState",
    "OrderType",
    "RateLimitExceededException",
    "TimeInForce",
]
