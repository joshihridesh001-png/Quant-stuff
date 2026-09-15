"""Execution Gateway Subsystem for Live Market Order Routing and Lifecycle Management.

Purpose:
    Exposes domain models, enums, exceptions, diagnostic error codes, gateway protocols,
    and simulated paper broker engines for live and simulated execution lifecycle management.

Dependencies:
    - quant.execution.models: Domain entities, enums, exceptions, and error code constants.
    - quant.execution.fsm: Deterministic order state machine with causal out-of-order reconciliation.
    - quant.execution.idempotency: Deterministic cryptographic idempotency router and ring buffer.
    - quant.execution.gateway: ExecutionGateway protocol and PaperExecutionGateway.
    - quant.execution.audit: Non-blocking asynchronous SQLite WAL order audit logger.

Structural Relationship:
    - Root public export boundary for quant.execution package.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity)
    - INV-GW-002 (Cryptographic Idempotency Token Uniqueness)
    - INV-GW-003 (Execution Mass Conservation)
    - INV-GW-004 (Causal Out-of-Order Packet Reconciliation)
    - INV-GW-005 (Strict Non-Finite Input Protection)
    - INV-GW-006 (Hot-Path Latency SLA)
"""

from quant.execution.audit import OrderAuditLogger
from quant.execution.fsm import OrderStateMachine
from quant.execution.gateway import (
    ExecutionGateway,
    PaperExecutionGateway,
)
from quant.execution.idempotency import (
    NAMESPACE_QUANT_ORDER,
    IdempotencyRouter,
)
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
    "ExecutionGateway",
    "ExecutionReport",
    "GatewayDisconnectedException",
    "GatewayError",
    "IdempotencyRouter",
    "InsufficientMarginException",
    "InvalidOrderInputException",
    "InvalidStateTransitionException",
    "NAMESPACE_QUANT_ORDER",
    "Order",
    "OrderAuditLogger",
    "OrderSide",
    "OrderStateMachine",
    "OrderState",
    "OrderType",
    "PaperExecutionGateway",
    "RateLimitExceededException",
    "TimeInForce",
]
