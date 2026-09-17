"""Execution Gateway and Smart Order Routing Subsystem for Live Market Order Lifecycle Management.

Purpose:
    Exposes domain models, enums, exceptions, diagnostic error codes, gateway protocols,
    simulated paper broker engines, microstructural venue profiles, and consolidated NBBO
    quotes for live and simulated execution lifecycle management.

Dependencies:
    - quant.execution.models: Gateway domain entities, enums, exceptions, and error code constants.
    - quant.execution.fsm: Deterministic order state machine with causal out-of-order reconciliation.
    - quant.execution.idempotency: Deterministic cryptographic idempotency router and ring buffer.
    - quant.execution.gateway: ExecutionGateway protocol and PaperExecutionGateway.
    - quant.execution.audit: Non-blocking asynchronous SQLite WAL order audit logger.
    - quant.execution.venues: VenueProfile, ConsolidatedQuote, VenueType, and SORError hierarchy.

Structural Relationship:
    - Root public export boundary for quant.execution package.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity)
    - INV-GW-002 (Cryptographic Idempotency Token Uniqueness)
    - INV-GW-003 (Execution Mass Conservation)
    - INV-GW-004 (Causal Out-of-Order Packet Reconciliation)
    - INV-GW-005 (Strict Non-Finite Input Protection)
    - INV-GW-006 (Hot-Path Latency SLA)
    - INV-SOR-001 (Parent-Child Mass Conservation)
    - INV-SOR-002 (Non-Worse-Than-NBBO & Dark Midpoint Price Improvement)
    - INV-SOR-005 (Strict Non-Finite Input Protection in Routing)
"""

from quant.execution.algorithms import (
    ExecutionScheduler,
    NonlinearArrivalPriceScheduler,
    PoissonTWAPScheduler,
    ScheduledSlice,
    VolumeAdaptiveVWAPScheduler,
)
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
from quant.execution.venues import (
    ERR_SOR_ALGORITHM_TIMEOUT,
    ERR_SOR_CHILD_ORDER_FAILED,
    ERR_SOR_INSUFFICIENT_LIQUIDITY,
    ERR_SOR_INVALID_SCHEDULE,
    ERR_SOR_MASS_CONSERVATION_BREACH,
    ERR_SOR_NBBO_VIOLATION,
    ERR_SOR_NON_FINITE_INPUT,
    AlgorithmTimeoutException,
    ChildOrderFailedException,
    ConsolidatedQuote,
    InsufficientLiquidityException,
    InvalidScheduleException,
    InvalidSORInputException,
    MassConservationException,
    NBBOViolationException,
    NonFiniteInputException,
    SORError,
    VenueProfile,
    VenueType,
)

__all__ = [
    "ERR_GW_DISCONNECTED",
    "ERR_GW_DUPLICATE_ORDER_ID",
    "ERR_GW_INSUFFICIENT_MARGIN",
    "ERR_GW_INVALID_STATE_TRANSITION",
    "ERR_GW_NON_FINITE_INPUT",
    "ERR_GW_RATE_LIMIT_EXCEEDED",
    "ERR_SOR_ALGORITHM_TIMEOUT",
    "ERR_SOR_CHILD_ORDER_FAILED",
    "ERR_SOR_INSUFFICIENT_LIQUIDITY",
    "ERR_SOR_INVALID_SCHEDULE",
    "ERR_SOR_MASS_CONSERVATION_BREACH",
    "ERR_SOR_NBBO_VIOLATION",
    "ERR_SOR_NON_FINITE_INPUT",
    "AlgorithmTimeoutException",
    "ChildOrderFailedException",
    "ConsolidatedQuote",
    "DuplicateOrderException",
    "ExecutionGateway",
    "ExecutionReport",
    "ExecutionScheduler",
    "GatewayDisconnectedException",
    "GatewayError",
    "IdempotencyRouter",
    "InsufficientLiquidityException",
    "InsufficientMarginException",
    "InvalidOrderInputException",
    "InvalidSORInputException",
    "InvalidScheduleException",
    "InvalidStateTransitionException",
    "MassConservationException",
    "NAMESPACE_QUANT_ORDER",
    "NBBOViolationException",
    "NonFiniteInputException",
    "NonlinearArrivalPriceScheduler",
    "Order",
    "OrderAuditLogger",
    "OrderSide",
    "OrderStateMachine",
    "OrderState",
    "OrderType",
    "PaperExecutionGateway",
    "PoissonTWAPScheduler",
    "RateLimitExceededException",
    "SORError",
    "ScheduledSlice",
    "TimeInForce",
    "VenueProfile",
    "VenueType",
    "VolumeAdaptiveVWAPScheduler",
]
