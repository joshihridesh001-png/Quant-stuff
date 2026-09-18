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
    - quant.execution.risk: PreTradeRiskFirewall, RiskLimits, PortfolioRiskState, and RiskError hierarchy.
    - quant.execution.heartbeat: HeartbeatWatchdog, HeartbeatConfig, HeartbeatRecord, ConnectionStatus, TransportProtocol, and HeartbeatError hierarchy.

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
    - INV-RSK-001 (Single-Order Fat-Finger Bounds)
    - INV-RSK-002 (Portfolio Leverage Bounds)
    - INV-RSK-003 (Single-Asset NAV Concentration)
    - INV-RSK-004 (Intraday Drawdown Circuit Breaker)
    - INV-RSK-005 (Margin & Borrow Sufficiency)
    - INV-RSK-006 (Sub-10us Hot-Path Latency SLA)
    - INV-RSK-007 (Strict Non-Finite Input Sanitization)
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
from quant.execution.heartbeat import (
    ERR_HB_DISCONNECTED,
    ERR_HB_LATENCY_DEGRADED,
    ERR_HB_SEQUENCE_GAP,
    ConnectionStatus,
    HeartbeatConfig,
    HeartbeatError,
    HeartbeatLatencyDegradedError,
    HeartbeatRecord,
    HeartbeatSequenceGapError,
    HeartbeatTimeoutError,
    HeartbeatWatchdog,
    TransportProtocol,
)
from quant.execution.idempotency import (
    NAMESPACE_QUANT_ORDER,
    IdempotencyRouter,
)
from quant.execution.kill_switch import (
    ERR_INVALID_ADMIN_TOKEN,
    ERR_KILL_SWITCH_DISARMED,
    EmergencyKillSwitch,
    InvalidAdminTokenException,
    KillSwitchDisarmedException,
    KillSwitchEvent,
    KillSwitchState,
    PanicMode,
    PanicTriggerReason,
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
from quant.execution.parent_order import (
    ChildFillRecord,
    ImplementationShortfallReport,
    ParentOrder,
)
from quant.execution.risk import (
    ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED,
    ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED,
    ERR_RSK_FAT_FINGER_NOTIONAL,
    ERR_RSK_FAT_FINGER_QUANTITY,
    ERR_RSK_INSUFFICIENT_MARGIN,
    ERR_RSK_KILL_SWITCH_ACTIVE,
    ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
    ERR_RSK_NON_FINITE_INPUT,
    ConcentrationLimitExceededException,
    DrawdownLimitExceededException,
    FatFingerNotionalException,
    FatFingerQuantityException,
    InsufficientMarginRiskException,
    KillSwitchActiveException,
    LeverageLimitExceededException,
    NonFiniteRiskInputException,
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskError,
    RiskLimits,
)
from quant.execution.risk_orchestrator import (
    ERR_ORCHESTRATOR_DISCONNECTED,
    ERR_ORCHESTRATOR_DRAWDOWN,
    ERR_ORCHESTRATOR_KILL_ACTIVE,
    ERR_ORCHESTRATOR_NON_FINITE,
    RiskOrchestrator,
)
from quant.execution.sor import (
    RoutedVenueOrder,
    SmartOrderRouter,
    VenueHealth,
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
    "ERR_HB_DISCONNECTED",
    "ERR_HB_LATENCY_DEGRADED",
    "ERR_HB_SEQUENCE_GAP",
    "ERR_INVALID_ADMIN_TOKEN",
    "ERR_KILL_SWITCH_DISARMED",
    "ERR_ORCHESTRATOR_DISCONNECTED",
    "ERR_ORCHESTRATOR_DRAWDOWN",
    "ERR_ORCHESTRATOR_KILL_ACTIVE",
    "ERR_ORCHESTRATOR_NON_FINITE",
    "ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED",
    "ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED",
    "ERR_RSK_FAT_FINGER_NOTIONAL",
    "ERR_RSK_FAT_FINGER_QUANTITY",
    "ERR_RSK_INSUFFICIENT_MARGIN",
    "ERR_RSK_KILL_SWITCH_ACTIVE",
    "ERR_RSK_LEVERAGE_LIMIT_EXCEEDED",
    "ERR_RSK_NON_FINITE_INPUT",
    "ERR_SOR_ALGORITHM_TIMEOUT",
    "ERR_SOR_CHILD_ORDER_FAILED",
    "ERR_SOR_INSUFFICIENT_LIQUIDITY",
    "ERR_SOR_INVALID_SCHEDULE",
    "ERR_SOR_MASS_CONSERVATION_BREACH",
    "ERR_SOR_NBBO_VIOLATION",
    "ERR_SOR_NON_FINITE_INPUT",
    "AlgorithmTimeoutException",
    "ChildFillRecord",
    "ChildOrderFailedException",
    "ConcentrationLimitExceededException",
    "ConnectionStatus",
    "ConsolidatedQuote",
    "DrawdownLimitExceededException",
    "DuplicateOrderException",
    "EmergencyKillSwitch",
    "ExecutionGateway",
    "ExecutionReport",
    "ExecutionScheduler",
    "FatFingerNotionalException",
    "FatFingerQuantityException",
    "GatewayDisconnectedException",
    "GatewayError",
    "HeartbeatConfig",
    "HeartbeatError",
    "HeartbeatLatencyDegradedError",
    "HeartbeatRecord",
    "HeartbeatSequenceGapError",
    "HeartbeatTimeoutError",
    "HeartbeatWatchdog",
    "IdempotencyRouter",
    "ImplementationShortfallReport",
    "InsufficientLiquidityException",
    "InsufficientMarginException",
    "InsufficientMarginRiskException",
    "InvalidAdminTokenException",
    "InvalidOrderInputException",
    "InvalidSORInputException",
    "InvalidScheduleException",
    "InvalidStateTransitionException",
    "KillSwitchActiveException",
    "KillSwitchDisarmedException",
    "KillSwitchEvent",
    "KillSwitchState",
    "LeverageLimitExceededException",
    "MassConservationException",
    "NAMESPACE_QUANT_ORDER",
    "NBBOViolationException",
    "NonFiniteInputException",
    "NonFiniteRiskInputException",
    "NonlinearArrivalPriceScheduler",
    "Order",
    "OrderAuditLogger",
    "OrderSide",
    "OrderStateMachine",
    "OrderState",
    "OrderType",
    "PanicMode",
    "PanicTriggerReason",
    "PaperExecutionGateway",
    "ParentOrder",
    "PoissonTWAPScheduler",
    "PortfolioRiskState",
    "PreTradeRiskFirewall",
    "RateLimitExceededException",
    "RiskError",
    "RiskLimits",
    "RiskOrchestrator",
    "RoutedVenueOrder",
    "SORError",
    "ScheduledSlice",
    "SmartOrderRouter",
    "TimeInForce",
    "TransportProtocol",
    "VenueHealth",
    "VenueProfile",
    "VenueType",
    "VolumeAdaptiveVWAPScheduler",
]
