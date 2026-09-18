"""Institutional Emergency Panic Kill Switch and Multi-Gateway Mass Cancellation Engine.

Purpose:
    Implements the institutional emergency panic kill switch, asynchronous multi-gateway
    bulk cancellation engine, algorithmic scheduler freeze hooks, administrative reset
    mechanisms, and pre-trade submission lockout under Phase 6 Step 3 (Live Execution
    Quality & Safety Gateway).

Dependencies:
    - asyncio: Non-blocking coroutines, concurrency locks, and concurrent gather.
    - dataclasses: High-performance memory-compact slots and immutable event structures.
    - enum: Python 3.11+ StrEnum for zero-overhead string-compatible enumerations.
    - hmac: Constant-time cryptographic comparison (hmac.compare_digest) for admin tokens.
    - inspect: Dynamic inspection of synchronous vs asynchronous callback callables.
    - math: Strict non-finite scalar validation (math.isfinite).
    - time: High-resolution nanosecond epoch timestamps (time.time_ns).
    - typing: Static typing annotations, Final constants, and protocol interfaces.
    - quant.execution.gateway: ExecutionGateway protocol interface for broker connectivity.
    - quant.execution.models: Order domain entity, OrderSide, OrderType, and state enums.
    - quant.execution.risk: Diagnostic fault codes, KillSwitchActiveException, and RiskError.

Structural Relationship:
    - Root emergency circuit breaker for the execution subsystem.
    - Upstream Triggers: PreTradeRiskFirewall (drawdown/fat-finger), HeartbeatWatchdog
      (exchange disconnect), RiskOrchestrator, and Manual Operator API.
    - Downstream Targets:
        1. ExecutionGateways (cancels 100% of open resting orders across all venues).
        2. ExecutionScheduler / Schedulers (PoissonTWAP, VolumeAdaptiveVWAP, Arrival Price).
        3. PreTradeRiskFirewall (locks out 100% of subsequent order submissions).
        4. Strategy Orchestrator & Audit WAL (receives KillSwitchEvent telemetry).

Invariants Enforced:
    - INV-RSK-008 (Atomic Kill Switch Mass Cancellation): Upon panic trigger, atomically
      transitions state to PANIC_TRIGGERED, dispatches bulk cancellations across all
      registered gateways, invokes scheduler freeze hooks, and locks out order submissions
      with KillSwitchActiveException(ERR_RSK_KILL_SWITCH_ACTIVE) in < 50ms.
    - INV-RSK-006 (Hot-Path Latency SLA): validate_submission completes in < 10us in-memory.
    - INV-RSK-007 (Strict Input Sanitization): Immediate rejection of NaN, Inf, bool, and
      non-positive scalars.
    - Rule 1: Four-tier line annotations on every class, method, function, and property.
    - Rule 2: Catalog diagnostic fault codes (ERR_RSK_KILL_SWITCH_ACTIVE = "ERR-RSK-008").
    - Rule 4: Mandatory adversarial red-teaming: constant-time admin authentication,
      dual-phase concurrent cancellation, hung-socket timeouts, and transport error isolation.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from quant.execution.gateway import ExecutionGateway
from quant.execution.models import Order, OrderSide, OrderType
from quant.execution.risk import (
    ERR_RSK_KILL_SWITCH_ACTIVE,
    ERR_RSK_NON_FINITE_INPUT,
    KillSwitchActiveException,
    NonFiniteRiskInputException,
    RiskError,
)

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_KILL_SWITCH_ACTIVE: Final[str] = ERR_RSK_KILL_SWITCH_ACTIVE
ERR_INVALID_ADMIN_TOKEN: Final[str] = "ERR-RSK-009"
ERR_KILL_SWITCH_DISARMED: Final[str] = "ERR-RSK-010"


# ============================================================================
# Domain Enumerations (StrEnum)
# ============================================================================


class KillSwitchState(StrEnum):
    """Deterministic operational states of the emergency kill switch."""

    # Functional Purpose: Represent discrete lifecycle states for the safety gatekeeper.
    # Explicit Dependency Tracking: StrEnum base class.
    # Structural Relationship: Queried by validate_submission and state management methods.
    # Defensive Invariant: State must be one of ARMED_STANDBY, PANIC_TRIGGERED, or DISARMED.

    ARMED_STANDBY = "ARMED_STANDBY"
    PANIC_TRIGGERED = "PANIC_TRIGGERED"
    DISARMED = "DISARMED"


class PanicTriggerReason(StrEnum):
    """Institutional etiology and tripwire classification for emergency panic triggers."""

    # Functional Purpose: Categorize originating root cause for audit and compliance logging.
    # Explicit Dependency Tracking: StrEnum base class.
    # Structural Relationship: Encapsulated in KillSwitchEvent and transmitted to audit WAL.
    # Defensive Invariant: Must reflect legitimate operational, risk, or transport fault modes.

    MANUAL_OPERATOR = "MANUAL_OPERATOR"
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    GATEWAY_DISCONNECT = "GATEWAY_DISCONNECT"
    FAT_FINGER_BREACH = "FAT_FINGER_BREACH"
    ROGUE_FILLS = "ROGUE_FILLS"


class PanicMode(StrEnum):
    """Operational response posture executed upon emergency panic activation."""

    # Functional Purpose: Define whether to purely cancel orders or also liquidate inventory.
    # Explicit Dependency Tracking: StrEnum base class.
    # Structural Relationship: Configured per strategy or selected at trigger time.
    # Defensive Invariant: CANCEL_ONLY leaves positions intact; CANCEL_AND_FLATTEN closes positions.

    CANCEL_ONLY = "CANCEL_ONLY"
    CANCEL_AND_FLATTEN = "CANCEL_AND_FLATTEN"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Diagnostic Exception Protocol)
# ============================================================================


class InvalidAdminTokenException(RiskError, PermissionError):
    """Raised when an invalid administrative token is supplied for kill switch operations."""

    def __init__(
        self,
        message: str = "Invalid admin token provided for emergency kill switch operation.",
        code: str = ERR_INVALID_ADMIN_TOKEN,
    ) -> None:
        # Functional Purpose: Prevent unauthorized or forged administrative resets and disarms.
        # Explicit Dependency Tracking: ERR_INVALID_ADMIN_TOKEN diagnostic code.
        # Structural Relationship: Emitted by EmergencyKillSwitch.reset, disarm, and arm.
        # Defensive Invariant: Inherits from both RiskError and PermissionError.
        super().__init__(message=message, code=code)


class KillSwitchDisarmedException(RiskError):
    """Raised when an emergency panic trigger is attempted while the kill switch is disarmed."""

    def __init__(
        self,
        message: str = "Cannot trigger panic: Emergency kill switch is DISARMED.",
        code: str = ERR_KILL_SWITCH_DISARMED,
    ) -> None:
        # Functional Purpose: Guard against automated tripwires firing during maintenance disarm.
        # Explicit Dependency Tracking: ERR_KILL_SWITCH_DISARMED diagnostic code.
        # Structural Relationship: Emitted by EmergencyKillSwitch.trigger_panic when DISARMED.
        # Defensive Invariant: Signals operational lockout due to maintenance posture.
        super().__init__(message=message, code=code)


# ============================================================================
# Domain Value Objects (Rule 1 & Rule 4: Best-of-the-Best Contracts)
# ============================================================================


@dataclass(frozen=True, slots=True)
class KillSwitchEvent:
    """Immutable audit record of an emergency panic kill switch activation event.

    Attributes:
        timestamp_ns: Nanosecond timestamp when panic was triggered (finite int > 0).
        trigger_reason: Institutional etiology (MANUAL_OPERATOR, DRAWDOWN_BREACH, etc.).
        trigger_source: Originating subsystem or operator identifier (non-empty str).
        panic_mode: Operational posture (CANCEL_ONLY or CANCEL_AND_FLATTEN).
        cancelled_orders_count: Total active orders successfully cancelled across venues.
        positions_snapshot: Snapshot of portfolio instrument positions at trigger time.
        details: Diagnostic or contextual explanation of the trigger event.
    """

    timestamp_ns: int
    trigger_reason: PanicTriggerReason
    trigger_source: str
    panic_mode: PanicMode
    cancelled_orders_count: int
    positions_snapshot: dict[str, float]
    details: str

    def __post_init__(self) -> None:
        """Enforce domain invariants, non-finite guards, and type boundaries upon instantiation."""
        # Functional Purpose: Ensure immutable kill switch events have valid, sanitized parameters.
        # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException, ERR_RSK_NON_FINITE_INPUT.
        # Structural Relationship: Serialized to audit WAL and consumed by regulatory compliance logging.
        # Defensive Invariant: timestamp_ns positive int; non-negative cancelled_orders_count; valid enums.

        # 1. Validate timestamp_ns
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns <= 0
        ):
            raise NonFiniteRiskInputException(
                f"timestamp_ns must be positive integer, got {self.timestamp_ns!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        # 2. Validate trigger_reason
        if not isinstance(self.trigger_reason, PanicTriggerReason):
            if isinstance(self.trigger_reason, str):
                try:
                    object.__setattr__(
                        self, "trigger_reason", PanicTriggerReason(self.trigger_reason)
                    )
                except ValueError as err:
                    raise ValueError(
                        f"Invalid PanicTriggerReason: {self.trigger_reason!r}"
                    ) from err
            else:
                raise TypeError(
                    f"trigger_reason must be PanicTriggerReason, got {type(self.trigger_reason).__name__}"
                )

        # 3. Validate trigger_source
        if (
            isinstance(self.trigger_source, bool)
            or not isinstance(self.trigger_source, str)
            or not self.trigger_source.strip()
        ):
            raise ValueError(
                f"trigger_source must be non-empty string, got {self.trigger_source!r}"
            )

        # 4. Validate panic_mode
        if not isinstance(self.panic_mode, PanicMode):
            if isinstance(self.panic_mode, str):
                try:
                    object.__setattr__(self, "panic_mode", PanicMode(self.panic_mode))
                except ValueError as err:
                    raise ValueError(f"Invalid PanicMode: {self.panic_mode!r}") from err
            else:
                raise TypeError(
                    f"panic_mode must be PanicMode, got {type(self.panic_mode).__name__}"
                )

        # 5. Validate cancelled_orders_count
        if (
            isinstance(self.cancelled_orders_count, bool)
            or not isinstance(self.cancelled_orders_count, int)
            or self.cancelled_orders_count < 0
        ):
            raise NonFiniteRiskInputException(
                f"cancelled_orders_count must be non-negative int, got {self.cancelled_orders_count!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        # 6. Validate positions_snapshot
        if not isinstance(self.positions_snapshot, dict):
            raise TypeError(
                f"positions_snapshot must be dict, got {type(self.positions_snapshot).__name__}"
            )
        for sym, qty in self.positions_snapshot.items():
            if not isinstance(sym, str):
                raise TypeError(f"positions_snapshot symbol must be str, got {type(sym).__name__}")
            if isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(qty):
                raise NonFiniteRiskInputException(
                    f"positions_snapshot quantity for {sym} must be finite float, got {qty!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
        # Deep defensively clone positions dictionary to prevent external caller mutation
        object.__setattr__(self, "positions_snapshot", dict(self.positions_snapshot))

        # 7. Validate details
        if not isinstance(self.details, str):
            raise TypeError(f"details must be str, got {type(self.details).__name__}")


# ============================================================================
# Emergency Panic Kill Switch Engine (Rule 1, Rule 2, Rule 4)
# ============================================================================


class EmergencyKillSwitch:
    """Institutional emergency panic kill switch and multi-gateway mass cancellation engine.

    Enforces INV-RSK-008:
    1. Atomic state transition ARMED_STANDBY -> PANIC_TRIGGERED under mutex protection.
    2. Concurrent multi-gateway bulk order cancellation in < 50ms via asyncio.gather.
    3. Complete transport exception isolation across heterogeneous exchange venues.
    4. Algorithmic execution scheduler termination hooks.
    5. Hot-path pre-trade firewall submission lockout (< 10us).
    6. Cryptographic constant-time admin authorization for reset/arm/disarm.
    7. Idempotent multi-triggering under concurrent tripwire conditions.
    """

    __slots__ = (
        "_admin_token",
        "_gateway_timeout_seconds",
        "_gateways",
        "_last_event",
        "_lock",
        "_panic_listeners",
        "_scheduler_hooks",
        "_state",
    )

    def __init__(
        self,
        admin_token: str = "DEFAULT_ROOT_ADMIN_TOKEN_SECURE",
        initial_state: KillSwitchState = KillSwitchState.ARMED_STANDBY,
        gateway_timeout_seconds: float = 2.0,
    ) -> None:
        """Initialize emergency kill switch state, admin credential, and registries.

        Args:
            admin_token: Cryptographic administrator authentication token for reset/arm/disarm.
            initial_state: Initial operational state (default: KillSwitchState.ARMED_STANDBY).
            gateway_timeout_seconds: Maximum per-gateway socket timeout for cancellation (seconds).

        Raises:
            NonFiniteRiskInputException: If admin_token or gateway_timeout_seconds violates bounds.
            ValueError / TypeError: If initial_state is invalid.
        """
        # Functional Purpose: Establish safety gatekeeper state, authorization token, and gateway registry.
        # Explicit Dependency Tracking: KillSwitchState, NonFiniteRiskInputException, ERR_RSK_NON_FINITE_INPUT.
        # Structural Relationship: Core safety component registered in RiskOrchestrator.
        # Defensive Invariant: admin_token non-empty string; initial_state KillSwitchState; timeout > 0.
        if (
            isinstance(admin_token, bool)
            or not isinstance(admin_token, str)
            or not admin_token.strip()
        ):
            raise NonFiniteRiskInputException(
                f"admin_token must be non-empty string, got {admin_token!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if not isinstance(initial_state, KillSwitchState):
            if isinstance(initial_state, str):
                try:
                    initial_state = KillSwitchState(initial_state)
                except ValueError as err:
                    raise ValueError(f"Invalid KillSwitchState: {initial_state!r}") from err
            else:
                raise TypeError(
                    f"initial_state must be KillSwitchState, got {type(initial_state).__name__}"
                )

        if (
            isinstance(gateway_timeout_seconds, bool)
            or not isinstance(gateway_timeout_seconds, (int, float))
            or not math.isfinite(gateway_timeout_seconds)
            or gateway_timeout_seconds <= 0.0
        ):
            raise NonFiniteRiskInputException(
                f"gateway_timeout_seconds must be finite float > 0.0, got {gateway_timeout_seconds!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        self._admin_token: str = admin_token.strip()
        self._state: KillSwitchState = initial_state
        self._gateway_timeout_seconds: float = float(gateway_timeout_seconds)
        self._gateways: dict[str, ExecutionGateway] = {}
        self._scheduler_hooks: list[Callable[[], Awaitable[None] | None]] = []
        self._panic_listeners: list[Callable[[KillSwitchEvent], Awaitable[None] | None]] = []
        self._last_event: KillSwitchEvent | None = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Acquire or lazily initialize the asynchronous event loop mutex."""
        # Functional Purpose: Provide loop-safe asyncio.Lock initialization for panic concurrency.
        # Explicit Dependency Tracking: asyncio.Lock.
        # Structural Relationship: Used by trigger_panic to guard critical state transitions.
        # Defensive Invariant: Returns active asyncio.Lock bound to current event loop.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def state(self) -> KillSwitchState:
        """Current operational state of the emergency kill switch."""
        # Functional Purpose: Expose instantaneous kill switch lifecycle state.
        # Explicit Dependency Tracking: self._state.
        # Structural Relationship: Queried by health monitors, admin dashboards, and audit loggers.
        # Defensive Invariant: Returns valid KillSwitchState enum member.
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the kill switch is currently in the active PANIC_TRIGGERED state."""
        # Functional Purpose: Fast boolean predicate for hot-path order submission guards.
        # Explicit Dependency Tracking: self._state, KillSwitchState.PANIC_TRIGGERED.
        # Structural Relationship: Queried by PreTradeRiskFirewall and router hot path.
        # Defensive Invariant: True iff state is PANIC_TRIGGERED.
        return self._state == KillSwitchState.PANIC_TRIGGERED

    @property
    def is_armed(self) -> bool:
        """Whether the kill switch is currently in the ARMED_STANDBY state."""
        # Functional Purpose: Fast boolean predicate for operational readiness inspection.
        # Explicit Dependency Tracking: self._state, KillSwitchState.ARMED_STANDBY.
        # Structural Relationship: Queried by system startup health checks.
        # Defensive Invariant: True iff state is ARMED_STANDBY.
        return self._state == KillSwitchState.ARMED_STANDBY

    @property
    def is_disarmed(self) -> bool:
        """Whether the kill switch is currently in the DISARMED maintenance state."""
        # Functional Purpose: Fast boolean predicate for maintenance mode inspection.
        # Explicit Dependency Tracking: self._state, KillSwitchState.DISARMED.
        # Structural Relationship: Queried during manual maintenance windows.
        # Defensive Invariant: True iff state is DISARMED.
        return self._state == KillSwitchState.DISARMED

    @property
    def last_event(self) -> KillSwitchEvent | None:
        """Most recent emergency kill switch event, or None if never triggered."""
        # Functional Purpose: Expose immutable audit record of the latest panic execution.
        # Explicit Dependency Tracking: self._last_event.
        # Structural Relationship: Read by telemetry dashboards and post-mortem tools.
        # Defensive Invariant: Returns KillSwitchEvent or None; immutable value object.
        return self._last_event

    @property
    def gateways(self) -> dict[str, ExecutionGateway]:
        """Shallow copy of currently registered execution gateways."""
        # Functional Purpose: Inspect registered broker venues participating in mass cancellation.
        # Explicit Dependency Tracking: self._gateways.
        # Structural Relationship: Queried by router and health monitors.
        # Defensive Invariant: Returns dictionary copy protecting internal registry from mutation.
        return dict(self._gateways)

    @property
    def scheduler_hooks(self) -> list[Callable[[], Awaitable[None] | None]]:
        """Shallow copy of registered algorithmic scheduler hooks."""
        # Functional Purpose: Inspect registered scheduler cancellation hooks.
        # Explicit Dependency Tracking: self._scheduler_hooks.
        # Structural Relationship: Queried by execution diagnostic suites.
        # Defensive Invariant: Returns list copy protecting internal hooks registry.
        return list(self._scheduler_hooks)

    @property
    def panic_listeners(self) -> list[Callable[[KillSwitchEvent], Awaitable[None] | None]]:
        """Shallow copy of registered panic event notification listeners."""
        # Functional Purpose: Inspect registered panic event subscribers.
        # Explicit Dependency Tracking: self._panic_listeners.
        # Structural Relationship: Queried by notification management diagnostics.
        # Defensive Invariant: Returns list copy protecting internal listeners registry.
        return list(self._panic_listeners)

    def register_gateway(self, gateway: ExecutionGateway, gateway_id: str | None = None) -> str:
        """Register an execution gateway with the kill switch for mass order cancellation.

        Args:
            gateway: Object conforming to ExecutionGateway protocol.
            gateway_id: Optional unique string identifier for the gateway. If omitted,
                derives from gateway.gateway_id, gateway.venue_id, or gateway class name.

        Returns:
            Resolved unique string identifier under which the gateway was registered.

        Raises:
            TypeError: If gateway does not conform to ExecutionGateway protocol.
            ValueError: If gateway_id is an invalid string.
        """
        # Functional Purpose: Register an asynchronous execution gateway to participate in panic sweeps.
        # Explicit Dependency Tracking: ExecutionGateway, self._gateways.
        # Structural Relationship: Called during system startup or dynamic venue routing registration.
        # Defensive Invariant: gateway must implement ExecutionGateway protocol; returns unique gateway_id.
        if not isinstance(gateway, ExecutionGateway):
            raise TypeError(
                f"Expected ExecutionGateway protocol implementation, got {type(gateway).__name__}"
            )

        if gateway_id is not None:
            if (
                isinstance(gateway_id, bool)
                or not isinstance(gateway_id, str)
                or not gateway_id.strip()
            ):
                raise ValueError(f"gateway_id must be a non-empty string, got {gateway_id!r}")
            gid = gateway_id.strip()
        else:
            if (
                hasattr(gateway, "gateway_id")
                and isinstance(gateway.gateway_id, str)
                and gateway.gateway_id.strip()
            ):
                gid = gateway.gateway_id.strip()
            elif (
                hasattr(gateway, "venue_id")
                and isinstance(gateway.venue_id, str)
                and gateway.venue_id.strip()
            ):
                gid = gateway.venue_id.strip()
            else:
                gid = f"{gateway.__class__.__name__}_{id(gateway)}"

        self._gateways[gid] = gateway
        return gid

    def unregister_gateway(self, gateway_id: str) -> None:
        """Unregister an execution gateway by its identifier.

        Args:
            gateway_id: Unique string identifier of the gateway to unregister.

        Raises:
            ValueError: If gateway_id is empty or invalid.
            KeyError: If gateway_id is not found in the registry.
        """
        # Functional Purpose: Remove a decommissioned gateway from participating in cancellation sweeps.
        # Explicit Dependency Tracking: self._gateways.
        # Structural Relationship: Invoked when a venue is decommissioned or gateway session terminates.
        # Defensive Invariant: gateway_id must be non-empty string; raises KeyError if not found.
        if (
            isinstance(gateway_id, bool)
            or not isinstance(gateway_id, str)
            or not gateway_id.strip()
        ):
            raise ValueError(f"gateway_id must be a non-empty string, got {gateway_id!r}")
        gid = gateway_id.strip()
        if gid not in self._gateways:
            raise KeyError(f"Gateway '{gid}' is not registered.")
        del self._gateways[gid]

    def register_scheduler_hook(self, hook: Callable[[], Awaitable[None] | None]) -> None:
        """Register an algorithmic scheduler cancellation hook to freeze parent order slicers.

        Args:
            hook: Sync or async callable taking no arguments that halts algorithmic schedulers.

        Raises:
            TypeError: If hook is not callable.
        """
        # Functional Purpose: Register callback to freeze algorithmic schedulers (TWAP, VWAP, Arrival Price).
        # Explicit Dependency Tracking: self._scheduler_hooks.
        # Structural Relationship: Invoked by ExecutionScheduler, SOR, or Strategy Orchestrator.
        # Defensive Invariant: hook must be callable.
        if not callable(hook):
            raise TypeError(f"hook must be callable, got {type(hook).__name__}")
        self._scheduler_hooks.append(hook)

    def register_panic_listener(
        self, listener: Callable[[KillSwitchEvent], Awaitable[None] | None]
    ) -> None:
        """Register a subscriber callback notified whenever emergency panic is triggered.

        Args:
            listener: Sync or async callable accepting a KillSwitchEvent instance.

        Raises:
            TypeError: If listener is not callable.
        """
        # Functional Purpose: Notify downstream risk monitors, alerting webhooks, and audit loggers.
        # Explicit Dependency Tracking: self._panic_listeners.
        # Structural Relationship: Invoked by risk dashboards, audit loggers, and notification dispatchers.
        # Defensive Invariant: listener must be callable.
        if not callable(listener):
            raise TypeError(f"listener must be callable, got {type(listener).__name__}")
        self._panic_listeners.append(listener)

    def _verify_admin_token(self, token: object) -> bool:
        """Verify administrator authorization token using constant-time comparison."""
        # Functional Purpose: Prevent timing side-channel attacks during administrative privilege elevation.
        # Explicit Dependency Tracking: hmac.compare_digest.
        # Structural Relationship: Invoked by reset, disarm, and arm state transition controls.
        # Defensive Invariant: Rejects booleans and non-strings; performs constant-time byte comparison.
        if isinstance(token, bool) or not isinstance(token, str) or not token:
            return False
        return hmac.compare_digest(token.encode("utf-8"), self._admin_token.encode("utf-8"))

    def validate_submission(self) -> None:
        """Verify that the emergency kill switch is not currently in PANIC_TRIGGERED state.

        Raises:
            KillSwitchActiveException: If the kill switch is currently in PANIC_TRIGGERED state.
        """
        # Functional Purpose: Gatekeeper intercepting outbound order submissions in the hot path (< 10us).
        # Explicit Dependency Tracking: self._state, KillSwitchState.PANIC_TRIGGERED, ERR_RSK_KILL_SWITCH_ACTIVE.
        # Structural Relationship: Called by PreTradeRiskFirewall and RiskOrchestrator prior to routing.
        # Defensive Invariant: Raises KillSwitchActiveException(ERR_RSK_KILL_SWITCH_ACTIVE) if PANIC_TRIGGERED.
        if self._state == KillSwitchState.PANIC_TRIGGERED:
            reason_str = self._last_event.trigger_reason.value if self._last_event else "UNKNOWN"
            source_str = self._last_event.trigger_source if self._last_event else "UNKNOWN"
            raise KillSwitchActiveException(
                f"Order submission blocked: Emergency Kill Switch is ACTIVE "
                f"(reason={reason_str}, source={source_str})",
                code=ERR_RSK_KILL_SWITCH_ACTIVE,
            )

    def reset(self, admin_token: str) -> None:
        """Reset emergency kill switch from PANIC_TRIGGERED back to ARMED_STANDBY.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Administrative re-arming allowing order flow resumption after incident review.
        # Explicit Dependency Tracking: self._verify_admin_token, KillSwitchState.ARMED_STANDBY.
        # Structural Relationship: Invoked by Chief Risk Officer / Operator console.
        # Defensive Invariant: Constant-time authentication check; transitions state to ARMED_STANDBY.
        if not self._verify_admin_token(admin_token):
            raise InvalidAdminTokenException(
                "Invalid admin token provided for emergency kill switch reset."
            )
        self._state = KillSwitchState.ARMED_STANDBY

    def disarm(self, admin_token: str) -> None:
        """Disarm emergency kill switch into DISARMED maintenance state.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Administrative suspension of panic tripwires for planned maintenance.
        # Explicit Dependency Tracking: self._verify_admin_token, KillSwitchState.DISARMED.
        # Structural Relationship: Invoked during offline testing or exchange maintenance windows.
        # Defensive Invariant: Constant-time authentication check; transitions state to DISARMED.
        if not self._verify_admin_token(admin_token):
            raise InvalidAdminTokenException(
                "Invalid admin token provided for emergency kill switch disarm."
            )
        self._state = KillSwitchState.DISARMED

    def arm(self, admin_token: str) -> None:
        """Arm emergency kill switch into ARMED_STANDBY monitoring state.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Re-activate kill switch protection after maintenance disarm.
        # Explicit Dependency Tracking: self._verify_admin_token, KillSwitchState.ARMED_STANDBY.
        # Structural Relationship: Invoked when transitioning from maintenance back to live trading.
        # Defensive Invariant: Constant-time authentication check; transitions state to ARMED_STANDBY.
        if not self._verify_admin_token(admin_token):
            raise InvalidAdminTokenException(
                "Invalid admin token provided for emergency kill switch arm."
            )
        self._state = KillSwitchState.ARMED_STANDBY

    async def _execute_scheduler_hooks(self) -> None:
        """Execute all registered algorithmic scheduler cancellation hooks with exception isolation."""
        # Functional Purpose: Freeze parent order slicers and execution schedulers under emergency panic.
        # Explicit Dependency Tracking: self._scheduler_hooks, inspect.isawaitable.
        # Structural Relationship: Called by trigger_panic prior to gateway cancellations.
        # Defensive Invariant: Isolates all hook exceptions so that individual callback failures do not abort panic.
        for hook in self._scheduler_hooks:
            try:
                res = hook()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                # Defensive isolation: a failing hook must never prevent order cancellation or lockdown
                pass

    async def _execute_gateway_cancellations(self) -> tuple[int, dict[str, float]]:
        """Execute concurrent multi-gateway open order cancellations and snapshot positions."""
        # Functional Purpose: Query open orders and broadcast cancellation requests across all venues concurrently.
        # Explicit Dependency Tracking: asyncio.gather, asyncio.wait_for, ExecutionGateway.
        # Structural Relationship: Core cancellation engine fulfilling INV-RSK-008 sub-50ms SLA.
        # Defensive Invariant: Isolates gateway transport exceptions and timeout errors; returns (count, positions).
        if not self._gateways:
            return 0, {}

        # Phase 1: Concurrently fetch open orders and positions across all gateways
        async def _fetch_gw_data(
            gw_id: str, gw: ExecutionGateway
        ) -> tuple[str, list[Order] | Exception, dict[str, float] | Exception]:
            async def _get_orders() -> list[Order]:
                return await gw.get_open_orders()

            async def _get_pos() -> dict[str, float]:
                return await gw.get_positions()

            orders_res: list[Order] | Exception
            try:
                orders_res = await asyncio.wait_for(
                    _get_orders(), timeout=self._gateway_timeout_seconds
                )
            except Exception as e:
                orders_res = e

            pos_res: dict[str, float] | Exception
            try:
                pos_res = await asyncio.wait_for(_get_pos(), timeout=self._gateway_timeout_seconds)
            except Exception as e:
                pos_res = e

            return gw_id, orders_res, pos_res

        fetch_tasks = [_fetch_gw_data(gw_id, gw) for gw_id, gw in self._gateways.items()]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Phase 2: Dispatch cancellations for ALL open orders across ALL gateways concurrently
        cancel_coros = []
        positions_snapshot: dict[str, float] = {}

        for item in fetch_results:
            if isinstance(item, Exception) or not isinstance(item, tuple):
                continue
            gw_id, orders_res, pos_res = item

            # Aggregate positions snapshot
            if isinstance(pos_res, dict):
                for sym, qty in pos_res.items():
                    if (
                        isinstance(qty, (int, float))
                        and math.isfinite(qty)
                        and not isinstance(qty, bool)
                    ):
                        positions_snapshot[sym] = positions_snapshot.get(sym, 0.0) + float(qty)

            # Build parallel cancellation tasks
            if isinstance(orders_res, list):
                gw = self._gateways.get(gw_id)
                if gw is not None:
                    for order in orders_res:
                        if hasattr(order, "cl_ord_id"):

                            async def _cancel_one(g: ExecutionGateway, cid: str) -> None:
                                await asyncio.wait_for(
                                    g.cancel_order(cid),
                                    timeout=self._gateway_timeout_seconds,
                                )

                            cancel_coros.append(_cancel_one(gw, order.cl_ord_id))

        successful_cancellations = 0
        if cancel_coros:
            cancel_results = await asyncio.gather(*cancel_coros, return_exceptions=True)
            for res in cancel_results:
                if not isinstance(res, Exception):
                    successful_cancellations += 1

        return successful_cancellations, positions_snapshot

    async def _flatten_positions(self, positions_snapshot: dict[str, float]) -> int:
        """Attempt emergency position liquidation by submitting market offsetting orders."""
        # Functional Purpose: Liquidate inventory under PanicMode.CANCEL_AND_FLATTEN.
        # Explicit Dependency Tracking: Order, OrderSide, OrderType, asyncio.gather.
        # Structural Relationship: Invoked conditionally when panic_mode is CANCEL_AND_FLATTEN.
        # Defensive Invariant: Submits opposing MARKET orders; catches and isolates all submission errors.
        flatten_tasks = []
        now_ns = time.time_ns()
        counter = 0

        for gw_id, gw in self._gateways.items():
            try:
                gw_positions = await asyncio.wait_for(
                    gw.get_positions(), timeout=self._gateway_timeout_seconds
                )
            except Exception:
                continue

            for sym, pos_qty in gw_positions.items():
                if (
                    isinstance(pos_qty, (int, float))
                    and math.isfinite(pos_qty)
                    and not isinstance(pos_qty, bool)
                    and abs(pos_qty) > 1e-7
                ):
                    counter += 1
                    side = OrderSide.SELL if pos_qty > 0.0 else OrderSide.BUY
                    qty = abs(pos_qty)
                    cl_ord_id = f"panic-flat-{gw_id}-{sym}-{counter}-{now_ns}"
                    flat_order = Order(
                        cl_ord_id=cl_ord_id,
                        symbol=sym,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        created_at_ns=now_ns,
                        updated_at_ns=now_ns,
                    )

                    async def _submit_flat(g: ExecutionGateway, o: Order) -> None:
                        await asyncio.wait_for(
                            g.submit_order(o), timeout=self._gateway_timeout_seconds
                        )

                    flatten_tasks.append(_submit_flat(gw, flat_order))

        flattened_count = 0
        if flatten_tasks:
            flatten_results = await asyncio.gather(*flatten_tasks, return_exceptions=True)
            for res in flatten_results:
                if not isinstance(res, Exception):
                    flattened_count += 1
        return flattened_count

    async def _notify_listeners(self, event: KillSwitchEvent) -> None:
        """Dispatch panic event notifications to all registered subscriber callbacks."""
        # Functional Purpose: Alert downstream risk monitors and event telemetry of panic trigger.
        # Explicit Dependency Tracking: self._panic_listeners, inspect.isawaitable.
        # Structural Relationship: Called at conclusion of trigger_panic after lockdown is active.
        # Defensive Invariant: Catches and suppresses listener exceptions to prevent broadcast failures.
        for listener in self._panic_listeners:
            try:
                res = listener(event)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

    async def trigger_panic(
        self,
        reason: PanicTriggerReason | str,
        source: str,
        panic_mode: PanicMode | str = PanicMode.CANCEL_ONLY,
        details: str = "",
        current_timestamp_ns: int | None = None,
    ) -> KillSwitchEvent:
        """Trigger emergency panic: cancel all open orders, freeze schedulers, lock firewall.

        Args:
            reason: Trigger cause (PanicTriggerReason or string value).
            source: Initiating subsystem, tripwire, or operator identifier.
            panic_mode: Response posture (CANCEL_ONLY or CANCEL_AND_FLATTEN).
            details: Contextual explanatory message.
            current_timestamp_ns: Explicit epoch nanosecond timestamp (default: time.time_ns()).

        Returns:
            Immutable KillSwitchEvent recording the execution of the panic protocol.

        Raises:
            KillSwitchDisarmedException: If called while kill switch is DISARMED.
            ValueError / TypeError: If input parameters violate boundary types.
            NonFiniteRiskInputException: If timestamp is non-finite or non-positive.
        """
        # Functional Purpose: Execute atomic mass cancellation and order lockout under INV-RSK-008.
        # Explicit Dependency Tracking: asyncio.Lock, time.time_ns, KillSwitchEvent, KillSwitchActiveException.
        # Structural Relationship: Called by PreTradeRiskFirewall, HeartbeatWatchdog, or Manual Operator.
        # Defensive Invariant: Sub-50ms SLA; atomic idempotency under mutex; isolates all gateway transport errors.

        # 1. Validate reason
        if isinstance(reason, PanicTriggerReason):
            parsed_reason = reason
        elif isinstance(reason, str):
            try:
                parsed_reason = PanicTriggerReason(reason)
            except ValueError as err:
                raise ValueError(f"Invalid PanicTriggerReason: {reason!r}") from err
        else:
            raise TypeError(
                f"reason must be PanicTriggerReason or str, got {type(reason).__name__}"
            )

        # 2. Validate source
        if isinstance(source, bool) or not isinstance(source, str) or not source.strip():
            raise ValueError(f"source must be non-empty string, got {source!r}")
        clean_source = source.strip()

        # 3. Validate panic_mode
        if isinstance(panic_mode, PanicMode):
            parsed_mode = panic_mode
        elif isinstance(panic_mode, str):
            try:
                parsed_mode = PanicMode(panic_mode)
            except ValueError as err:
                raise ValueError(f"Invalid PanicMode: {panic_mode!r}") from err
        else:
            raise TypeError(f"panic_mode must be PanicMode or str, got {type(panic_mode).__name__}")

        # 4. Validate details
        if not isinstance(details, str):
            raise TypeError(f"details must be str, got {type(details).__name__}")

        # 5. Validate timestamp
        if current_timestamp_ns is None:
            ts = time.time_ns()
        else:
            if (
                isinstance(current_timestamp_ns, bool)
                or not isinstance(current_timestamp_ns, int)
                or current_timestamp_ns <= 0
            ):
                raise NonFiniteRiskInputException(
                    f"current_timestamp_ns must be positive integer, got {current_timestamp_ns!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            ts = current_timestamp_ns

        # 6. Concurrency lock protecting the entire panic cancellation lifecycle
        async with self._get_lock():
            if self._state == KillSwitchState.DISARMED:
                raise KillSwitchDisarmedException(
                    "Cannot trigger panic: Emergency kill switch is currently DISARMED."
                )

            if self._state == KillSwitchState.PANIC_TRIGGERED:
                # Idempotent re-trigger: return existing active event without redundant cancellation storm
                assert self._last_event is not None
                return self._last_event

            # Atomic transition to lock out order submissions immediately
            self._state = KillSwitchState.PANIC_TRIGGERED

            # 7. Execute scheduler freeze hooks
            await self._execute_scheduler_hooks()

            # 8. Multi-gateway mass cancellation sweep
            cancelled_count, positions_snapshot = await self._execute_gateway_cancellations()

            # 9. Optional emergency flatten execution
            if parsed_mode == PanicMode.CANCEL_AND_FLATTEN:
                await self._flatten_positions(positions_snapshot)

            # 10. Record immutable event
            event = KillSwitchEvent(
                timestamp_ns=ts,
                trigger_reason=parsed_reason,
                trigger_source=clean_source,
                panic_mode=parsed_mode,
                cancelled_orders_count=cancelled_count,
                positions_snapshot=positions_snapshot,
                details=details.strip(),
            )
            self._last_event = event

            # 11. Notify listeners
            await self._notify_listeners(event)

            return event
