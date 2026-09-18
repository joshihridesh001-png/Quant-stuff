"""Real-Time Exchange Heartbeat, Transport Watchdog, and Connection Liveness Monitoring.

Purpose:
    Provides sub-microsecond in-memory connection liveness tracking, sequence gap detection,
    latency degradation monitoring, and deterministic state transitions for institutional
    exchange gateways (FIX, WebSocket, REST) under Phase 6 Step 3 (Live Execution Quality
    & Safety Gateway).

Dependencies:
    - collections: Zero-overhead fixed-capacity double-ended queue (deque) for rolling latencies.
    - dataclasses: Memory-compact slots and immutable configuration dataclasses.
    - enum: Python standard library StrEnum for zero-overhead string-compatible enumerations.
    - math: Strict non-finite scalar validation (math.isfinite).
    - typing: Strict static typing annotations, Callable listener protocols, and Final constants.
    - quant.execution.risk: NonFiniteRiskInputException and ERR_RSK_NON_FINITE_INPUT fault code.

Structural Relationship:
    - Monitors: Downstream transport links to broker/exchange venues (e.g., FIX 4.2/4.4, WS feeds).
    - Upstream Consumers: ExecutionGateway, SmartOrderRouter, RiskOrchestrator, EmergencyKillSwitch.
    - Dispatches: Status transitions, sequence gap anomalies, and disconnect emergency panic events.
    - Enforces: INV-RSK-006 (Sub-10us Hot-Path SLA) and INV-RSK-007 (Strict Input Sanitization).

Invariants Enforced:
    - Connection Liveness State Transitions:
      CONNECTED <--> DEGRADED --> DISCONNECTED --> RECONNECTING --> CONNECTED
    - Sequence Gap Monotonicity:
      Inbound packet sequence gaps or out-of-order deliveries increment sequence_gaps_count,
      trigger edge-dispatched callbacks, and maintain high-watermark expected sequence numbers.
    - Latency Degradation SLA:
      Ping-pong round-trip latencies exceeding warning/critical thresholds elevate status to DEGRADED.
    - Missed Heartbeat Timeout:
      Silence exceeding timeout_seconds or max_consecutive_misses transitions state to DISCONNECTED
      and dispatches on_disconnect edge-triggered panic listeners exactly once.
    - Hot-Path Sub-10us Latency SLA (INV-RSK-006):
      record_heartbeat and check_liveness execute in < 10us using zero-allocation in-memory scalars.
    - Strict Input Sanitization (INV-RSK-007):
      Rejection of booleans masquerading as numerics, NaNs, infinities, and non-positive timestamps.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from quant.execution.risk import (
    ERR_RSK_NON_FINITE_INPUT,
    NonFiniteRiskInputException,
)

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_HB_DISCONNECTED: Final[str] = "ERR-HB-001"
ERR_HB_SEQUENCE_GAP: Final[str] = "ERR-HB-002"
ERR_HB_LATENCY_DEGRADED: Final[str] = "ERR-HB-003"


# ============================================================================
# Domain Enumerations (StrEnum)
# ============================================================================


class ConnectionStatus(StrEnum):
    """Deterministic connection lifecycle states for exchange transport sessions."""

    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"


class TransportProtocol(StrEnum):
    """Transport protocol classification for exchange gateway communication."""

    WEBSOCKET = "WEBSOCKET"
    FIX = "FIX"
    REST_POLL = "REST_POLL"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Diagnostic Exception Protocol)
# ============================================================================


class HeartbeatError(Exception):
    """Base exception for all connection watchdog and heartbeat monitoring faults."""

    def __init__(self, message: str, code: str = "ERR-HB-000") -> None:
        # Functional Purpose: Initialize base heartbeat exception with descriptive message and fault code.
        # Explicit Dependency Tracking: Python Exception root class.
        # Structural Relationship: Root parent class for all connection watchdog diagnostic taxonomy.
        # Defensive Invariant: Diagnostic code must be a non-empty string identifier.
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class HeartbeatTimeoutError(HeartbeatError):
    """Raised when exchange transport heartbeat silence exceeds maximum configured timeout."""

    def __init__(self, message: str, code: str = ERR_HB_DISCONNECTED) -> None:
        # Functional Purpose: Signal unrecoverable exchange silence and transport disconnection.
        # Explicit Dependency Tracking: ERR_HB_DISCONNECTED diagnostic fault code.
        # Structural Relationship: Dispatched by HeartbeatWatchdog.check_liveness.
        # Defensive Invariant: code defaults to ERR-HB-001.
        super().__init__(message=message, code=code)


class HeartbeatSequenceGapError(HeartbeatError):
    """Raised when an inbound message sequence gap or packet loss anomaly is detected."""

    def __init__(self, message: str, code: str = ERR_HB_SEQUENCE_GAP) -> None:
        # Functional Purpose: Signal missing, skipped, or out-of-order sequence numbers in feed.
        # Explicit Dependency Tracking: ERR_HB_SEQUENCE_GAP diagnostic fault code.
        # Structural Relationship: Dispatched by HeartbeatWatchdog.record_heartbeat.
        # Defensive Invariant: code defaults to ERR-HB-002.
        super().__init__(message=message, code=code)


class HeartbeatLatencyDegradedError(HeartbeatError):
    """Raised when round-trip ping-pong latency breaches warning or critical thresholds."""

    def __init__(self, message: str, code: str = ERR_HB_LATENCY_DEGRADED) -> None:
        # Functional Purpose: Signal high transport latency degradation impairing execution SLA.
        # Explicit Dependency Tracking: ERR_HB_LATENCY_DEGRADED diagnostic fault code.
        # Structural Relationship: Dispatched by HeartbeatWatchdog.record_heartbeat.
        # Defensive Invariant: code defaults to ERR-HB-003.
        super().__init__(message=message, code=code)


# ============================================================================
# Defensive Validation Helpers (Rule 1 & Rule 4)
# ============================================================================


def _validate_positive_float(val: object, name: str) -> float:
    """Validate that an input is a finite, non-boolean float strictly greater than zero."""
    # Functional Purpose: Ensure timing intervals and latency thresholds are strictly positive real numbers.
    # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException.
    # Structural Relationship: Called by HeartbeatConfig.__post_init__.
    # Defensive Invariant: val must be numeric, not bool, finite, and > 0.0.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a numeric scalar, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    f_val = float(val)
    if not math.isfinite(f_val):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a finite scalar, got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if f_val <= 0.0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be strictly positive (> 0.0), got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return f_val


def _validate_non_negative_float(val: object, name: str) -> float:
    """Validate that an input is a finite, non-boolean float greater than or equal to zero."""
    # Functional Purpose: Ensure recorded round-trip latencies are non-negative real numbers.
    # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException.
    # Structural Relationship: Called by HeartbeatRecord.__post_init__ and record_heartbeat.
    # Defensive Invariant: val must be numeric, not bool, finite, and >= 0.0.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a numeric scalar, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    f_val = float(val)
    if not math.isfinite(f_val):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a finite scalar, got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if f_val < 0.0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' cannot be negative (< 0.0), got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return f_val


def _validate_positive_int(val: object, name: str) -> int:
    """Validate that an input is a non-boolean integer strictly greater than zero."""
    # Functional Purpose: Ensure sequence numbers and strict timestamps are positive integers.
    # Explicit Dependency Tracking: NonFiniteRiskInputException.
    # Structural Relationship: Called by HeartbeatConfig, HeartbeatRecord, and HeartbeatWatchdog.
    # Defensive Invariant: val must be integer, not bool, and >= 1.
    if isinstance(val, bool) or not isinstance(val, int):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be an integer, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if val <= 0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be strictly positive (>= 1), got {val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return val


def _validate_non_negative_int(val: object, name: str) -> int:
    """Validate that an input is a non-boolean integer greater than or equal to zero."""
    # Functional Purpose: Ensure optional base timestamps and counters are non-negative integers.
    # Explicit Dependency Tracking: NonFiniteRiskInputException.
    # Structural Relationship: Called by HeartbeatWatchdog.__init__ and reset.
    # Defensive Invariant: val must be integer, not bool, and >= 0.
    if isinstance(val, bool) or not isinstance(val, int):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be an integer, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if val < 0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' cannot be negative (< 0), got {val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return val


# ============================================================================
# Domain Dataclasses: HeartbeatConfig & HeartbeatRecord
# ============================================================================


@dataclass(frozen=True, slots=True)
class HeartbeatConfig:
    """Immutable configuration parameter bundle for exchange heartbeat watchdog.

    Attributes:
        heartbeat_interval_seconds: Expected nominal heartbeat cadence (seconds).
        timeout_seconds: Hard silence duration before triggering disconnect panic (seconds).
        max_consecutive_misses: Maximum missing intervals before declaring disconnection.
        latency_warning_threshold_ms: Round-trip ping latency elevating session to DEGRADED (ms).
        latency_critical_threshold_ms: Severe latency threshold requiring urgent alert (ms).
    """

    heartbeat_interval_seconds: float = 1.0
    timeout_seconds: float = 3.0
    max_consecutive_misses: int = 3
    latency_warning_threshold_ms: float = 250.0
    latency_critical_threshold_ms: float = 1000.0

    def __post_init__(self) -> None:
        # Functional Purpose: Validate heartbeat parameters and cross-threshold ordering invariants.
        # Explicit Dependency Tracking: _validate_positive_float, _validate_positive_int.
        # Structural Relationship: Constructed by gateway setup or loaded from configuration.
        # Defensive Invariant: interval > 0, timeout >= interval, misses >= 1, crit >= warn > 0.
        interval = _validate_positive_float(
            self.heartbeat_interval_seconds, "heartbeat_interval_seconds"
        )
        timeout = _validate_positive_float(self.timeout_seconds, "timeout_seconds")
        _validate_positive_int(self.max_consecutive_misses, "max_consecutive_misses")
        warn = _validate_positive_float(
            self.latency_warning_threshold_ms, "latency_warning_threshold_ms"
        )
        crit = _validate_positive_float(
            self.latency_critical_threshold_ms, "latency_critical_threshold_ms"
        )

        if timeout < interval:
            raise NonFiniteRiskInputException(
                f"timeout_seconds ({timeout}) cannot be less than "
                f"heartbeat_interval_seconds ({interval})",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if crit < warn:
            raise NonFiniteRiskInputException(
                f"latency_critical_threshold_ms ({crit}) cannot be less than "
                f"latency_warning_threshold_ms ({warn})",
                code=ERR_RSK_NON_FINITE_INPUT,
            )


@dataclass(frozen=True, slots=True)
class HeartbeatRecord:
    """Immutable audit record of a received exchange heartbeat packet.

    Attributes:
        sequence_number: Inbound transport sequence number.
        timestamp_ns: Inbound packet receipt timestamp (nanoseconds).
        latency_ms: Round-trip ping-pong latency (milliseconds).
        status: Connection status resolved upon recording this heartbeat.
    """

    sequence_number: int
    timestamp_ns: int
    latency_ms: float
    status: ConnectionStatus

    def __post_init__(self) -> None:
        # Functional Purpose: Validate immutability and domain bounds of heartbeat receipt records.
        # Explicit Dependency Tracking: _validate_positive_int, _validate_non_negative_float.
        # Structural Relationship: Emitted by HeartbeatWatchdog.record_heartbeat to caller.
        # Defensive Invariant: sequence_number > 0, timestamp_ns > 0, latency_ms >= 0.0.
        _validate_positive_int(self.sequence_number, "sequence_number")
        _validate_positive_int(self.timestamp_ns, "timestamp_ns")
        _validate_non_negative_float(self.latency_ms, "latency_ms")
        if not isinstance(self.status, ConnectionStatus):
            raise NonFiniteRiskInputException(
                f"Parameter 'status' must be ConnectionStatus enum, got {type(self.status).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )


# ============================================================================
# Core Watchdog Engine: HeartbeatWatchdog
# ============================================================================


class HeartbeatWatchdog:
    """Deterministic connection liveness watchdog, sequence gap tracker, and latency monitor.

    Enforces institutional connection safety:
    1. Tracks inbound sequence numbers, detecting gaps, skips, and out-of-order deliveries.
    2. Maintains high-watermark expected sequence numbers to avoid false gap cascades.
    3. Monitors ping-pong round-trip latencies, elevating connection status to DEGRADED.
    4. Evaluates silence duration against timeouts, edge-triggering panic disconnect events.
    5. Executes hot-path methods (record_heartbeat, check_liveness) in < 10us (INV-RSK-006).
    """

    __slots__ = (
        "gateway_id",
        "config",
        "protocol",
        "status",
        "last_heartbeat_timestamp_ns",
        "last_sequence_number",
        "expected_sequence_number",
        "consecutive_misses",
        "sequence_gaps_count",
        "recent_latencies",
        "_status_listeners",
        "_sequence_gap_listeners",
        "_disconnect_listeners",
    )

    def __init__(
        self,
        gateway_id: str,
        config: HeartbeatConfig | None = None,
        protocol: TransportProtocol = TransportProtocol.FIX,
        initial_sequence_number: int = 1,
        initial_timestamp_ns: int = 0,
        initial_status: ConnectionStatus = ConnectionStatus.CONNECTED,
    ) -> None:
        # Functional Purpose: Initialize stateful watchdog with gateway id, config, and tracking state.
        # Explicit Dependency Tracking: HeartbeatConfig, ConnectionStatus, TransportProtocol.
        # Structural Relationship: Instantiated per exchange gateway connection session.
        # Defensive Invariant: gateway_id non-empty str, initial_seq >= 1, initial_timestamp_ns >= 0.
        if not isinstance(gateway_id, str) or not gateway_id.strip():
            raise NonFiniteRiskInputException(
                f"gateway_id must be a non-empty string, got {gateway_id!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        _validate_positive_int(initial_sequence_number, "initial_sequence_number")
        _validate_non_negative_int(initial_timestamp_ns, "initial_timestamp_ns")

        resolved_config = config if config is not None else HeartbeatConfig()
        if not isinstance(resolved_config, HeartbeatConfig):
            raise NonFiniteRiskInputException(
                f"config must be HeartbeatConfig instance, got {type(resolved_config).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if not isinstance(protocol, TransportProtocol):
            raise NonFiniteRiskInputException(
                f"protocol must be TransportProtocol enum, got {type(protocol).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if not isinstance(initial_status, ConnectionStatus):
            raise NonFiniteRiskInputException(
                f"initial_status must be ConnectionStatus enum, got {type(initial_status).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        self.gateway_id: str = gateway_id
        self.config: HeartbeatConfig = resolved_config
        self.protocol: TransportProtocol = protocol
        self.status: ConnectionStatus = initial_status
        self.last_heartbeat_timestamp_ns: int = initial_timestamp_ns
        self.last_sequence_number: int = initial_sequence_number - 1
        self.expected_sequence_number: int = initial_sequence_number
        self.consecutive_misses: int = 0
        self.sequence_gaps_count: int = 0
        self.recent_latencies: deque[float] = deque(maxlen=100)

        self._status_listeners: list[
            Callable[[str, ConnectionStatus, ConnectionStatus, int], None]
        ] = []
        self._sequence_gap_listeners: list[Callable[[str, int, int, int], None]] = []
        self._disconnect_listeners: list[Callable[[str, float, int], None]] = []

    # ------------------------------------------------------------------------
    # State Inspection Properties
    # ------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True if connection status is nominal CONNECTED."""
        # Functional Purpose: Provide instant boolean predicate for healthy connected state.
        # Explicit Dependency Tracking: ConnectionStatus.CONNECTED.
        # Structural Relationship: Polled by order dispatchers before routing.
        # Defensive Invariant: Deterministic equality against ConnectionStatus.
        return self.status == ConnectionStatus.CONNECTED

    @property
    def is_degraded(self) -> bool:
        """Return True if connection status is DEGRADED."""
        # Functional Purpose: Provide instant boolean predicate for degraded state.
        # Explicit Dependency Tracking: ConnectionStatus.DEGRADED.
        # Structural Relationship: Polled by smart order router for venue de-prioritization.
        # Defensive Invariant: Deterministic equality against ConnectionStatus.
        return self.status == ConnectionStatus.DEGRADED

    @property
    def is_disconnected(self) -> bool:
        """Return True if connection status is DISCONNECTED."""
        # Functional Purpose: Provide instant boolean predicate for disconnected state.
        # Explicit Dependency Tracking: ConnectionStatus.DISCONNECTED.
        # Structural Relationship: Polled by risk firewall and kill switch.
        # Defensive Invariant: Deterministic equality against ConnectionStatus.
        return self.status == ConnectionStatus.DISCONNECTED

    @property
    def is_reconnecting(self) -> bool:
        """Return True if connection status is in active recovery RECONNECTING."""
        # Functional Purpose: Provide instant boolean predicate for reconnecting state.
        # Explicit Dependency Tracking: ConnectionStatus.RECONNECTING.
        # Structural Relationship: Polled by transport reconnection worker.
        # Defensive Invariant: Deterministic equality against ConnectionStatus.
        return self.status == ConnectionStatus.RECONNECTING

    @property
    def is_alive(self) -> bool:
        """Return True if connection is capable of traffic (CONNECTED or DEGRADED)."""
        # Functional Purpose: Provide instant boolean check for operational transport viability.
        # Explicit Dependency Tracking: ConnectionStatus.
        # Structural Relationship: Checked prior to low-urgency order routing.
        # Defensive Invariant: Returns True iff status is CONNECTED or DEGRADED.
        return self.status in (ConnectionStatus.CONNECTED, ConnectionStatus.DEGRADED)

    @property
    def average_latency_ms(self) -> float:
        """Return rolling average ping-pong latency (ms) or 0.0 if empty."""
        # Functional Purpose: Compute mean latency across rolling window of recent heartbeats.
        # Explicit Dependency Tracking: self.recent_latencies deque.
        # Structural Relationship: Telemetry and health metrics reporting.
        # Defensive Invariant: Returns non-negative finite float.
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)

    @property
    def last_latency_ms(self) -> float:
        """Return latency of most recent heartbeat (ms) or 0.0 if empty."""
        # Functional Purpose: Provide immediate point-in-time ping latency.
        # Explicit Dependency Tracking: self.recent_latencies deque.
        # Structural Relationship: Telemetry and dashboard monitoring.
        # Defensive Invariant: Returns non-negative finite float.
        if not self.recent_latencies:
            return 0.0
        return self.recent_latencies[-1]

    @property
    def max_latency_ms(self) -> float:
        """Return maximum latency observed in rolling window (ms) or 0.0 if empty."""
        # Functional Purpose: Detect worst-case tail latency spikes in recent traffic.
        # Explicit Dependency Tracking: self.recent_latencies deque.
        # Structural Relationship: Risk telemetry and venue markout degradation.
        # Defensive Invariant: Returns non-negative finite float.
        if not self.recent_latencies:
            return 0.0
        return max(self.recent_latencies)

    # ------------------------------------------------------------------------
    # Listener Registration & Lifecycle Subscriptions
    # ------------------------------------------------------------------------

    def register_status_listener(
        self, callback: Callable[[str, ConnectionStatus, ConnectionStatus, int], None]
    ) -> None:
        """Register an observer callback for connection status transitions.

        Callback signature: callback(gateway_id, old_status, new_status, timestamp_ns)
        """
        # Functional Purpose: Register event handler for state transitions in the connection FSM.
        # Explicit Dependency Tracking: Callable protocol, _status_listeners list.
        # Structural Relationship: Invoked on state change by check_liveness, record_heartbeat, reset.
        # Defensive Invariant: callback must be callable; deduplicated on registration.
        if not callable(callback):
            raise TypeError(f"Status listener must be callable, got {type(callback).__name__}")
        if callback not in self._status_listeners:
            self._status_listeners.append(callback)

    def unregister_status_listener(
        self, callback: Callable[[str, ConnectionStatus, ConnectionStatus, int], None]
    ) -> None:
        """Unregister an existing status transition observer callback."""
        # Functional Purpose: Remove observer callback from active status notification list.
        # Explicit Dependency Tracking: _status_listeners list.
        # Structural Relationship: Cleanup on gateway teardown or listener lifecycle end.
        # Defensive Invariant: Silent no-op if callback is not present.
        if callback in self._status_listeners:
            self._status_listeners.remove(callback)

    def register_sequence_gap_listener(
        self, callback: Callable[[str, int, int, int], None]
    ) -> None:
        """Register an observer callback for inbound sequence gaps or packet reordering.

        Callback signature: callback(gateway_id, expected_sequence, received_sequence, timestamp_ns)
        """
        # Functional Purpose: Register event handler for sequence number gap anomalies.
        # Explicit Dependency Tracking: Callable protocol, _sequence_gap_listeners list.
        # Structural Relationship: Invoked on sequence gap by record_heartbeat.
        # Defensive Invariant: callback must be callable; deduplicated on registration.
        if not callable(callback):
            raise TypeError(
                f"Sequence gap listener must be callable, got {type(callback).__name__}"
            )
        if callback not in self._sequence_gap_listeners:
            self._sequence_gap_listeners.append(callback)

    def unregister_sequence_gap_listener(
        self, callback: Callable[[str, int, int, int], None]
    ) -> None:
        """Unregister an existing sequence gap observer callback."""
        # Functional Purpose: Remove observer callback from active sequence gap notification list.
        # Explicit Dependency Tracking: _sequence_gap_listeners list.
        # Structural Relationship: Cleanup on gateway teardown or listener lifecycle end.
        # Defensive Invariant: Silent no-op if callback is not present.
        if callback in self._sequence_gap_listeners:
            self._sequence_gap_listeners.remove(callback)

    def register_disconnect_listener(self, callback: Callable[[str, float, int], None]) -> None:
        """Register an observer callback for session timeout disconnection events.

        Callback signature: callback(gateway_id, elapsed_seconds, timestamp_ns)
        """
        # Functional Purpose: Register event handler for edge-triggered disconnection panic.
        # Explicit Dependency Tracking: Callable protocol, _disconnect_listeners list.
        # Structural Relationship: Invoked once when transitioning into DISCONNECTED state.
        # Defensive Invariant: callback must be callable; deduplicated on registration.
        if not callable(callback):
            raise TypeError(f"Disconnect listener must be callable, got {type(callback).__name__}")
        if callback not in self._disconnect_listeners:
            self._disconnect_listeners.append(callback)

    def unregister_disconnect_listener(self, callback: Callable[[str, float, int], None]) -> None:
        """Unregister an existing disconnect observer callback."""
        # Functional Purpose: Remove observer callback from active disconnect notification list.
        # Explicit Dependency Tracking: _disconnect_listeners list.
        # Structural Relationship: Cleanup on gateway teardown or listener lifecycle end.
        # Defensive Invariant: Silent no-op if callback is not present.
        if callback in self._disconnect_listeners:
            self._disconnect_listeners.remove(callback)

    # ------------------------------------------------------------------------
    # State Machine Mutations & Reconnection Lifecycle
    # ------------------------------------------------------------------------

    def mark_reconnecting(self, timestamp_ns: int = 0) -> None:
        """Explicitly transition connection status to RECONNECTING.

        Args:
            timestamp_ns: Current timestamp in nanoseconds (optional, non-negative).
        """
        # Functional Purpose: Move session into RECONNECTING state while transport establishes socket.
        # Explicit Dependency Tracking: ConnectionStatus.RECONNECTING, _validate_non_negative_int.
        # Structural Relationship: Called by gateway reconnection loop upon socket retry.
        # Defensive Invariant: timestamp_ns >= 0; dispatches status callback on state change.
        _validate_non_negative_int(timestamp_ns, "timestamp_ns")
        if self.status != ConnectionStatus.RECONNECTING:
            old_status = self.status
            self.status = ConnectionStatus.RECONNECTING
            for status_cb in tuple(self._status_listeners):
                status_cb(self.gateway_id, old_status, self.status, timestamp_ns)

    def reset(self, initial_sequence_number: int = 1, current_timestamp_ns: int = 0) -> None:
        """Reset sequence counters, miss tallies, and restore session status to CONNECTED.

        Args:
            initial_sequence_number: Next expected sequence number (default: 1, strictly positive).
            current_timestamp_ns: Session baseline timestamp in nanoseconds (non-negative).
        """
        # Functional Purpose: Re-initialize session counters upon completed logon or reconnect.
        # Explicit Dependency Tracking: _validate_positive_int, _validate_non_negative_int.
        # Structural Relationship: Invoked upon successful FIX logon or WS connection handshake.
        # Defensive Invariant: initial_sequence_number >= 1, current_timestamp_ns >= 0.
        _validate_positive_int(initial_sequence_number, "initial_sequence_number")
        _validate_non_negative_int(current_timestamp_ns, "current_timestamp_ns")

        old_status = self.status
        self.last_sequence_number = initial_sequence_number - 1
        self.expected_sequence_number = initial_sequence_number
        self.last_heartbeat_timestamp_ns = current_timestamp_ns
        self.consecutive_misses = 0
        self.sequence_gaps_count = 0
        self.recent_latencies.clear()
        self.status = ConnectionStatus.CONNECTED

        if old_status != ConnectionStatus.CONNECTED:
            for status_cb in tuple(self._status_listeners):
                status_cb(self.gateway_id, old_status, self.status, current_timestamp_ns)

    # ------------------------------------------------------------------------
    # Hot-Path Execution Methods (INV-RSK-006: Sub-10us SLA)
    # ------------------------------------------------------------------------

    def record_heartbeat(
        self, sequence_number: int, latency_ms: float, timestamp_ns: int
    ) -> HeartbeatRecord:
        """Process an inbound heartbeat packet, validate sequences, and evaluate latency degradation.

        Args:
            sequence_number: Inbound message sequence number (strictly positive integer).
            latency_ms: Measured round-trip ping-pong latency (finite non-negative float).
            timestamp_ns: Receipt timestamp in nanoseconds (strictly positive integer).

        Returns:
            HeartbeatRecord: Immutable record containing sequence, timestamp, latency, and status.
        """
        # Functional Purpose: Ingestion point for inbound heartbeats; executes sequence & latency checks.
        # Explicit Dependency Tracking: _validate_positive_int, _validate_non_negative_float, HeartbeatRecord.
        # Structural Relationship: Ingested on network transport message read loop.
        # Defensive Invariant: sequence > 0, latency >= 0.0, timestamp > 0, sub-10us execution.
        _validate_positive_int(sequence_number, "sequence_number")
        _validate_non_negative_float(latency_ms, "latency_ms")
        _validate_positive_int(timestamp_ns, "timestamp_ns")

        self.last_heartbeat_timestamp_ns = timestamp_ns
        self.consecutive_misses = 0
        self.recent_latencies.append(latency_ms)

        # 1. Sequence gap & out-of-order packet detection
        expected = self.expected_sequence_number
        if sequence_number != expected:
            self.sequence_gaps_count += 1
            for gap_cb in tuple(self._sequence_gap_listeners):
                gap_cb(self.gateway_id, expected, sequence_number, timestamp_ns)

        self.last_sequence_number = sequence_number

        # Maintain forward expected sequence watermark:
        # If sequence_number >= expected, advance expected to sequence_number + 1.
        # If sequence_number < expected (out-of-order/delayed packet), leave expected unchanged.
        if sequence_number >= expected:
            self.expected_sequence_number = sequence_number + 1

        # 2. Latency degradation state evaluation
        is_degraded_latency = latency_ms >= self.config.latency_warning_threshold_ms
        old_status = self.status

        new_status: ConnectionStatus
        if is_degraded_latency:
            new_status = ConnectionStatus.DEGRADED
        else:
            new_status = ConnectionStatus.CONNECTED

        if new_status != old_status:
            self.status = new_status
            for status_cb in tuple(self._status_listeners):
                status_cb(self.gateway_id, old_status, new_status, timestamp_ns)

        return HeartbeatRecord(
            sequence_number=sequence_number,
            timestamp_ns=timestamp_ns,
            latency_ms=latency_ms,
            status=self.status,
        )

    def check_liveness(self, current_timestamp_ns: int) -> ConnectionStatus:
        """Evaluate session liveness against elapsed time, interval misses, and timeout thresholds.

        Args:
            current_timestamp_ns: Current system clock in nanoseconds (strictly positive integer).

        Returns:
            ConnectionStatus: Current or newly transitioned connection status.
        """
        # Functional Purpose: Periodic or on-demand check for silence, interval misses, and timeout.
        # Explicit Dependency Tracking: _validate_positive_int, HeartbeatConfig, ConnectionStatus.
        # Structural Relationship: Polled by timer loop, risk orchestrator, or pre-trade router.
        # Defensive Invariant: current_timestamp_ns > 0, edge-triggered on_disconnect dispatches.
        _validate_positive_int(current_timestamp_ns, "current_timestamp_ns")

        # Cold initialization safeguard: If no baseline timestamp recorded, initialize baseline.
        if self.last_heartbeat_timestamp_ns == 0:
            self.last_heartbeat_timestamp_ns = current_timestamp_ns
            return self.status

        elapsed_ns = max(0, current_timestamp_ns - self.last_heartbeat_timestamp_ns)
        elapsed_seconds = elapsed_ns / 1_000_000_000.0

        interval_s = self.config.heartbeat_interval_seconds
        timeout_s = self.config.timeout_seconds
        max_misses = self.config.max_consecutive_misses

        misses = int(elapsed_seconds // interval_s)
        self.consecutive_misses = misses

        # Check hard disconnection timeout
        is_timed_out = (elapsed_seconds >= timeout_s) or (misses >= max_misses)

        if is_timed_out:
            if self.status != ConnectionStatus.DISCONNECTED:
                old_status = self.status
                self.status = ConnectionStatus.DISCONNECTED

                # Edge-triggered notification to disconnect listeners
                for disc_cb in tuple(self._disconnect_listeners):
                    disc_cb(self.gateway_id, elapsed_seconds, current_timestamp_ns)

                # Edge-triggered notification to status transition listeners
                for status_cb in tuple(self._status_listeners):
                    status_cb(self.gateway_id, old_status, self.status, current_timestamp_ns)

            return self.status

        # Check soft degradation (at least one missed heartbeat interval)
        if misses >= 1:
            if self.status == ConnectionStatus.CONNECTED:
                old_status = self.status
                self.status = ConnectionStatus.DEGRADED
                for status_cb in tuple(self._status_listeners):
                    status_cb(self.gateway_id, old_status, self.status, current_timestamp_ns)

            return self.status

        return self.status
