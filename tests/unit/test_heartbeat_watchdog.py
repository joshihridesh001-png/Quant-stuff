"""Comprehensive unit tests for Exchange Heartbeat & Connection Watchdog.

Governing Standards:
- Rules.md:
  - Rule 1: Defensive Invariants and explicit contract testing.
  - Rule 2: Zero-execution deterministic diagnostic codes (ERR-HB-001..003, ERR-RSK-007).
  - Rule 3: Quality gates (100% pass rate, strict typing, >= 90% statement coverage).
  - Rule 4: Mandatory adversarial red-teaming, non-finite/bool guards, sub-10us SLA.
- Invariants:
  - INV-RSK-006: Hot-Path Latency SLA (< 10us)
  - INV-RSK-007: Strict Non-Finite & Boolean Input Sanitization
  - Connection Liveness FSM: CONNECTED <-> DEGRADED -> DISCONNECTED -> RECONNECTING -> CONNECTED
"""

from __future__ import annotations

import math
import sys
import time

import pytest

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
from quant.execution.risk import (
    ERR_RSK_NON_FINITE_INPUT,
    NonFiniteRiskInputException,
)

# ============================================================================
# Section 1: Enum & Exception Hierarchy Tests
# ============================================================================


def test_connection_status_values() -> None:
    """Verify ConnectionStatus enum members and string compatibility."""
    assert ConnectionStatus.CONNECTED == "CONNECTED"
    assert ConnectionStatus.DEGRADED == "DEGRADED"
    assert ConnectionStatus.DISCONNECTED == "DISCONNECTED"
    assert ConnectionStatus.RECONNECTING == "RECONNECTING"


def test_transport_protocol_values() -> None:
    """Verify TransportProtocol enum members and string compatibility."""
    assert TransportProtocol.WEBSOCKET == "WEBSOCKET"
    assert TransportProtocol.FIX == "FIX"
    assert TransportProtocol.REST_POLL == "REST_POLL"


def test_heartbeat_exception_hierarchy() -> None:
    """Verify exception hierarchy, inheritance, and fault codes."""
    base_err = HeartbeatError("base fault", code="ERR-HB-000")
    assert isinstance(base_err, Exception)
    assert base_err.code == "ERR-HB-000"
    assert base_err.message == "base fault"

    timeout_err = HeartbeatTimeoutError("timeout fault")
    assert isinstance(timeout_err, HeartbeatError)
    assert timeout_err.code == ERR_HB_DISCONNECTED

    gap_err = HeartbeatSequenceGapError("sequence gap")
    assert isinstance(gap_err, HeartbeatError)
    assert gap_err.code == ERR_HB_SEQUENCE_GAP

    degraded_err = HeartbeatLatencyDegradedError("latency degraded")
    assert isinstance(degraded_err, HeartbeatError)
    assert degraded_err.code == ERR_HB_LATENCY_DEGRADED


# ============================================================================
# Section 2: HeartbeatConfig Validation & Boundary Defenses
# ============================================================================


def test_heartbeat_config_defaults() -> None:
    """Verify nominal HeartbeatConfig default values."""
    config = HeartbeatConfig()
    assert config.heartbeat_interval_seconds == 1.0
    assert config.timeout_seconds == 3.0
    assert config.max_consecutive_misses == 3
    assert config.latency_warning_threshold_ms == 250.0
    assert config.latency_critical_threshold_ms == 1000.0


def test_heartbeat_config_custom_valid() -> None:
    """Verify customized HeartbeatConfig parameters."""
    config = HeartbeatConfig(
        heartbeat_interval_seconds=0.5,
        timeout_seconds=2.0,
        max_consecutive_misses=4,
        latency_warning_threshold_ms=100.0,
        latency_critical_threshold_ms=500.0,
    )
    assert config.heartbeat_interval_seconds == 0.5
    assert config.timeout_seconds == 2.0
    assert config.max_consecutive_misses == 4
    assert config.latency_warning_threshold_ms == 100.0
    assert config.latency_critical_threshold_ms == 500.0


@pytest.mark.parametrize(
    "bad_interval",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "1.0", None],
)
def test_heartbeat_config_rejects_invalid_interval(bad_interval: object) -> None:
    """Verify HeartbeatConfig rejects non-positive, non-finite, and boolean intervals."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(heartbeat_interval_seconds=bad_interval)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_timeout",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "3.0", None],
)
def test_heartbeat_config_rejects_invalid_timeout(bad_timeout: object) -> None:
    """Verify HeartbeatConfig rejects non-positive, non-finite, and boolean timeouts."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(timeout_seconds=bad_timeout)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_heartbeat_config_rejects_timeout_less_than_interval() -> None:
    """Verify timeout_seconds cannot be less than heartbeat_interval_seconds."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(heartbeat_interval_seconds=2.0, timeout_seconds=1.0)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT
    assert "cannot be less than" in exc_info.value.message


@pytest.mark.parametrize(
    "bad_misses",
    [0, -1, 1.5, float("nan"), float("inf"), True, False, "3", None],
)
def test_heartbeat_config_rejects_invalid_max_misses(bad_misses: object) -> None:
    """Verify max_consecutive_misses must be integer >= 1."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(max_consecutive_misses=bad_misses)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_warn",
    [0.0, -50.0, float("nan"), float("inf"), True, False, "100.0", None],
)
def test_heartbeat_config_rejects_invalid_latency_warning(bad_warn: object) -> None:
    """Verify latency_warning_threshold_ms rejects non-positive, non-finite, or bool values."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(latency_warning_threshold_ms=bad_warn)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_heartbeat_config_rejects_critical_less_than_warning() -> None:
    """Verify latency_critical_threshold_ms cannot be less than latency_warning_threshold_ms."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatConfig(latency_warning_threshold_ms=500.0, latency_critical_threshold_ms=200.0)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT
    assert "cannot be less than" in exc_info.value.message


# ============================================================================
# Section 3: HeartbeatRecord Validation & Immutability
# ============================================================================


def test_heartbeat_record_valid() -> None:
    """Verify creation and attributes of valid HeartbeatRecord."""
    rec = HeartbeatRecord(
        sequence_number=1,
        timestamp_ns=1_000_000_000,
        latency_ms=12.5,
        status=ConnectionStatus.CONNECTED,
    )
    assert rec.sequence_number == 1
    assert rec.timestamp_ns == 1_000_000_000
    assert rec.latency_ms == 12.5
    assert rec.status == ConnectionStatus.CONNECTED


@pytest.mark.parametrize(
    "bad_seq",
    [0, -1, 1.5, float("nan"), float("inf"), True, False, "1", None],
)
def test_heartbeat_record_rejects_invalid_sequence(bad_seq: object) -> None:
    """Verify HeartbeatRecord rejects non-positive, non-int, and bool sequence numbers."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatRecord(
            sequence_number=bad_seq,  # type: ignore[arg-type]
            timestamp_ns=1_000_000_000,
            latency_ms=10.0,
            status=ConnectionStatus.CONNECTED,
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_ts",
    [0, -1, 100.5, float("nan"), True, False, "1000", None],
)
def test_heartbeat_record_rejects_invalid_timestamp(bad_ts: object) -> None:
    """Verify HeartbeatRecord rejects non-positive, non-int, and bool timestamps."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatRecord(
            sequence_number=1,
            timestamp_ns=bad_ts,  # type: ignore[arg-type]
            latency_ms=10.0,
            status=ConnectionStatus.CONNECTED,
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_lat",
    [-1.0, float("nan"), float("inf"), float("-inf"), True, False, "10.0", None],
)
def test_heartbeat_record_rejects_invalid_latency(bad_lat: object) -> None:
    """Verify HeartbeatRecord rejects negative, non-finite, and bool latencies."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatRecord(
            sequence_number=1,
            timestamp_ns=1_000_000_000,
            latency_ms=bad_lat,  # type: ignore[arg-type]
            status=ConnectionStatus.CONNECTED,
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_heartbeat_record_rejects_invalid_status() -> None:
    """Verify HeartbeatRecord rejects non-ConnectionStatus types."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatRecord(
            sequence_number=1,
            timestamp_ns=1_000_000_000,
            latency_ms=10.0,
            status="CONNECTED",  # type: ignore[arg-type]
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


# ============================================================================
# Section 4: Watchdog Initialization & Input Sanitization
# ============================================================================


def test_watchdog_init_defaults() -> None:
    """Verify nominal HeartbeatWatchdog default state."""
    watchdog = HeartbeatWatchdog("BINANCE_SPOT")
    assert watchdog.gateway_id == "BINANCE_SPOT"
    assert watchdog.protocol == TransportProtocol.FIX
    assert watchdog.status == ConnectionStatus.CONNECTED
    assert watchdog.last_heartbeat_timestamp_ns == 0
    assert watchdog.last_sequence_number == 0
    assert watchdog.expected_sequence_number == 1
    assert watchdog.consecutive_misses == 0
    assert watchdog.sequence_gaps_count == 0
    assert len(watchdog.recent_latencies) == 0

    # Properties
    assert watchdog.is_connected is True
    assert watchdog.is_degraded is False
    assert watchdog.is_disconnected is False
    assert watchdog.is_reconnecting is False
    assert watchdog.is_alive is True
    assert watchdog.average_latency_ms == 0.0
    assert watchdog.last_latency_ms == 0.0
    assert watchdog.max_latency_ms == 0.0


@pytest.mark.parametrize("bad_gw", ["", "   ", None, 123, True])
def test_watchdog_init_rejects_invalid_gateway_id(bad_gw: object) -> None:
    """Verify HeartbeatWatchdog rejects empty or non-string gateway_id."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog(bad_gw)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_watchdog_init_rejects_invalid_config_type() -> None:
    """Verify HeartbeatWatchdog rejects non-HeartbeatConfig config object."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog("GW1", config="bad_config")  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_watchdog_init_rejects_invalid_protocol() -> None:
    """Verify HeartbeatWatchdog rejects non-TransportProtocol protocol."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog("GW1", protocol="FIX")  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_watchdog_init_rejects_invalid_initial_status() -> None:
    """Verify HeartbeatWatchdog rejects non-ConnectionStatus initial status."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog("GW1", initial_status="CONNECTED")  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize("bad_seq", [0, -1, 1.5, True, False, "1"])
def test_watchdog_init_rejects_invalid_initial_sequence(bad_seq: object) -> None:
    """Verify HeartbeatWatchdog rejects invalid initial sequence numbers."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog("GW1", initial_sequence_number=bad_seq)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize("bad_ts", [-1, 1.5, True, False, "0"])
def test_watchdog_init_rejects_invalid_initial_timestamp(bad_ts: object) -> None:
    """Verify HeartbeatWatchdog rejects negative or non-int initial timestamps."""
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        HeartbeatWatchdog("GW1", initial_timestamp_ns=bad_ts)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


# ============================================================================
# Section 5: Normal Heartbeat Ingestion & Latency Tracking
# ============================================================================


def test_normal_heartbeat_stream_maintains_connected() -> None:
    """Verify continuous in-order heartbeats maintain CONNECTED status."""
    watchdog = HeartbeatWatchdog("COINBASE", protocol=TransportProtocol.WEBSOCKET)

    # Ingest packets 1 through 5
    for seq in range(1, 6):
        rec = watchdog.record_heartbeat(
            sequence_number=seq,
            latency_ms=15.0 + seq,
            timestamp_ns=seq * 1_000_000_000,
        )
        assert rec.sequence_number == seq
        assert rec.status == ConnectionStatus.CONNECTED
        assert watchdog.last_sequence_number == seq
        assert watchdog.expected_sequence_number == seq + 1
        assert watchdog.sequence_gaps_count == 0
        assert watchdog.is_connected is True

    assert watchdog.consecutive_misses == 0
    assert len(watchdog.recent_latencies) == 5
    assert math.isclose(watchdog.last_latency_ms, 20.0)
    assert math.isclose(watchdog.average_latency_ms, 18.0)
    assert math.isclose(watchdog.max_latency_ms, 20.0)


# ============================================================================
# Section 6: Sequence Gap & Out-of-Order Packet Detection
# ============================================================================


def test_sequence_gap_detection_forward_skip() -> None:
    """Verify forward sequence skip triggers gap callback and advances expected sequence."""
    watchdog = HeartbeatWatchdog("KRAKEN", initial_sequence_number=1)

    gap_events: list[tuple[str, int, int, int]] = []

    def on_gap(gw: str, expected: int, received: int, ts: int) -> None:
        gap_events.append((gw, expected, received, ts))

    watchdog.register_sequence_gap_listener(on_gap)

    # Packet 1 (expected 1 -> received 1, ok)
    watchdog.record_heartbeat(sequence_number=1, latency_ms=10.0, timestamp_ns=1_000_000_000)
    assert len(gap_events) == 0
    assert watchdog.expected_sequence_number == 2

    # Packet 4 (expected 2 -> received 4: gap of packets 2 and 3!)
    watchdog.record_heartbeat(sequence_number=4, latency_ms=10.0, timestamp_ns=2_000_000_000)
    assert len(gap_events) == 1
    assert gap_events[0] == ("KRAKEN", 2, 4, 2_000_000_000)
    assert watchdog.sequence_gaps_count == 1
    # Crucial: next expected must advance to 5, not regress or stay stuck!
    assert watchdog.expected_sequence_number == 5

    # Packet 5 (expected 5 -> received 5, ok!)
    watchdog.record_heartbeat(sequence_number=5, latency_ms=10.0, timestamp_ns=3_000_000_000)
    assert len(gap_events) == 1  # No new gap!
    assert watchdog.sequence_gaps_count == 1
    assert watchdog.expected_sequence_number == 6


def test_sequence_gap_detection_out_of_order_packet() -> None:
    """Verify delayed / duplicate packet triggers gap callback but maintains expected watermark."""
    watchdog = HeartbeatWatchdog("BYBIT", initial_sequence_number=1)

    gap_events: list[tuple[str, int, int, int]] = []
    watchdog.register_sequence_gap_listener(
        lambda gw, exp, rcv, ts: gap_events.append((gw, exp, rcv, ts))
    )

    # Packet 1, 2, 3 normal
    watchdog.record_heartbeat(sequence_number=1, latency_ms=10.0, timestamp_ns=1_000_000_000)
    watchdog.record_heartbeat(sequence_number=2, latency_ms=10.0, timestamp_ns=2_000_000_000)
    watchdog.record_heartbeat(sequence_number=3, latency_ms=10.0, timestamp_ns=3_000_000_000)
    assert watchdog.expected_sequence_number == 4
    assert len(gap_events) == 0

    # Delayed packet 2 arrives again (duplicate or re-ordered)
    watchdog.record_heartbeat(sequence_number=2, latency_ms=10.0, timestamp_ns=3_500_000_000)
    assert len(gap_events) == 1
    assert gap_events[0] == ("BYBIT", 4, 2, 3_500_000_000)
    assert watchdog.sequence_gaps_count == 1
    # Expected sequence MUST NOT regress to 3; it must stay at high-watermark 4!
    assert watchdog.expected_sequence_number == 4

    # Packet 4 arrives normally
    watchdog.record_heartbeat(sequence_number=4, latency_ms=10.0, timestamp_ns=4_000_000_000)
    assert len(gap_events) == 1  # No new gap
    assert watchdog.expected_sequence_number == 5


# ============================================================================
# Section 7: Latency SLA Degradation & Recovery
# ============================================================================


def test_latency_warning_threshold_triggers_degraded() -> None:
    """Verify ping-pong latency >= warning threshold elevates connection to DEGRADED."""
    config = HeartbeatConfig(
        latency_warning_threshold_ms=200.0, latency_critical_threshold_ms=500.0
    )
    watchdog = HeartbeatWatchdog("OKX", config=config)

    status_transitions: list[tuple[str, ConnectionStatus, ConnectionStatus, int]] = []
    watchdog.register_status_listener(
        lambda gw, old_st, new_st, ts: status_transitions.append((gw, old_st, new_st, ts))
    )

    # Normal heartbeat (100ms < 200ms)
    watchdog.record_heartbeat(sequence_number=1, latency_ms=100.0, timestamp_ns=1_000_000_000)
    assert watchdog.status == ConnectionStatus.CONNECTED
    assert len(status_transitions) == 0

    # High latency heartbeat (250ms >= 200ms) => elevates to DEGRADED
    watchdog.record_heartbeat(sequence_number=2, latency_ms=250.0, timestamp_ns=2_000_000_000)
    assert watchdog.status == ConnectionStatus.DEGRADED
    assert watchdog.is_degraded is True
    assert watchdog.is_alive is True
    assert len(status_transitions) == 1
    assert status_transitions[0] == (
        "OKX",
        ConnectionStatus.CONNECTED,
        ConnectionStatus.DEGRADED,
        2_000_000_000,
    )

    # Critical latency heartbeat (600ms >= 500ms) => remains DEGRADED, no duplicate transition
    watchdog.record_heartbeat(sequence_number=3, latency_ms=600.0, timestamp_ns=3_000_000_000)
    assert watchdog.status == ConnectionStatus.DEGRADED
    assert len(status_transitions) == 1

    # Healthy latency recovers back to CONNECTED (50ms < 200ms)
    watchdog.record_heartbeat(sequence_number=4, latency_ms=50.0, timestamp_ns=4_000_000_000)
    assert watchdog.status == ConnectionStatus.CONNECTED
    assert watchdog.is_connected is True
    assert len(status_transitions) == 2
    assert status_transitions[1] == (
        "OKX",
        ConnectionStatus.DEGRADED,
        ConnectionStatus.CONNECTED,
        4_000_000_000,
    )


# ============================================================================
# Section 8: Liveness Checking, Timeout & Disconnect Transitions
# ============================================================================


def test_check_liveness_nominal_silence() -> None:
    """Verify check_liveness returns CONNECTED when elapsed silence is within interval."""
    config = HeartbeatConfig(heartbeat_interval_seconds=1.0, timeout_seconds=3.0)
    watchdog = HeartbeatWatchdog("CME", config=config, initial_timestamp_ns=1_000_000_000)

    # Elapsed: 0.5s (< 1.0s interval)
    status = watchdog.check_liveness(current_timestamp_ns=1_500_000_000)
    assert status == ConnectionStatus.CONNECTED
    assert watchdog.consecutive_misses == 0


def test_check_liveness_missed_interval_triggers_degraded() -> None:
    """Verify missing at least one interval elevates status to DEGRADED."""
    config = HeartbeatConfig(
        heartbeat_interval_seconds=1.0, timeout_seconds=3.0, max_consecutive_misses=3
    )
    watchdog = HeartbeatWatchdog("CME", config=config, initial_timestamp_ns=1_000_000_000)

    status_events: list[tuple[str, ConnectionStatus, ConnectionStatus, int]] = []
    watchdog.register_status_listener(lambda gw, o, n, ts: status_events.append((gw, o, n, ts)))

    # Elapsed: 1.5s (1 missed heartbeat, but < 3.0s timeout)
    status = watchdog.check_liveness(current_timestamp_ns=2_500_000_000)
    assert status == ConnectionStatus.DEGRADED
    assert watchdog.consecutive_misses == 1
    assert len(status_events) == 1
    assert status_events[0] == (
        "CME",
        ConnectionStatus.CONNECTED,
        ConnectionStatus.DEGRADED,
        2_500_000_000,
    )

    # Heartbeat arrives and recovers to CONNECTED
    watchdog.record_heartbeat(sequence_number=1, latency_ms=20.0, timestamp_ns=2_600_000_000)
    assert watchdog.status == ConnectionStatus.CONNECTED
    assert watchdog.consecutive_misses == 0
    assert len(status_events) == 2
    assert status_events[1] == (
        "CME",
        ConnectionStatus.DEGRADED,
        ConnectionStatus.CONNECTED,
        2_600_000_000,
    )


def test_check_liveness_timeout_triggers_disconnect_and_listeners() -> None:
    """Verify silence exceeding timeout_seconds triggers DISCONNECTED and on_disconnect callback."""
    config = HeartbeatConfig(
        heartbeat_interval_seconds=1.0, timeout_seconds=3.0, max_consecutive_misses=3
    )
    watchdog = HeartbeatWatchdog("NASDAQ", config=config, initial_timestamp_ns=1_000_000_000)

    disconnect_events: list[tuple[str, float, int]] = []
    status_events: list[tuple[str, ConnectionStatus, ConnectionStatus, int]] = []

    watchdog.register_disconnect_listener(lambda gw, el, ts: disconnect_events.append((gw, el, ts)))
    watchdog.register_status_listener(lambda gw, o, n, ts: status_events.append((gw, o, n, ts)))

    # Elapsed: 3.5s (>= 3.0s timeout and >= 3 misses)
    status = watchdog.check_liveness(current_timestamp_ns=4_500_000_000)
    assert status == ConnectionStatus.DISCONNECTED
    assert watchdog.is_disconnected is True
    assert watchdog.is_alive is False
    assert watchdog.consecutive_misses == 3

    assert len(disconnect_events) == 1
    gw_id, elapsed, ts = disconnect_events[0]
    assert gw_id == "NASDAQ"
    assert math.isclose(elapsed, 3.5)
    assert ts == 4_500_000_000

    assert len(status_events) == 1
    assert status_events[0] == (
        "NASDAQ",
        ConnectionStatus.CONNECTED,
        ConnectionStatus.DISCONNECTED,
        4_500_000_000,
    )

    # Crucial: Subsequent check_liveness calls while disconnected MUST NOT re-trigger disconnect listeners!
    status_2 = watchdog.check_liveness(current_timestamp_ns=5_000_000_000)
    assert status_2 == ConnectionStatus.DISCONNECTED
    assert len(disconnect_events) == 1
    assert len(status_events) == 1


def test_check_liveness_cold_start_initializes_baseline() -> None:
    """Verify check_liveness on cold start (timestamp=0) initializes baseline gracefully."""
    watchdog = HeartbeatWatchdog("ICE")
    assert watchdog.last_heartbeat_timestamp_ns == 0

    status = watchdog.check_liveness(current_timestamp_ns=1_000_000_000)
    assert status == ConnectionStatus.CONNECTED
    assert watchdog.last_heartbeat_timestamp_ns == 1_000_000_000


# ============================================================================
# Section 9: Reconnection, Reset, & State Machine Lifecycle
# ============================================================================


def test_mark_reconnecting_transitions_and_notifies() -> None:
    """Verify mark_reconnecting transitions state to RECONNECTING."""
    watchdog = HeartbeatWatchdog("EUREX", initial_status=ConnectionStatus.DISCONNECTED)
    assert watchdog.is_disconnected is True

    status_events: list[tuple[str, ConnectionStatus, ConnectionStatus, int]] = []
    watchdog.register_status_listener(lambda gw, o, n, ts: status_events.append((gw, o, n, ts)))

    watchdog.mark_reconnecting(timestamp_ns=2_000_000_000)
    assert watchdog.status == ConnectionStatus.RECONNECTING
    assert watchdog.is_reconnecting is True
    assert len(status_events) == 1
    assert status_events[0] == (
        "EUREX",
        ConnectionStatus.DISCONNECTED,
        ConnectionStatus.RECONNECTING,
        2_000_000_000,
    )

    # Repeated mark_reconnecting is idempotent
    watchdog.mark_reconnecting(timestamp_ns=2_100_000_000)
    assert len(status_events) == 1


def test_reset_restores_state_and_counters() -> None:
    """Verify reset re-initializes sequence tracking, clears misses, and restores CONNECTED."""
    watchdog = HeartbeatWatchdog("LSE")
    # Simulate degraded with missed packets and latency
    watchdog.record_heartbeat(sequence_number=1, latency_ms=300.0, timestamp_ns=1_000_000_000)
    watchdog.record_heartbeat(sequence_number=5, latency_ms=350.0, timestamp_ns=2_000_000_000)
    assert watchdog.sequence_gaps_count == 1
    assert watchdog.status == ConnectionStatus.DEGRADED
    assert len(watchdog.recent_latencies) == 2

    status_events: list[tuple[str, ConnectionStatus, ConnectionStatus, int]] = []
    watchdog.register_status_listener(lambda gw, o, n, ts: status_events.append((gw, o, n, ts)))

    # Reset with initial_sequence_number=100
    watchdog.reset(initial_sequence_number=100, current_timestamp_ns=3_000_000_000)

    assert watchdog.status == ConnectionStatus.CONNECTED
    assert watchdog.last_sequence_number == 99
    assert watchdog.expected_sequence_number == 100
    assert watchdog.last_heartbeat_timestamp_ns == 3_000_000_000
    assert watchdog.consecutive_misses == 0
    assert watchdog.sequence_gaps_count == 0
    assert len(watchdog.recent_latencies) == 0

    assert len(status_events) == 1
    assert status_events[0] == (
        "LSE",
        ConnectionStatus.DEGRADED,
        ConnectionStatus.CONNECTED,
        3_000_000_000,
    )

    # Ingest packet 100 without gap
    rec = watchdog.record_heartbeat(
        sequence_number=100, latency_ms=15.0, timestamp_ns=4_000_000_000
    )
    assert rec.sequence_number == 100
    assert watchdog.sequence_gaps_count == 0
    assert watchdog.expected_sequence_number == 101


# ============================================================================
# Section 10: Listener Management & Deduplication
# ============================================================================


def test_listener_registration_deduplication_and_unregistration() -> None:
    """Verify listeners can be registered, deduplicated, and cleanly unregistered."""
    watchdog = HeartbeatWatchdog("TEST_GW")

    def dummy_status(gw: str, o: ConnectionStatus, n: ConnectionStatus, ts: int) -> None:
        pass

    def dummy_gap(gw: str, exp: int, rcv: int, ts: int) -> None:
        pass

    def dummy_disc(gw: str, el: float, ts: int) -> None:
        pass

    # Status listener
    watchdog.register_status_listener(dummy_status)
    watchdog.register_status_listener(dummy_status)  # duplicate
    assert len(watchdog._status_listeners) == 1
    watchdog.unregister_status_listener(dummy_status)
    assert len(watchdog._status_listeners) == 0
    watchdog.unregister_status_listener(dummy_status)  # no-op

    # Gap listener
    watchdog.register_sequence_gap_listener(dummy_gap)
    watchdog.register_sequence_gap_listener(dummy_gap)  # duplicate
    assert len(watchdog._sequence_gap_listeners) == 1
    watchdog.unregister_sequence_gap_listener(dummy_gap)
    assert len(watchdog._sequence_gap_listeners) == 0
    watchdog.unregister_sequence_gap_listener(dummy_gap)  # no-op

    # Disconnect listener
    watchdog.register_disconnect_listener(dummy_disc)
    watchdog.register_disconnect_listener(dummy_disc)  # duplicate
    assert len(watchdog._disconnect_listeners) == 1
    watchdog.unregister_disconnect_listener(dummy_disc)
    assert len(watchdog._disconnect_listeners) == 0
    watchdog.unregister_disconnect_listener(dummy_disc)  # no-op


def test_listener_rejects_non_callable() -> None:
    """Verify listener registration rejects non-callable objects."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(TypeError, match="Status listener must be callable"):
        watchdog.register_status_listener("not_callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Sequence gap listener must be callable"):
        watchdog.register_sequence_gap_listener(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Disconnect listener must be callable"):
        watchdog.register_disconnect_listener(None)  # type: ignore[arg-type]


# ============================================================================
# Section 11: Adversarial Input Sanitization (INV-RSK-007)
# ============================================================================


@pytest.mark.parametrize(
    "bad_seq",
    [0, -1, 1.5, float("nan"), float("inf"), True, False, "1", None],
)
def test_record_heartbeat_rejects_invalid_sequence(bad_seq: object) -> None:
    """Verify record_heartbeat rejects invalid sequence numbers."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.record_heartbeat(
            sequence_number=bad_seq,  # type: ignore[arg-type]
            latency_ms=10.0,
            timestamp_ns=1_000_000_000,
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_lat",
    [-1.0, float("nan"), float("inf"), float("-inf"), True, False, "10.0", None],
)
def test_record_heartbeat_rejects_invalid_latency(bad_lat: object) -> None:
    """Verify record_heartbeat rejects invalid latencies."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.record_heartbeat(
            sequence_number=1,
            latency_ms=bad_lat,  # type: ignore[arg-type]
            timestamp_ns=1_000_000_000,
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_ts",
    [0, -1, 100.5, float("nan"), True, False, "1000", None],
)
def test_record_heartbeat_rejects_invalid_timestamp(bad_ts: object) -> None:
    """Verify record_heartbeat rejects invalid timestamps."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.record_heartbeat(
            sequence_number=1,
            latency_ms=10.0,
            timestamp_ns=bad_ts,  # type: ignore[arg-type]
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_ts",
    [0, -1, 100.5, float("nan"), True, False, "1000", None],
)
def test_check_liveness_rejects_invalid_timestamp(bad_ts: object) -> None:
    """Verify check_liveness rejects invalid timestamps."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.check_liveness(current_timestamp_ns=bad_ts)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize("bad_ts", [-1, 1.5, True, False, "100"])
def test_mark_reconnecting_rejects_invalid_timestamp(bad_ts: object) -> None:
    """Verify mark_reconnecting rejects negative, non-int, or boolean timestamps."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.mark_reconnecting(timestamp_ns=bad_ts)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize("bad_seq", [0, -5, 1.2, True, False, "10"])
def test_reset_rejects_invalid_initial_sequence(bad_seq: object) -> None:
    """Verify reset rejects invalid initial sequence numbers."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.reset(initial_sequence_number=bad_seq)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize("bad_ts", [-1, 1.5, True, False, "0"])
def test_reset_rejects_invalid_current_timestamp(bad_ts: object) -> None:
    """Verify reset rejects negative, non-int, or boolean current timestamps."""
    watchdog = HeartbeatWatchdog("TEST_GW")
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        watchdog.reset(current_timestamp_ns=bad_ts)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


# ============================================================================
# Section 12: Sub-10us Hot-Path Latency Benchmark (INV-RSK-006)
# ============================================================================


def test_heartbeat_hot_path_latency_sla() -> None:
    """Verify record_heartbeat and check_liveness execute in < 10us on hot path (INV-RSK-006)."""
    watchdog = HeartbeatWatchdog("HOT_PATH_GW")

    # Warm-up JIT and CPU caches
    for i in range(1, 101):
        ts = i * 1_000_000
        watchdog.record_heartbeat(sequence_number=i, latency_ms=5.0, timestamp_ns=ts)
        watchdog.check_liveness(current_timestamp_ns=ts + 100_000)

    # Benchmark 1,000 record_heartbeat iterations
    iterations = 1_000
    base_ts = 200_000_000
    start_ns = time.perf_counter_ns()
    for i in range(101, 101 + iterations):
        watchdog.record_heartbeat(
            sequence_number=i,
            latency_ms=8.0,
            timestamp_ns=base_ts + i * 100_000,
        )
    elapsed_ns = time.perf_counter_ns() - start_ns

    avg_record_us = (elapsed_ns / iterations) / 1_000.0
    is_traced = (
        sys.gettrace() is not None
        or "coverage" in sys.modules
        or "pytest_cov" in sys.modules
        or (
            hasattr(sys, "monitoring")
            and any(sys.monitoring.get_tool(i) is not None for i in range(6))
        )
    )
    sla_us = 25.0 if is_traced else 10.0
    assert avg_record_us < sla_us, (
        f"record_heartbeat average latency {avg_record_us:.3f}us breached {sla_us}us SLA"
    )

    # Benchmark 1,000 check_liveness iterations
    check_ts = base_ts + 101 * 100_000
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        watchdog.check_liveness(current_timestamp_ns=check_ts + 10_000)
    elapsed_check_ns = time.perf_counter_ns() - start_ns

    avg_check_us = (elapsed_check_ns / iterations) / 1_000.0
    assert avg_check_us < sla_us, (
        f"check_liveness average latency {avg_check_us:.3f}us breached {sla_us}us SLA"
    )
