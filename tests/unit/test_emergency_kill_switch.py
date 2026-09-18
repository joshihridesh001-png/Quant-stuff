"""Comprehensive unit tests for Institutional Emergency Kill Switch and Mass Cancellation.

Governing Standards:
- Rules.md:
  - Rule 1: Defensive Invariants and explicit contract testing.
  - Rule 2: Zero-execution deterministic diagnostic codes (ERR-RSK-008, ERR-RSK-007, ERR-RSK-009, ERR-RSK-010).
  - Rule 3: Quality gates (100% pass rate, strict typing, >= 90% statement coverage on kill_switch.py).
  - Rule 4: Mandatory adversarial red-teaming, non-finite/bool guards, sub-50ms SLA.
- Invariants:
  - INV-RSK-008: Atomic Kill Switch Mass Cancellation (< 50ms SLA, 100% open orders cancelled)
  - INV-RSK-006: Hot-Path Latency SLA (< 10us validate_submission)
  - INV-RSK-007: Strict Non-Finite & Boolean Input Sanitization
  - Gateway Exception Isolation: asyncio.gather(..., return_exceptions=True)
  - Constant-Time Admin Token Verification: hmac.compare_digest
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from typing import Final

import pytest

from quant.execution.gateway import PaperExecutionGateway
from quant.execution.kill_switch import (
    ERR_INVALID_ADMIN_TOKEN,
    ERR_KILL_SWITCH_ACTIVE,
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
    Order,
    OrderSide,
    OrderType,
)
from quant.execution.risk import (
    ERR_RSK_NON_FINITE_INPUT,
    KillSwitchActiveException,
    NonFiniteRiskInputException,
)

_TEST_ADMIN_TOKEN: Final[str] = "SUPER_SECURE_ROOT_ADMIN_TOKEN_999"


# ============================================================================
# Section 1: Enum & Exception Hierarchy Tests
# ============================================================================


def test_kill_switch_state_values() -> None:
    """Verify KillSwitchState enum values and string compatibility."""
    assert KillSwitchState.ARMED_STANDBY == "ARMED_STANDBY"
    assert KillSwitchState.PANIC_TRIGGERED == "PANIC_TRIGGERED"
    assert KillSwitchState.DISARMED == "DISARMED"


def test_panic_trigger_reason_values() -> None:
    """Verify PanicTriggerReason enum members and string compatibility."""
    assert PanicTriggerReason.MANUAL_OPERATOR == "MANUAL_OPERATOR"
    assert PanicTriggerReason.DRAWDOWN_BREACH == "DRAWDOWN_BREACH"
    assert PanicTriggerReason.GATEWAY_DISCONNECT == "GATEWAY_DISCONNECT"
    assert PanicTriggerReason.FAT_FINGER_BREACH == "FAT_FINGER_BREACH"
    assert PanicTriggerReason.ROGUE_FILLS == "ROGUE_FILLS"


def test_panic_mode_values() -> None:
    """Verify PanicMode enum values and string compatibility."""
    assert PanicMode.CANCEL_ONLY == "CANCEL_ONLY"
    assert PanicMode.CANCEL_AND_FLATTEN == "CANCEL_AND_FLATTEN"


def test_exception_fault_codes() -> None:
    """Verify exception diagnostic fault codes and inheritance hierarchy."""
    exc1 = InvalidAdminTokenException()
    assert exc1.code == ERR_INVALID_ADMIN_TOKEN
    assert isinstance(exc1, PermissionError)

    exc2 = KillSwitchDisarmedException()
    assert exc2.code == ERR_KILL_SWITCH_DISARMED

    exc3 = KillSwitchActiveException("Kill switch triggered")
    assert exc3.code == ERR_KILL_SWITCH_ACTIVE


# ============================================================================
# Section 2: KillSwitchEvent Value Object Invariants
# ============================================================================


def test_kill_switch_event_valid_instantiation() -> None:
    """Verify valid instantiation of immutable KillSwitchEvent."""
    now_ns = time.time_ns()
    event = KillSwitchEvent(
        timestamp_ns=now_ns,
        trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
        trigger_source="web_console_user_1",
        panic_mode=PanicMode.CANCEL_ONLY,
        cancelled_orders_count=5,
        positions_snapshot={"AAPL": 100.0, "MSFT": -50.0},
        details="Manual test panic",
    )
    assert event.timestamp_ns == now_ns
    assert event.trigger_reason == PanicTriggerReason.MANUAL_OPERATOR
    assert event.trigger_source == "web_console_user_1"
    assert event.panic_mode == PanicMode.CANCEL_ONLY
    assert event.cancelled_orders_count == 5
    assert event.positions_snapshot == {"AAPL": 100.0, "MSFT": -50.0}
    assert event.details == "Manual test panic"


def test_kill_switch_event_string_enum_coercion() -> None:
    """Verify string enum values are coerced to StrEnum instances in KillSwitchEvent."""
    now_ns = time.time_ns()
    event = KillSwitchEvent(
        timestamp_ns=now_ns,
        trigger_reason="DRAWDOWN_BREACH",  # type: ignore[arg-type]
        trigger_source="risk_firewall",
        panic_mode="CANCEL_AND_FLATTEN",  # type: ignore[arg-type]
        cancelled_orders_count=0,
        positions_snapshot={},
        details="",
    )
    assert event.trigger_reason == PanicTriggerReason.DRAWDOWN_BREACH
    assert event.panic_mode == PanicMode.CANCEL_AND_FLATTEN


def test_kill_switch_event_rejects_corrupt_inputs() -> None:
    """Verify KillSwitchEvent rejects booleans, non-finite values, and bad types."""
    now_ns = time.time_ns()

    # Boolean timestamp
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        KillSwitchEvent(
            timestamp_ns=True,  # type: ignore[arg-type]
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT

    # Negative / zero timestamp
    with pytest.raises(NonFiniteRiskInputException):
        KillSwitchEvent(
            timestamp_ns=-100,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    # Invalid trigger reason
    with pytest.raises(ValueError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason="UNKNOWN_REASON",  # type: ignore[arg-type]
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    with pytest.raises(TypeError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=123,  # type: ignore[arg-type]
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    # Empty or boolean source
    with pytest.raises(ValueError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="   ",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    with pytest.raises(ValueError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source=False,  # type: ignore[arg-type]
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    # Invalid panic mode
    with pytest.raises(ValueError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode="INVALID_MODE",  # type: ignore[arg-type]
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    with pytest.raises(TypeError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=None,  # type: ignore[arg-type]
            cancelled_orders_count=0,
            positions_snapshot={},
            details="",
        )

    # Negative or boolean cancelled count
    with pytest.raises(NonFiniteRiskInputException):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=-1,
            positions_snapshot={},
            details="",
        )

    with pytest.raises(NonFiniteRiskInputException):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=True,  # type: ignore[arg-type]
            positions_snapshot={},
            details="",
        )

    # Corrupt positions snapshot (NaN / Inf / bool)
    with pytest.raises(NonFiniteRiskInputException):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={"AAPL": float("nan")},
            details="",
        )

    with pytest.raises(NonFiniteRiskInputException):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={"AAPL": True},  # type: ignore[dict-item]
            details="",
        )

    with pytest.raises(TypeError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot="not_a_dict",  # type: ignore[arg-type]
            details="",
        )

    with pytest.raises(TypeError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={123: 10.0},  # type: ignore[dict-item]
            details="",
        )

    with pytest.raises(TypeError):
        KillSwitchEvent(
            timestamp_ns=now_ns,
            trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
            trigger_source="admin",
            panic_mode=PanicMode.CANCEL_ONLY,
            cancelled_orders_count=0,
            positions_snapshot={},
            details=123,  # type: ignore[arg-type]
        )


def test_kill_switch_event_immutability() -> None:
    """Verify KillSwitchEvent is frozen and its dictionary cannot be mutated externally."""
    now_ns = time.time_ns()
    positions = {"AAPL": 100.0}
    event = KillSwitchEvent(
        timestamp_ns=now_ns,
        trigger_reason=PanicTriggerReason.MANUAL_OPERATOR,
        trigger_source="admin",
        panic_mode=PanicMode.CANCEL_ONLY,
        cancelled_orders_count=1,
        positions_snapshot=positions,
        details="immutability test",
    )
    # Mutating original dictionary should not affect event
    positions["AAPL"] = 999.0
    assert event.positions_snapshot["AAPL"] == 100.0

    # Mutating event attributes directly raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        event.details = "changed"  # type: ignore[misc]


# ============================================================================
# Section 3: EmergencyKillSwitch Initialization & State Lifecycle
# ============================================================================


def test_kill_switch_init_defaults() -> None:
    """Verify default initialization parameters and properties."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    assert ks.state == KillSwitchState.ARMED_STANDBY
    assert ks.is_armed is True
    assert ks.is_active is False
    assert ks.is_disarmed is False
    assert ks.last_event is None
    assert ks.gateways == {}
    assert ks.scheduler_hooks == []
    assert ks.panic_listeners == []


def test_kill_switch_init_validation() -> None:
    """Verify defensive rejection of invalid initialization parameters."""
    with pytest.raises(NonFiniteRiskInputException):
        EmergencyKillSwitch(admin_token="   ")

    with pytest.raises(NonFiniteRiskInputException):
        EmergencyKillSwitch(admin_token=True)  # type: ignore[arg-type]

    with pytest.raises(NonFiniteRiskInputException):
        EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN, gateway_timeout_seconds=-1.0)

    with pytest.raises(NonFiniteRiskInputException):
        EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN, gateway_timeout_seconds=float("nan"))

    with pytest.raises(NonFiniteRiskInputException):
        EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN, gateway_timeout_seconds=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        EmergencyKillSwitch(
            admin_token=_TEST_ADMIN_TOKEN,
            initial_state="INVALID_STATE",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        EmergencyKillSwitch(
            admin_token=_TEST_ADMIN_TOKEN,
            initial_state=123,  # type: ignore[arg-type]
        )


def test_submission_validation_in_armed_state() -> None:
    """Verify validate_submission passes without exception in ARMED_STANDBY."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    # Should not raise
    ks.validate_submission()


def test_admin_state_transitions() -> None:
    """Verify disarm, arm, and reset state transitions with admin token authentication."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    # Unauthorized disarm fails
    with pytest.raises(InvalidAdminTokenException) as exc_info:
        ks.disarm("WRONG_TOKEN")
    assert exc_info.value.code == ERR_INVALID_ADMIN_TOKEN
    assert ks.state == KillSwitchState.ARMED_STANDBY

    # Authorized disarm succeeds
    ks.disarm(_TEST_ADMIN_TOKEN)
    assert ks.state == KillSwitchState.DISARMED
    assert ks.is_disarmed is True
    assert ks.is_armed is False
    assert ks.is_active is False
    ks.validate_submission()  # Disarmed passes submissions

    # Unauthorized arm fails
    with pytest.raises(InvalidAdminTokenException):
        ks.arm("BAD_TOKEN")
    assert ks.state == KillSwitchState.DISARMED

    # Authorized arm succeeds
    ks.arm(_TEST_ADMIN_TOKEN)
    assert ks.state == KillSwitchState.ARMED_STANDBY
    assert ks.is_armed is True

    # Unauthorized reset fails
    with pytest.raises(InvalidAdminTokenException):
        ks.reset("BAD_TOKEN")

    # Authorized reset succeeds
    ks.reset(_TEST_ADMIN_TOKEN)
    assert ks.state == KillSwitchState.ARMED_STANDBY


# ============================================================================
# Section 4: Gateway & Callback Registration Invariants
# ============================================================================


def test_gateway_registration_and_unregistration() -> None:
    """Verify gateway registration with explicit ID, implicit ID, and unregistration."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    gw1 = PaperExecutionGateway(initial_balance=1_000_000.0)
    gw2 = PaperExecutionGateway(initial_balance=500_000.0)

    # Register with explicit ID
    resolved_id1 = ks.register_gateway(gw1, gateway_id="venue_alpha")
    assert resolved_id1 == "venue_alpha"
    assert "venue_alpha" in ks.gateways
    assert ks.gateways["venue_alpha"] is gw1

    # Register with implicit ID
    resolved_id2 = ks.register_gateway(gw2)
    assert resolved_id2.startswith("PaperExecutionGateway_")
    assert resolved_id2 in ks.gateways

    # Unregister existing gateway
    ks.unregister_gateway("venue_alpha")
    assert "venue_alpha" not in ks.gateways

    # Unregister unknown gateway raises KeyError
    with pytest.raises(KeyError):
        ks.unregister_gateway("non_existent_gw")

    # Invalid gateway_id inputs
    with pytest.raises(ValueError):
        ks.unregister_gateway("   ")
    with pytest.raises(ValueError):
        ks.unregister_gateway(True)  # type: ignore[arg-type]


def test_gateway_registration_type_enforcement() -> None:
    """Verify non-ExecutionGateway objects are rejected upon registration."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    with pytest.raises(TypeError):
        ks.register_gateway("not_a_gateway")  # type: ignore[arg-type]

    gw = PaperExecutionGateway()
    with pytest.raises(ValueError):
        ks.register_gateway(gw, gateway_id="  ")
    with pytest.raises(ValueError):
        ks.register_gateway(gw, gateway_id=True)  # type: ignore[arg-type]


def test_scheduler_hooks_and_panic_listeners_registration() -> None:
    """Verify registration and type enforcement of hooks and listeners."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    def dummy_hook() -> None:
        pass

    async def dummy_listener(event: KillSwitchEvent) -> None:
        pass

    ks.register_scheduler_hook(dummy_hook)
    ks.register_panic_listener(dummy_listener)

    assert len(ks.scheduler_hooks) == 1
    assert len(ks.panic_listeners) == 1

    with pytest.raises(TypeError):
        ks.register_scheduler_hook("not_callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        ks.register_panic_listener(123)  # type: ignore[arg-type]


# ============================================================================
# Section 5: Panic Triggering, Mass Cancellation & Order Lockout
# ============================================================================


@pytest.mark.asyncio
async def test_manual_panic_trigger_and_submission_lockout() -> None:
    """Verify panic trigger transitions to PANIC_TRIGGERED and locks out subsequent orders."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    gw = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw.connect()
    ks.register_gateway(gw, gateway_id="paper_1")

    now_ns = time.time_ns()
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.MANUAL_OPERATOR,
        source="desk_head",
        panic_mode=PanicMode.CANCEL_ONLY,
        details="Operator pressed physical red button",
        current_timestamp_ns=now_ns,
    )

    assert ks.state == KillSwitchState.PANIC_TRIGGERED
    assert ks.is_active is True
    assert ks.is_armed is False
    assert ks.last_event is event
    assert event.timestamp_ns == now_ns
    assert event.trigger_reason == PanicTriggerReason.MANUAL_OPERATOR
    assert event.trigger_source == "desk_head"
    assert event.details == "Operator pressed physical red button"

    # Submission validation must now strictly raise KillSwitchActiveException (INV-RSK-008)
    with pytest.raises(KillSwitchActiveException) as exc_info:
        ks.validate_submission()
    assert exc_info.value.code == ERR_KILL_SWITCH_ACTIVE
    assert "desk_head" in str(exc_info.value)

    # Resetting with admin token restores ARMED_STANDBY
    ks.reset(_TEST_ADMIN_TOKEN)
    assert ks.state == KillSwitchState.ARMED_STANDBY
    assert ks.is_active is False
    # Validate submission passes again
    ks.validate_submission()


@pytest.mark.asyncio
async def test_disarmed_kill_switch_blocks_panic_trigger() -> None:
    """Verify triggering panic while DISARMED raises KillSwitchDisarmedException."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    ks.disarm(_TEST_ADMIN_TOKEN)
    assert ks.is_disarmed is True

    with pytest.raises(KillSwitchDisarmedException) as exc_info:
        await ks.trigger_panic(
            reason=PanicTriggerReason.DRAWDOWN_BREACH,
            source="tripwire",
        )
    assert exc_info.value.code == ERR_KILL_SWITCH_DISARMED


@pytest.mark.asyncio
async def test_concurrent_multi_gateway_bulk_cancellation() -> None:
    """Verify simultaneous mass cancellation across multiple independent execution venues."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    # Venue 1: Paper Gateway with 3 resting orders
    gw1 = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw1.connect()
    ks.register_gateway(gw1, gateway_id="venue_1")

    # Venue 2: Paper Gateway with 2 resting orders
    gw2 = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw2.connect()
    ks.register_gateway(gw2, gateway_id="venue_2")

    now_ns = time.time_ns()
    # Submit resting orders on gw1
    for i in range(3):
        order = Order(
            cl_ord_id=f"gw1-ord-{i}",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=100.0 + i,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
        )
        await gw1.submit_order(order)

    # Submit resting orders on gw2
    for i in range(2):
        order = Order(
            cl_ord_id=f"gw2-ord-{i}",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=200.0 + i,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
        )
        await gw2.submit_order(order)

    # Confirm resting open orders exist
    assert len(await gw1.get_open_orders()) == 3
    assert len(await gw2.get_open_orders()) == 2

    # Trigger panic
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.DRAWDOWN_BREACH,
        source="drawdown_tripwire",
        panic_mode=PanicMode.CANCEL_ONLY,
    )

    # Verify 100% of open orders across both venues are cancelled
    assert event.cancelled_orders_count == 5
    assert len(await gw1.get_open_orders()) == 0
    assert len(await gw2.get_open_orders()) == 0


@pytest.mark.asyncio
async def test_gateway_transport_exception_isolation() -> None:
    """Verify failing/disconnected gateways do not abort cancellation on healthy gateways."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    # Healthy gateway
    gw_healthy = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw_healthy.connect()
    ks.register_gateway(gw_healthy, gateway_id="healthy_venue")

    # Disconnected / failing gateway
    gw_failing = PaperExecutionGateway(initial_balance=1_000_000.0)
    # Note: gw_failing is NOT connected, so cancel_order will raise GatewayDisconnectedException
    ks.register_gateway(gw_failing, gateway_id="failing_venue")

    now_ns = time.time_ns()
    order = Order(
        cl_ord_id="healthy-ord-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=150.0,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
    )
    await gw_healthy.submit_order(order)
    assert len(await gw_healthy.get_open_orders()) == 1

    # Trigger panic: must isolate failing gateway without crashing
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.GATEWAY_DISCONNECT,
        source="heartbeat_watchdog",
    )

    assert event.cancelled_orders_count == 1
    assert len(await gw_healthy.get_open_orders()) == 0
    assert ks.state == KillSwitchState.PANIC_TRIGGERED


@pytest.mark.asyncio
async def test_scheduler_hooks_execution_and_exception_isolation() -> None:
    """Verify scheduler cancellation hooks are executed with full exception isolation."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    hook_1_called = False
    hook_2_called = False

    def sync_hook() -> None:
        nonlocal hook_1_called
        hook_1_called = True

    async def async_hook() -> None:
        nonlocal hook_2_called
        await asyncio.sleep(0.001)
        hook_2_called = True

    def faulty_hook() -> None:
        raise RuntimeError("Algorithmic scheduler hook crashed!")

    ks.register_scheduler_hook(sync_hook)
    ks.register_scheduler_hook(faulty_hook)
    ks.register_scheduler_hook(async_hook)

    event = await ks.trigger_panic(
        reason=PanicTriggerReason.FAT_FINGER_BREACH,
        source="pre_trade_risk",
    )

    assert hook_1_called is True
    assert hook_2_called is True
    assert event.trigger_reason == PanicTriggerReason.FAT_FINGER_BREACH


@pytest.mark.asyncio
async def test_panic_listeners_execution_and_exception_isolation() -> None:
    """Verify registered panic listeners receive event notifications with exception isolation."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    received_events: list[KillSwitchEvent] = []

    def sync_listener(ev: KillSwitchEvent) -> None:
        received_events.append(ev)

    async def async_listener(ev: KillSwitchEvent) -> None:
        await asyncio.sleep(0.001)
        received_events.append(ev)

    def crashing_listener(ev: KillSwitchEvent) -> None:
        raise ValueError("Telemetry listener crashed!")

    ks.register_panic_listener(sync_listener)
    ks.register_panic_listener(crashing_listener)
    ks.register_panic_listener(async_listener)

    event = await ks.trigger_panic(
        reason=PanicTriggerReason.ROGUE_FILLS,
        source="markout_monitor",
    )

    assert len(received_events) == 2
    assert received_events[0] is event
    assert received_events[1] is event


@pytest.mark.asyncio
async def test_idempotent_multi_triggering() -> None:
    """Verify simultaneous or repeated trigger_panic calls return identical active event."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    gw = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw.connect()
    ks.register_gateway(gw, gateway_id="venue_idempotent")

    now_ns = time.time_ns()
    order = Order(
        cl_ord_id="idemp-ord-1",
        symbol="GOOGL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=5.0,
        price=120.0,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
    )
    await gw.submit_order(order)

    # Concurrently fire 5 panic triggers from multiple simulated tripwires
    results = await asyncio.gather(
        ks.trigger_panic(PanicTriggerReason.MANUAL_OPERATOR, source="operator"),
        ks.trigger_panic(PanicTriggerReason.DRAWDOWN_BREACH, source="tripwire"),
        ks.trigger_panic(PanicTriggerReason.GATEWAY_DISCONNECT, source="watchdog"),
        ks.trigger_panic(PanicTriggerReason.FAT_FINGER_BREACH, source="firewall"),
        ks.trigger_panic(PanicTriggerReason.ROGUE_FILLS, source="markout"),
    )

    # All returned events must be the exact same object reference
    first_event = results[0]
    for ev in results:
        assert ev is first_event

    assert first_event.cancelled_orders_count == 1
    assert len(await gw.get_open_orders()) == 0


@pytest.mark.asyncio
async def test_panic_mode_cancel_and_flatten() -> None:
    """Verify PanicMode.CANCEL_AND_FLATTEN cancels orders and submits market flattening orders."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    gw = PaperExecutionGateway(initial_balance=1_000_000.0)
    await gw.connect()
    gw.set_market_price("AAPL", 150.0)
    gw.set_market_price("MSFT", 200.0)
    ks.register_gateway(gw, gateway_id="paper_flatten")

    now_ns = time.time_ns()
    # Create existing long position in AAPL (10 units)
    buy_order = Order(
        cl_ord_id="fill-buy-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
    )
    await gw.submit_order(buy_order)

    # Place a resting limit order in MSFT
    limit_order = Order(
        cl_ord_id="rest-limit-1",
        symbol="MSFT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=5.0,
        price=180.0,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
    )
    await gw.submit_order(limit_order)

    pos_before = await gw.get_positions()
    assert pos_before.get("AAPL") == 10.0
    assert len(await gw.get_open_orders()) == 1

    # Trigger panic with CANCEL_AND_FLATTEN
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.MANUAL_OPERATOR,
        source="risk_officer",
        panic_mode=PanicMode.CANCEL_AND_FLATTEN,
    )

    assert event.cancelled_orders_count == 1
    assert event.positions_snapshot.get("AAPL") == 10.0

    # Limit order must be cancelled, and inventory flattened to 0
    assert len(await gw.get_open_orders()) == 0
    pos_after = await gw.get_positions()
    assert pos_after.get("AAPL", 0.0) == 0.0


@pytest.mark.asyncio
async def test_sub_50ms_mass_cancellation_sla_benchmark() -> None:
    """Verify INV-RSK-008 Sub-50ms SLA across multiple venues and 50 resting orders."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    # Setup 3 gateways with 20 resting orders each (60 orders total)
    gateways: list[PaperExecutionGateway] = []
    for g_idx in range(3):
        gw = PaperExecutionGateway(initial_balance=2_000_000.0)
        await gw.connect()
        ks.register_gateway(gw, gateway_id=f"perf_venue_{g_idx}")
        gateways.append(gw)

        now_ns = time.time_ns()
        for o_idx in range(20):
            order = Order(
                cl_ord_id=f"perf-ord-{g_idx}-{o_idx}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                price=50.0 + o_idx,
                created_at_ns=now_ns,
                updated_at_ns=now_ns,
            )
            await gw.submit_order(order)

    # Benchmark mass cancellation latency
    start_time = time.perf_counter()
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.DRAWDOWN_BREACH,
        source="sla_benchmark",
    )
    elapsed_seconds = time.perf_counter() - start_time
    elapsed_ms = elapsed_seconds * 1000.0

    assert event.cancelled_orders_count == 60
    for gw in gateways:
        assert len(await gw.get_open_orders()) == 0

    # Strict SLA verification: < 50ms
    assert elapsed_ms < 50.0, f"Mass cancellation took {elapsed_ms:.2f}ms, exceeding 50ms SLA!"


# ============================================================================
# Section 6: Adversarial Red-Teaming & Edge Cases
# ============================================================================


def test_constant_time_admin_token_timing_defense() -> None:
    """Verify admin verification handles arbitrary objects, bools, and empty tokens."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    # Bad token types
    assert ks._verify_admin_token("") is False
    assert ks._verify_admin_token(None) is False
    assert ks._verify_admin_token(True) is False
    assert ks._verify_admin_token(12345) is False
    assert ks._verify_admin_token("WRONG_LENGTH") is False
    assert ks._verify_admin_token(_TEST_ADMIN_TOKEN) is True


@pytest.mark.asyncio
async def test_trigger_panic_input_validation() -> None:
    """Verify trigger_panic input parameter defensive boundaries."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    with pytest.raises(ValueError):
        await ks.trigger_panic("INVALID_REASON", source="desk")

    with pytest.raises(TypeError):
        await ks.trigger_panic(123, source="desk")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await ks.trigger_panic(PanicTriggerReason.MANUAL_OPERATOR, source="   ")

    with pytest.raises(ValueError):
        await ks.trigger_panic(PanicTriggerReason.MANUAL_OPERATOR, source=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await ks.trigger_panic(
            PanicTriggerReason.MANUAL_OPERATOR,
            source="desk",
            panic_mode="INVALID_MODE",
        )

    with pytest.raises(TypeError):
        await ks.trigger_panic(
            PanicTriggerReason.MANUAL_OPERATOR,
            source="desk",
            panic_mode=None,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        await ks.trigger_panic(
            PanicTriggerReason.MANUAL_OPERATOR,
            source="desk",
            details=None,  # type: ignore[arg-type]
        )

    with pytest.raises(NonFiniteRiskInputException):
        await ks.trigger_panic(
            PanicTriggerReason.MANUAL_OPERATOR,
            source="desk",
            current_timestamp_ns=-500,
        )

    with pytest.raises(NonFiniteRiskInputException):
        await ks.trigger_panic(
            PanicTriggerReason.MANUAL_OPERATOR,
            source="desk",
            current_timestamp_ns=True,  # type: ignore[arg-type]
        )


def test_gateway_attribute_id_discovery() -> None:
    """Verify gateway_id and venue_id discovery when gateway_id is omitted."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    class GatewayWithGatewayId(PaperExecutionGateway):
        gateway_id = "custom_gw_id_42"

    class GatewayWithVenueId(PaperExecutionGateway):
        venue_id = "custom_venue_id_99"

    gw1 = GatewayWithGatewayId()
    gw2 = GatewayWithVenueId()

    gid1 = ks.register_gateway(gw1)
    gid2 = ks.register_gateway(gw2)

    assert gid1 == "custom_gw_id_42"
    assert gid2 == "custom_venue_id_99"


@pytest.mark.asyncio
async def test_gateway_fetch_exceptions_during_cancellation_and_flatten() -> None:
    """Verify exception handling when get_open_orders or get_positions raises during cancellation."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)

    class BrokenFetchGateway(PaperExecutionGateway):
        async def get_open_orders(self) -> list[Order]:
            raise ConnectionResetError("Socket broken during order fetch")

        async def get_positions(self) -> dict[str, float]:
            raise TimeoutError("Socket timed out during position fetch")

    broken_gw = BrokenFetchGateway()
    ks.register_gateway(broken_gw, gateway_id="broken_fetch")

    # Should handle both exceptions gracefully and produce an event
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.GATEWAY_DISCONNECT,
        source="watchdog",
        panic_mode=PanicMode.CANCEL_AND_FLATTEN,
    )
    assert event.cancelled_orders_count == 0
    assert event.positions_snapshot == {}


@pytest.mark.asyncio
async def test_trigger_panic_with_enum_instances() -> None:
    """Verify trigger_panic accepts direct StrEnum instances without string coercion."""
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    event = await ks.trigger_panic(
        reason=PanicTriggerReason.MANUAL_OPERATOR,
        source="operator",
        panic_mode=PanicMode.CANCEL_ONLY,
    )
    assert event.trigger_reason == PanicTriggerReason.MANUAL_OPERATOR
    assert event.panic_mode == PanicMode.CANCEL_ONLY


@pytest.mark.asyncio
async def test_fetch_results_top_level_exception_isolation() -> None:
    """Verify isolation when asyncio.gather returns an unhandled Exception in fetch_results."""
    from unittest.mock import patch

    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN_TOKEN)
    gw = PaperExecutionGateway()
    ks.register_gateway(gw, gateway_id="mock_gw")

    async def mock_gather_side_effect(*args: object, **kwargs: object) -> list[object]:
        import inspect

        for arg in args:
            if inspect.iscoroutine(arg):
                arg.close()
        return [RuntimeError("Uncaught gather error")]

    with patch(
        "quant.execution.kill_switch.asyncio.gather",
        side_effect=mock_gather_side_effect,
    ):
        event = await ks.trigger_panic(PanicTriggerReason.MANUAL_OPERATOR, source="operator")
        assert event.cancelled_orders_count == 0
