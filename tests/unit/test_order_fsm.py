"""Unit tests for OrderStateMachine with causal out-of-order reconciliation and invariant protection.

Governing Standards:
- Rules.md (Rule 1: Line-by-line annotations; Rule 2: Diagnostic error codes; Rule 3: Quality gates; Rule 4: Mandatory adversarial red-teaming)
- INV-GW-001: Causal State Machine Monotonicity (Terminal states strictly immutable)
- INV-GW-003: Execution Mass Conservation (Cumulative quantity <= target quantity)
- INV-GW-004: Causal Out-of-Order Packet Reconciliation (PENDING_NEW -> NEW -> FILLED)
- INV-GW-005: Strict Non-Finite Input & Boundary Protection
- INV-GW-006: Sub-Millisecond Gateway Execution Latency SLA (< 0.10ms / 100us)
"""

from __future__ import annotations

import time

import pytest

from quant.execution.fsm import OrderStateMachine
from quant.execution.models import (
    ERR_GW_INVALID_STATE_TRANSITION,
    ERR_GW_NON_FINITE_INPUT,
    ExecutionReport,
    InvalidOrderInputException,
    InvalidStateTransitionException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)


def _make_sample_order(
    cl_ord_id: str = "ord-001",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: float = 100.0,
    price: float = 150.0,
    state: OrderState = OrderState.PENDING_NEW,
    filled_quantity: float = 0.0,
) -> Order:
    """Helper factory for generating standardized valid Order fixtures."""
    return Order(
        cl_ord_id=cl_ord_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
        state=state,
        filled_quantity=filled_quantity,
    )


def _make_sample_report(
    report_id: str = "rep-001",
    cl_ord_id: str = "ord-001",
    exchange_order_id: str = "ex-001",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    exec_type: OrderState = OrderState.NEW,
    last_quantity: float = 0.0,
    last_price: float = 0.0,
    cum_quantity: float = 0.0,
    leaves_quantity: float = 100.0,
    cum_quote_amount: float = 0.0,
    average_price: float = 0.0,
    fee: float = 0.0,
    timestamp_ns: int = 1000,
    text: str = "",
) -> ExecutionReport:
    """Helper factory for generating standardized valid ExecutionReport fixtures."""
    return ExecutionReport(
        report_id=report_id,
        cl_ord_id=cl_ord_id,
        exchange_order_id=exchange_order_id,
        symbol=symbol,
        side=side,
        exec_type=exec_type,
        last_quantity=last_quantity,
        last_price=last_price,
        cum_quantity=cum_quantity,
        leaves_quantity=leaves_quantity,
        cum_quote_amount=cum_quote_amount,
        average_price=average_price,
        fee=fee,
        timestamp_ns=timestamp_ns,
        text=text,
    )


# ============================================================================
# 1. Standard Happy-Path Lifecycles
# ============================================================================


def test_standard_order_lifecycle() -> None:
    """Test standard progression: PENDING_NEW -> NEW -> PARTIALLY_FILLED -> FILLED."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0, price=150.0)
    assert order.state == OrderState.PENDING_NEW
    assert order.leaves_quantity == 100.0
    assert order.is_active is True
    assert order.is_terminal is False

    # 1. Wire acknowledgment: PENDING_NEW -> NEW
    order = fsm.transition(order, OrderState.NEW, timestamp_ns=1_000_000)
    assert order.state == OrderState.NEW
    assert order.updated_at_ns == 1_000_000
    assert order.leaves_quantity == 100.0

    # 2. First partial execution: 40 shares @ $150.0
    rep1 = _make_sample_report(
        report_id="rep-1",
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=40.0,
        last_price=150.0,
        cum_quantity=40.0,
        leaves_quantity=60.0,
        cum_quote_amount=6000.0,
        average_price=150.0,
        fee=1.20,
        timestamp_ns=1_100_000,
    )
    order = fsm.apply_execution_report(order, rep1)
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_quantity == 40.0
    assert order.leaves_quantity == 60.0
    assert order.filled_quote_amount == 6000.0
    assert order.average_price == 150.0
    assert order.fees_paid == 1.20
    assert order.updated_at_ns == 1_100_000
    assert order.is_active is True

    # 3. Second partial execution: 30 shares @ $151.0
    rep2 = _make_sample_report(
        report_id="rep-2",
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=30.0,
        last_price=151.0,
        cum_quantity=70.0,
        leaves_quantity=30.0,
        cum_quote_amount=10530.0,
        average_price=150.42857,
        fee=0.90,
        timestamp_ns=1_200_000,
    )
    order = fsm.apply_execution_report(order, rep2)
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_quantity == 70.0
    assert order.leaves_quantity == 30.0
    assert order.fees_paid == pytest.approx(2.10)
    assert order.updated_at_ns == 1_200_000

    # 4. Final execution: remaining 30 shares @ $152.0 -> FILLED
    rep3 = _make_sample_report(
        report_id="rep-3",
        exec_type=OrderState.FILLED,
        last_quantity=30.0,
        last_price=152.0,
        cum_quantity=100.0,
        leaves_quantity=0.0,
        cum_quote_amount=15090.0,
        average_price=150.90,
        fee=0.90,
        timestamp_ns=1_300_000,
    )
    order = fsm.apply_execution_report(order, rep3)
    assert order.state == OrderState.FILLED
    assert order.filled_quantity == 100.0
    assert order.leaves_quantity == 0.0
    assert order.fees_paid == pytest.approx(3.00)
    assert order.is_active is False
    assert order.is_terminal is True


def test_standard_cancellation_lifecycle() -> None:
    """Test standard cancellation: PENDING_NEW -> NEW -> PENDING_CANCEL -> CANCELLED."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=50.0)

    # Acknowledged
    fsm.transition(order, OrderState.NEW, timestamp_ns=100)
    assert order.state == OrderState.NEW

    # Cancel requested
    fsm.transition(order, OrderState.PENDING_CANCEL, timestamp_ns=200)
    assert order.state == OrderState.PENDING_CANCEL
    assert order.is_active is True

    # Cancel confirmed by execution report
    rep = _make_sample_report(
        report_id="rep-cxl",
        exec_type=OrderState.CANCELLED,
        cum_quantity=0.0,
        leaves_quantity=0.0,
        timestamp_ns=300,
    )
    fsm.apply_execution_report(order, rep)
    assert order.state == OrderState.CANCELLED
    assert order.is_terminal is True
    assert order.is_active is False
    assert order.leaves_quantity == 0.0


def test_partial_fill_then_cancellation() -> None:
    """Test partial fill followed by cancellation of resting leaves."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)

    fsm.transition(order, OrderState.NEW, timestamp_ns=100)

    # 40 filled
    rep_fill = _make_sample_report(
        report_id="rep-pf",
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=40.0,
        last_price=100.0,
        cum_quantity=40.0,
        leaves_quantity=60.0,
        cum_quote_amount=4000.0,
        average_price=100.0,
        timestamp_ns=200,
    )
    fsm.apply_execution_report(order, rep_fill)
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.leaves_quantity == 60.0

    # Cancel request dispatched
    fsm.transition(order, OrderState.PENDING_CANCEL, timestamp_ns=300)
    assert order.state == OrderState.PENDING_CANCEL
    assert order.leaves_quantity == 60.0

    # Cancel confirmed: terminal state
    rep_cxl = _make_sample_report(
        report_id="rep-cxl",
        exec_type=OrderState.CANCELLED,
        cum_quantity=40.0,
        leaves_quantity=0.0,
        timestamp_ns=400,
    )
    fsm.apply_execution_report(order, rep_cxl)
    assert order.state == OrderState.CANCELLED
    assert order.is_terminal is True
    # State-aware leaves quantity MUST report 0.0 open resting leaves
    assert order.leaves_quantity == 0.0
    assert order.filled_quantity == 40.0


def test_immediate_rejection_lifecycle() -> None:
    """Test exchange rejection: PENDING_NEW -> REJECTED."""
    fsm = OrderStateMachine()
    order = _make_sample_order()

    rep = _make_sample_report(
        report_id="rep-rej",
        exec_type=OrderState.REJECTED,
        timestamp_ns=500,
        text="Order rejected by pre-trade credit check",
    )
    fsm.apply_execution_report(order, rep)
    assert order.state == OrderState.REJECTED
    assert order.is_terminal is True
    assert order.leaves_quantity == 0.0


@pytest.mark.parametrize(
    "initial_state",
    [OrderState.PENDING_NEW, OrderState.NEW, OrderState.PARTIALLY_FILLED],
)
def test_expiration_lifecycles(initial_state: OrderState) -> None:
    """Test order expiration (IOC/FOK/session end) from all eligible states."""
    fsm = OrderStateMachine()
    order = _make_sample_order(
        state=initial_state,
        filled_quantity=20.0 if initial_state == OrderState.PARTIALLY_FILLED else 0.0,
    )

    rep = _make_sample_report(
        report_id="rep-exp",
        exec_type=OrderState.EXPIRED,
        cum_quantity=order.filled_quantity,
        leaves_quantity=0.0,
        timestamp_ns=999,
        text="Time-in-force IOC expired",
    )
    fsm.apply_execution_report(order, rep)
    assert order.state == OrderState.EXPIRED
    assert order.is_terminal is True
    assert order.leaves_quantity == 0.0


# ============================================================================
# 2. Invariant 4: Causal Out-of-Order Packet Reconciliation (INV-GW-004)
# ============================================================================


def test_causal_out_of_order_filled_reconciliation() -> None:
    """Execution report FILLED arrives before order ack NEW (INV-GW-004)."""
    fsm = OrderStateMachine()
    order = _make_sample_order(cl_ord_id="ord-fast-1", quantity=100.0)
    assert order.state == OrderState.PENDING_NEW

    # Fast fill packet arrived from exchange multicast before HTTP/WebSocket NEW ack
    fill_report = _make_sample_report(
        report_id="rep-fast-fill",
        cl_ord_id="ord-fast-1",
        exchange_order_id="ex-fast-101",
        exec_type=OrderState.FILLED,
        last_quantity=100.0,
        last_price=150.0,
        cum_quantity=100.0,
        leaves_quantity=0.0,
        cum_quote_amount=15000.0,
        average_price=150.0,
        fee=2.0,
        timestamp_ns=1_000_050,
    )

    # Must synthesize intermediate NEW state and transition to FILLED without error
    updated_order = fsm.apply_execution_report(order, fill_report)
    assert updated_order.state == OrderState.FILLED
    assert updated_order.filled_quantity == 100.0
    assert updated_order.leaves_quantity == 0.0
    assert updated_order.exchange_order_id == "ex-fast-101"
    assert updated_order.updated_at_ns == 1_000_050
    assert updated_order.is_terminal is True


def test_causal_out_of_order_partially_filled_reconciliation() -> None:
    """Execution report PARTIALLY_FILLED arrives before order ack NEW (INV-GW-004)."""
    fsm = OrderStateMachine()
    order = _make_sample_order(cl_ord_id="ord-fast-2", quantity=100.0)
    assert order.state == OrderState.PENDING_NEW

    part_report = _make_sample_report(
        report_id="rep-fast-pf",
        cl_ord_id="ord-fast-2",
        exchange_order_id="ex-fast-102",
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=40.0,
        last_price=150.0,
        cum_quantity=40.0,
        leaves_quantity=60.0,
        cum_quote_amount=6000.0,
        average_price=150.0,
        fee=1.0,
        timestamp_ns=2_000_050,
    )

    updated_order = fsm.apply_execution_report(order, part_report)
    assert updated_order.state == OrderState.PARTIALLY_FILLED
    assert updated_order.filled_quantity == 40.0
    assert updated_order.leaves_quantity == 60.0
    assert updated_order.exchange_order_id == "ex-fast-102"
    assert updated_order.is_active is True


def test_transition_direct_reconciliation_from_pending_new() -> None:
    """Calling transition() directly from PENDING_NEW to FILLED/PARTIALLY_FILLED reconciles."""
    fsm = OrderStateMachine()
    order1 = _make_sample_order(quantity=100.0)
    fsm.transition(order1, OrderState.FILLED, timestamp_ns=5000)
    assert order1.state == OrderState.FILLED
    assert order1.updated_at_ns == 5000

    order2 = _make_sample_order(quantity=100.0)
    fsm.transition(order2, OrderState.PARTIALLY_FILLED, timestamp_ns=6000)
    assert order2.state == OrderState.PARTIALLY_FILLED
    assert order2.updated_at_ns == 6000


# ============================================================================
# 3. Late Fill Race Handling (PENDING_CANCEL -> FILLED / PARTIALLY_FILLED)
# ============================================================================


def test_late_fill_race_on_pending_cancel_to_filled() -> None:
    """Order matched on exchange right before cancel order processed."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)

    fsm.transition(order, OrderState.NEW, timestamp_ns=100)
    fsm.transition(order, OrderState.PENDING_CANCEL, timestamp_ns=200)
    assert order.state == OrderState.PENDING_CANCEL

    # Match event occurred on wire before cancel packet reached the order book
    fill_report = _make_sample_report(
        report_id="rep-late-fill",
        exec_type=OrderState.FILLED,
        last_quantity=100.0,
        last_price=150.0,
        cum_quantity=100.0,
        leaves_quantity=0.0,
        cum_quote_amount=15000.0,
        average_price=150.0,
        fee=2.0,
        timestamp_ns=250,
    )

    fsm.apply_execution_report(order, fill_report)
    assert order.state == OrderState.FILLED
    assert order.filled_quantity == 100.0
    assert order.leaves_quantity == 0.0
    assert order.is_terminal is True


def test_late_fill_race_on_pending_cancel_to_partially_filled() -> None:
    """Order partially matched right before cancel was processed."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)

    fsm.transition(order, OrderState.NEW, timestamp_ns=100)
    fsm.transition(order, OrderState.PENDING_CANCEL, timestamp_ns=200)

    late_pf_report = _make_sample_report(
        report_id="rep-late-pf",
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=50.0,
        last_price=150.0,
        cum_quantity=50.0,
        leaves_quantity=50.0,
        cum_quote_amount=7500.0,
        average_price=150.0,
        fee=1.0,
        timestamp_ns=250,
    )

    fsm.apply_execution_report(order, late_pf_report)
    assert order.state == OrderState.PARTIALLY_FILLED
    assert order.filled_quantity == 50.0
    assert order.leaves_quantity == 50.0
    assert order.is_active is True


# ============================================================================
# 4. Invariant 1: Terminal State Lockout (INV-GW-001)
# ============================================================================


@pytest.mark.parametrize(
    "terminal_state",
    [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED],
)
@pytest.mark.parametrize("target_state", list(OrderState))
def test_terminal_state_lockout_all_combinations(
    terminal_state: OrderState, target_state: OrderState
) -> None:
    """Terminal states are strictly immutable; transition attempts MUST raise InvalidStateTransitionException."""
    fsm = OrderStateMachine()
    order = _make_sample_order(state=terminal_state, filled_quantity=100.0)

    # 1. can_transition must return False for any terminal state
    assert fsm.can_transition(terminal_state, target_state) is False

    # 2. transition() must raise InvalidStateTransitionException with ERR-GW-001
    with pytest.raises(InvalidStateTransitionException) as exc:
        fsm.transition(order, target_state)
    assert exc.value.code == ERR_GW_INVALID_STATE_TRANSITION

    # 3. apply_execution_report() must raise InvalidStateTransitionException with ERR-GW-001
    rep = _make_sample_report(
        report_id=f"rep-fail-{terminal_state.value}-{target_state.value}",
        exec_type=target_state,
        cum_quantity=100.0,
    )
    with pytest.raises(InvalidStateTransitionException) as exc_rep:
        fsm.apply_execution_report(order, rep)
    assert exc_rep.value.code == ERR_GW_INVALID_STATE_TRANSITION


# ============================================================================
# 5. Invalid State Transitions (DAG Violations)
# ============================================================================


@pytest.mark.parametrize(
    "current_state,invalid_next_state",
    [
        (OrderState.PENDING_NEW, OrderState.CANCELLED),
        (OrderState.PENDING_NEW, OrderState.PENDING_CANCEL),
        (OrderState.NEW, OrderState.PENDING_NEW),
        (OrderState.NEW, OrderState.NEW),
        (OrderState.NEW, OrderState.REJECTED),
        (OrderState.PARTIALLY_FILLED, OrderState.PENDING_NEW),
        (OrderState.PARTIALLY_FILLED, OrderState.NEW),
        (OrderState.PARTIALLY_FILLED, OrderState.REJECTED),
        (OrderState.PENDING_CANCEL, OrderState.PENDING_NEW),
        (OrderState.PENDING_CANCEL, OrderState.NEW),
        (OrderState.PENDING_CANCEL, OrderState.PENDING_CANCEL),
        (OrderState.PENDING_CANCEL, OrderState.REJECTED),
        (OrderState.PENDING_CANCEL, OrderState.EXPIRED),
    ],
)
def test_invalid_transitions_rejected(
    current_state: OrderState, invalid_next_state: OrderState
) -> None:
    """DAG violations must be rejected with InvalidStateTransitionException (ERR-GW-001)."""
    fsm = OrderStateMachine()
    order = _make_sample_order(state=current_state, filled_quantity=20.0)

    assert fsm.can_transition(current_state, invalid_next_state) is False

    with pytest.raises(InvalidStateTransitionException) as exc:
        fsm.transition(order, invalid_next_state)
    assert exc.value.code == ERR_GW_INVALID_STATE_TRANSITION


# ============================================================================
# 6. Invariant 3: Mass Conservation & Monotonicity (INV-GW-003)
# ============================================================================


def test_mass_conservation_overfill_rejected() -> None:
    """Execution reports exceeding order target quantity (+ 1e-7) must be rejected."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)
    fsm.transition(order, OrderState.NEW)

    # Report claims 100.001 shares filled (violating 1e-7 tolerance)
    rep_overfill = _make_sample_report(
        report_id="rep-overfill",
        exec_type=OrderState.FILLED,
        last_quantity=100.001,
        last_price=150.0,
        cum_quantity=100.001,
        leaves_quantity=0.0,
        cum_quote_amount=15000.15,
        average_price=150.0,
    )
    with pytest.raises(InvalidOrderInputException) as exc:
        fsm.apply_execution_report(order, rep_overfill)
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT
    assert "exceeds order target quantity" in str(exc.value)


def test_mass_conservation_within_epsilon_accepted() -> None:
    """Execution report within 1e-7 floating-point tolerance of quantity is accepted."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)
    fsm.transition(order, OrderState.NEW)

    rep_tolerated = _make_sample_report(
        report_id="rep-eps",
        exec_type=OrderState.FILLED,
        last_quantity=100.0 + 5e-8,
        last_price=150.0,
        cum_quantity=100.0 + 5e-8,
        leaves_quantity=0.0,
        cum_quote_amount=15000.0,
        average_price=150.0,
    )
    fsm.apply_execution_report(order, rep_tolerated)
    assert order.state == OrderState.FILLED
    assert order.leaves_quantity == 0.0


def test_decreasing_cum_quantity_rejected() -> None:
    """Monotonicity: Cumulative filled quantity cannot regress across subsequent reports."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)
    fsm.transition(order, OrderState.NEW)

    # 1. Fill 50 shares
    rep1 = _make_sample_report(
        report_id="rep-1",
        exec_type=OrderState.PARTIALLY_FILLED,
        cum_quantity=50.0,
        leaves_quantity=50.0,
    )
    fsm.apply_execution_report(order, rep1)
    assert order.filled_quantity == 50.0

    # 2. Corrupt report claims cum_quantity is 40.0
    rep_regress = _make_sample_report(
        report_id="rep-2",
        exec_type=OrderState.PARTIALLY_FILLED,
        cum_quantity=40.0,
        leaves_quantity=60.0,
    )
    with pytest.raises(InvalidOrderInputException) as exc:
        fsm.apply_execution_report(order, rep_regress)
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT
    assert "monotonicity violation" in str(exc.value)


# ============================================================================
# 7. Identifier Mismatch & Type Safety Guards (INV-GW-005)
# ============================================================================


def test_order_id_mismatch_rejected() -> None:
    """ExecutionReport cl_ord_id mismatch must raise InvalidOrderInputException."""
    fsm = OrderStateMachine()
    order = _make_sample_order(cl_ord_id="ord-matching-id")

    mismatched_report = _make_sample_report(cl_ord_id="ord-alien-id")
    with pytest.raises(InvalidOrderInputException) as exc:
        fsm.apply_execution_report(order, mismatched_report)
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT
    assert "does not match ExecutionReport" in str(exc.value)


@pytest.mark.parametrize("bad_order", [None, "invalid_order", 123, {"cl_ord_id": "ord-1"}])
def test_invalid_order_instance_rejected(bad_order: object) -> None:
    """Non-Order objects passed to transition or apply_execution_report must raise InvalidOrderInputException."""
    fsm = OrderStateMachine()
    with pytest.raises(InvalidOrderInputException):
        fsm.transition(bad_order, OrderState.NEW)  # type: ignore[arg-type]

    with pytest.raises(InvalidOrderInputException):
        fsm.apply_execution_report(bad_order, _make_sample_report())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_report", [None, "invalid_report", 123, {"report_id": "rep-1"}])
def test_invalid_report_instance_rejected(bad_report: object) -> None:
    """Non-ExecutionReport objects passed to apply_execution_report must raise InvalidOrderInputException."""
    fsm = OrderStateMachine()
    order = _make_sample_order()
    with pytest.raises(InvalidOrderInputException):
        fsm.apply_execution_report(order, bad_report)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_state", [None, "NEW", 123, True])
def test_invalid_target_state_rejected(bad_state: object) -> None:
    """Non-OrderState objects passed to transition must raise InvalidOrderInputException."""
    fsm = OrderStateMachine()
    order = _make_sample_order()
    with pytest.raises(InvalidOrderInputException):
        fsm.transition(order, bad_state)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_ts", [-1, True, False, 1.5, "1000"])
def test_invalid_timestamp_rejected(bad_ts: object) -> None:
    """Invalid timestamp passed to transition must raise InvalidOrderInputException."""
    fsm = OrderStateMachine()
    order = _make_sample_order()
    with pytest.raises(InvalidOrderInputException):
        fsm.transition(order, OrderState.NEW, timestamp_ns=bad_ts)  # type: ignore[arg-type]


def test_transition_with_report_delegation() -> None:
    """Calling transition with report kwarg or positional report delegates to apply_execution_report."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=100.0)

    rep = _make_sample_report(
        exec_type=OrderState.NEW,
        timestamp_ns=777,
    )
    # 1. Positional report in transition
    fsm.transition(order, OrderState.NEW, rep)
    assert order.state == OrderState.NEW
    assert order.updated_at_ns == 777

    # 2. Keyword report in transition
    rep_fill = _make_sample_report(
        exec_type=OrderState.FILLED,
        cum_quantity=100.0,
        average_price=150.0,
        timestamp_ns=888,
    )
    fsm.transition(order, OrderState.FILLED, report=rep_fill)
    assert order.state == OrderState.FILLED
    assert order.filled_quantity == 100.0
    assert order.updated_at_ns == 888


def test_transition_with_mismatched_report_state_raises() -> None:
    """If target state does not match report.exec_type, transition() raises InvalidStateTransitionException."""
    fsm = OrderStateMachine()
    order = _make_sample_order()

    rep = _make_sample_report(exec_type=OrderState.NEW)
    with pytest.raises(InvalidStateTransitionException):
        fsm.transition(order, OrderState.FILLED, report=rep)


def test_transition_with_invalid_keyword_report_type_raises() -> None:
    """If report kwarg is not ExecutionReport, transition() raises InvalidOrderInputException."""
    fsm = OrderStateMachine()
    order = _make_sample_order()
    with pytest.raises(InvalidOrderInputException):
        fsm.transition(order, OrderState.NEW, report="invalid_report")  # type: ignore[arg-type]


def test_transition_default_timestamp_uses_system_clock() -> None:
    """If timestamp_ns is omitted, transition() sets updated_at_ns using current system clock."""
    fsm = OrderStateMachine()
    order = _make_sample_order()
    before_ns = time.time_ns()
    fsm.transition(order, OrderState.NEW)
    after_ns = time.time_ns()
    assert before_ns <= order.updated_at_ns <= after_ns


@pytest.mark.parametrize("bad_state", [None, "NEW", 123, True, 45.6])
def test_can_transition_invalid_types_return_false(bad_state: object) -> None:
    """can_transition must return False if either argument is not an OrderState."""
    fsm = OrderStateMachine()
    assert fsm.can_transition(bad_state, OrderState.NEW) is False  # type: ignore[arg-type]
    assert fsm.can_transition(OrderState.NEW, bad_state) is False  # type: ignore[arg-type]


def test_apply_execution_report_invalid_transition_raises() -> None:
    """Applying an execution report with a DAG-invalid transition raises InvalidStateTransitionException."""
    fsm = OrderStateMachine()
    order = _make_sample_order(state=OrderState.NEW)

    # An acknowledged order (NEW) cannot be REJECTED
    rep_rej = _make_sample_report(
        report_id="rep-invalid-rej",
        exec_type=OrderState.REJECTED,
        timestamp_ns=200,
    )
    with pytest.raises(InvalidStateTransitionException) as exc:
        fsm.apply_execution_report(order, rep_rej)
    assert exc.value.code == ERR_GW_INVALID_STATE_TRANSITION


# ============================================================================
# 8. Hot-Path Execution Latency SLA Benchmark (INV-GW-006)
# ============================================================================


def test_hot_path_execution_latency_sla() -> None:
    """Hot-path state transition must execute in < 0.10ms (100 microseconds) under INV-GW-006."""
    fsm = OrderStateMachine()
    order = _make_sample_order(quantity=10_000.0)
    fsm.transition(order, OrderState.NEW)

    report = _make_sample_report(
        report_id="rep-bench",
        exec_type=OrderState.PARTIALLY_FILLED,
        cum_quantity=1.0,
        last_quantity=1.0,
        average_price=150.0,
        timestamp_ns=1000,
    )

    # Warmup JIT / CPU caches
    for _ in range(100):
        fsm.can_transition(OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED)

    iterations = 5_000
    start_time = time.perf_counter()
    for _ in range(iterations):
        fsm.can_transition(OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED)
    elapsed_total = time.perf_counter() - start_time
    avg_latency_can_transition_s = elapsed_total / iterations

    # Benchmark apply_execution_report on an active order
    start_time = time.perf_counter()
    for _ in range(iterations):
        # We simulate minimal update without mutating cum_qty downwards
        order.state = OrderState.PARTIALLY_FILLED
        order.filled_quantity = 0.0
        fsm.apply_execution_report(order, report)
    elapsed_total = time.perf_counter() - start_time
    avg_latency_apply_report_s = elapsed_total / iterations

    # Invariant 6 SLA: <= 0.10ms (100 microseconds = 1e-4 seconds)
    assert avg_latency_can_transition_s < 1e-4, (
        f"can_transition latency {avg_latency_can_transition_s * 1e6:.2f}us exceeds 100us SLA"
    )
    assert avg_latency_apply_report_s < 1e-4, (
        f"apply_execution_report latency {avg_latency_apply_report_s * 1e6:.2f}us exceeds 100us SLA"
    )
