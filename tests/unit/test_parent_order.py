"""Unit tests for ParentOrder coordination and Implementation Shortfall (IS) attribution.

Purpose:
    Exhaustively verifies ParentOrder child execution tracking, overfill guard,
    timeout detection, defensive copying, and Perold (1988) exact additive
    Implementation Shortfall transaction cost attribution for both BUY and SELL sides.

Governing Standards:
    - Rule 1: Code transparency and defensive invariant verification.
    - Rule 2: Diagnostic fault code enforcement (ERR-SOR-004, ERR-SOR-006, ERR-SOR-007).
    - Rule 3: Quality gates (100% pass, >= 90% statement coverage).
    - Rule 4: Mandatory adversarial red-teaming, exact Perold additive identity, zero shortcuts.
"""

from __future__ import annotations

import math

import pytest

from quant.execution.models import OrderSide
from quant.execution.parent_order import (
    ChildFillRecord,
    ImplementationShortfallReport,
    ParentOrder,
)
from quant.execution.venues import (
    ERR_SOR_ALGORITHM_TIMEOUT,
    ERR_SOR_MASS_CONSERVATION_BREACH,
    ERR_SOR_NON_FINITE_INPUT,
    AlgorithmTimeoutException,
    InvalidSORInputException,
    MassConservationException,
    NonFiniteInputException,
)


def test_parent_order_child_tracking_and_is_attribution() -> None:
    """Plan baseline test: verify child fill tracking, completion, and IS attribution."""
    parent = ParentOrder(
        parent_id="parent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.00,
        decision_price=150.02,
        max_duration_seconds=60.0,
        start_time_ns=1_000_000_000,
    )
    assert not parent.is_completed
    assert parent.filled_quantity == 0.0
    assert parent.leaves_quantity == 100.0
    assert parent.average_execution_price == 0.0
    assert parent.total_fees == 0.0

    # Add child execution fills
    parent.record_child_fill(child_id="c-1", quantity=50.0, price=150.10, fee=1.0)
    assert not parent.is_completed
    assert parent.filled_quantity == 50.0
    assert parent.leaves_quantity == 50.0
    assert parent.average_execution_price == 150.10

    parent.record_child_fill(child_id="c-2", quantity=50.0, price=150.12, fee=1.0)
    assert parent.is_completed
    assert parent.filled_quantity == 100.0
    assert parent.leaves_quantity == 0.0
    assert parent.average_execution_price == 150.11
    assert parent.total_fees == 2.0

    is_report = parent.compute_implementation_shortfall(terminal_price=150.15)
    assert isinstance(is_report, ImplementationShortfallReport)
    assert is_report.delay_cost == pytest.approx(100.0 * (150.02 - 150.00))
    assert is_report.price_impact == pytest.approx(100.0 * (150.11 - 150.02))
    assert is_report.fees_paid == 2.0
    assert is_report.opportunity_cost == 0.0  # Fully filled
    assert is_report.total_shortfall == pytest.approx(
        is_report.delay_cost
        + is_report.price_impact
        + is_report.fees_paid
        + is_report.opportunity_cost
    )
    # Basis points verification
    benchmark_notional = 100.0 * 150.00
    assert is_report.total_shortfall_bps == pytest.approx(
        (is_report.total_shortfall / benchmark_notional) * 10_000.0
    )


def test_parent_order_constructor_validation() -> None:
    """Verify strict rejection of invalid inputs during ParentOrder instantiation."""
    # Reject empty or non-string parent_id
    with pytest.raises(InvalidSORInputException) as exc:
        ParentOrder(
            parent_id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
        )
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT

    with pytest.raises(InvalidSORInputException) as exc:
        ParentOrder(
            parent_id=123,
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
        )  # type: ignore[arg-type]
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT

    # Reject empty or non-string symbol
    with pytest.raises(InvalidSORInputException) as exc:
        ParentOrder(
            parent_id="p-1", symbol="", side=OrderSide.BUY, total_quantity=10.0, arrival_price=100.0
        )
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT

    # Reject invalid side
    with pytest.raises(InvalidSORInputException) as exc:
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side="INVALID_SIDE",
            total_quantity=10.0,
            arrival_price=100.0,
        )  # type: ignore[arg-type]
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT

    # Reject bool and non-positive / non-finite total_quantity
    with pytest.raises(NonFiniteInputException) as exc:
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=True,
            arrival_price=100.0,
        )  # type: ignore[arg-type]
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=0.0,
            arrival_price=100.0,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=-10.0,
            arrival_price=100.0,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=float("nan"),
            arrival_price=100.0,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=float("inf"),
            arrival_price=100.0,
        )

    # Reject bool and non-positive / non-finite arrival_price
    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=False,
        )  # type: ignore[arg-type]

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=0.0,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=-100.0,
        )

    # Reject bool and non-positive / non-finite decision_price
    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            decision_price=True,
        )  # type: ignore[arg-type]

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            decision_price=-10.0,
        )

    # Reject bool and non-positive max_duration_seconds
    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            max_duration_seconds=0.0,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            max_duration_seconds=True,
        )  # type: ignore[arg-type]

    # Reject bool and negative start_time_ns
    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            start_time_ns=-1,
        )

    with pytest.raises(NonFiniteInputException):
        ParentOrder(
            parent_id="p-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10.0,
            arrival_price=100.0,
            start_time_ns=True,
        )  # type: ignore[arg-type]


def test_child_fill_record_validation() -> None:
    """Verify ChildFillRecord strict type and boundary validations."""
    rec = ChildFillRecord(
        child_id="c-1",
        quantity=10.0,
        price=100.0,
        fee=0.05,
        timestamp_ns=1_000,
        spread_slippage=0.01,
    )
    assert rec.child_id == "c-1"
    assert rec.quantity == 10.0
    assert rec.price == 100.0
    assert rec.fee == 0.05
    assert rec.timestamp_ns == 1_000
    assert rec.spread_slippage == 0.01

    # Empty child_id
    with pytest.raises(InvalidSORInputException):
        ChildFillRecord(child_id="", quantity=10.0, price=100.0)

    # Non-string child_id
    with pytest.raises(InvalidSORInputException):
        ChildFillRecord(child_id=123, quantity=10.0, price=100.0)  # type: ignore[arg-type]

    # Boolean quantity
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=True, price=100.0)  # type: ignore[arg-type]

    # Non-positive quantity
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=0.0, price=100.0)

    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=-5.0, price=100.0)

    # Non-positive price
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=0.0)

    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=-100.0)

    # Negative fee
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=100.0, fee=-0.1)

    # Negative timestamp
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=100.0, timestamp_ns=-5)

    # Boolean timestamp
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=100.0, timestamp_ns=True)  # type: ignore[arg-type]

    # Non-finite spread slippage
    with pytest.raises(NonFiniteInputException):
        ChildFillRecord(child_id="c-1", quantity=10.0, price=100.0, spread_slippage=float("nan"))


def test_implementation_shortfall_report_validation() -> None:
    """Verify ImplementationShortfallReport validates fields and Perold identity."""
    valid_report = ImplementationShortfallReport(
        delay_cost=2.0,
        price_impact=9.0,
        spread_slippage=0.5,
        fees_paid=1.0,
        opportunity_cost=3.0,
        total_shortfall=15.0,  # 2 + 9 + 1 + 3 = 15.0
        total_shortfall_bps=10.0,
        delay_cost_bps=1.33,
        price_impact_bps=6.0,
        opportunity_cost_bps=2.0,
        fees_bps=0.67,
        filled_quantity=50.0,
        unfilled_quantity=50.0,
        average_price=150.18,
        arrival_price=150.00,
        decision_price=150.04,
        terminal_price=150.06,
    )
    assert valid_report.total_shortfall == 15.0

    # Breach Perold additive identity: total does not match components sum
    with pytest.raises(MassConservationException) as exc:
        ImplementationShortfallReport(
            delay_cost=2.0,
            price_impact=9.0,
            spread_slippage=0.0,
            fees_paid=1.0,
            opportunity_cost=3.0,
            total_shortfall=999.0,  # Inconsistent!
            total_shortfall_bps=10.0,
            delay_cost_bps=1.33,
            price_impact_bps=6.0,
            opportunity_cost_bps=2.0,
            fees_bps=0.67,
            filled_quantity=50.0,
            unfilled_quantity=50.0,
            average_price=150.18,
            arrival_price=150.00,
            decision_price=150.04,
            terminal_price=150.06,
        )
    assert exc.value.code == ERR_SOR_MASS_CONSERVATION_BREACH

    # Non-finite field
    with pytest.raises(NonFiniteInputException):
        ImplementationShortfallReport(
            delay_cost=float("nan"),
            price_impact=9.0,
            spread_slippage=0.0,
            fees_paid=1.0,
            opportunity_cost=3.0,
            total_shortfall=13.0,
            total_shortfall_bps=10.0,
            delay_cost_bps=1.33,
            price_impact_bps=6.0,
            opportunity_cost_bps=2.0,
            fees_bps=0.67,
            filled_quantity=50.0,
            unfilled_quantity=50.0,
            average_price=150.18,
            arrival_price=150.00,
            decision_price=150.04,
            terminal_price=150.06,
        )


def test_parent_order_sell_side_shortfall_attribution() -> None:
    """Verify SELL order sign reversal and exact shortfall attribution."""
    parent = ParentOrder(
        parent_id="sell-001",
        symbol="MSFT",
        side=OrderSide.SELL,
        total_quantity=200.0,
        arrival_price=300.00,
        decision_price=299.90,  # Price dropped before decision: loss of 0.10/sh for seller
        max_duration_seconds=120.0,
    )
    # Child fill executed at 299.70: price impact = -1 * (299.70 - 299.90) = +0.20/sh loss
    parent.record_child_fill(child_id="c-s1", quantity=200.0, price=299.70, fee=5.0)

    assert parent.is_completed
    assert parent.average_execution_price == 299.70

    report = parent.compute_implementation_shortfall(terminal_price=299.50)
    # Delay cost: -1.0 * 200.0 * (299.90 - 300.00) = +20.0
    assert report.delay_cost == pytest.approx(20.0)
    # Price impact: -1.0 * 200.0 * (299.70 - 299.90) = +40.0
    assert report.price_impact == pytest.approx(40.0)
    assert report.fees_paid == 5.0
    assert report.opportunity_cost == 0.0
    assert report.total_shortfall == pytest.approx(20.0 + 40.0 + 5.0 + 0.0)

    # Now verify favorable price movement (price went up for seller)
    parent_gain = ParentOrder(
        parent_id="sell-002",
        symbol="MSFT",
        side=OrderSide.SELL,
        total_quantity=100.0,
        arrival_price=300.00,
        decision_price=300.50,  # Price rose before decision: gain of 0.50/sh
    )
    parent_gain.record_child_fill(child_id="c-g1", quantity=100.0, price=301.00, fee=2.0)
    report_gain = parent_gain.compute_implementation_shortfall(terminal_price=301.00)
    # Delay cost: -1.0 * 100.0 * (300.50 - 300.00) = -50.0 (favorable)
    assert report_gain.delay_cost == pytest.approx(-50.0)
    # Price impact: -1.0 * 100.0 * (301.00 - 300.50) = -50.0 (favorable)
    assert report_gain.price_impact == pytest.approx(-50.0)
    assert report_gain.fees_paid == 2.0
    assert report_gain.total_shortfall == pytest.approx(-50.0 + -50.0 + 2.0)  # -98.0 net


def test_parent_order_partial_fill_and_opportunity_cost() -> None:
    """Verify opportunity cost computation on unfilled order leaves."""
    # BUY order with 60 filled, 40 unfilled leaves
    parent = ParentOrder(
        parent_id="partial-001",
        symbol="GOOGL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=100.00,
        decision_price=100.00,
    )
    parent.record_child_fill(child_id="c-1", quantity=60.0, price=100.50, fee=1.5)

    assert not parent.is_completed
    assert parent.filled_quantity == 60.0
    assert parent.leaves_quantity == 40.0

    # Market rallied to 102.00 at terminal horizon: opportunity cost on 40 leaves = 40 * (102 - 100) = +80.0
    report = parent.compute_implementation_shortfall(terminal_price=102.00)
    assert report.delay_cost == 0.0
    assert report.price_impact == pytest.approx(60.0 * (100.50 - 100.00))  # 30.0
    assert report.fees_paid == 1.5
    assert report.opportunity_cost == pytest.approx(40.0 * (102.00 - 100.00))  # 80.0
    assert report.total_shortfall == pytest.approx(30.0 + 1.5 + 80.0)

    # SELL order with partial fill and market drop:
    parent_sell = ParentOrder(
        parent_id="partial-sell-001",
        symbol="GOOGL",
        side=OrderSide.SELL,
        total_quantity=100.0,
        arrival_price=100.00,
        decision_price=100.00,
    )
    parent_sell.record_child_fill(child_id="c-s1", quantity=70.0, price=99.50, fee=2.0)
    # Market dropped to 97.00: opportunity cost on 30 unsold shares = -1 * 30 * (97 - 100) = +90.0
    report_sell = parent_sell.compute_implementation_shortfall(terminal_price=97.00)
    assert report_sell.opportunity_cost == pytest.approx(90.0)
    assert report_sell.total_shortfall == pytest.approx(
        report_sell.delay_cost
        + report_sell.price_impact
        + report_sell.fees_paid
        + report_sell.opportunity_cost
    )


def test_parent_order_zero_fills() -> None:
    """Verify zero-fill edge case: average price is 0.0, delay/impact are 0.0, all shortfall in opportunity."""
    parent = ParentOrder(
        parent_id="zero-001",
        symbol="NVDA",
        side=OrderSide.BUY,
        total_quantity=50.0,
        arrival_price=200.00,
        decision_price=201.00,
    )
    assert not parent.is_completed
    assert parent.filled_quantity == 0.0
    assert parent.leaves_quantity == 50.0
    assert parent.average_execution_price == 0.0
    assert parent.total_fees == 0.0

    # With terminal price 205.00
    report = parent.compute_implementation_shortfall(terminal_price=205.00)
    assert report.delay_cost == 0.0
    assert report.price_impact == 0.0
    assert report.fees_paid == 0.0
    assert report.opportunity_cost == pytest.approx(50.0 * (205.00 - 200.00))  # 250.0
    assert report.total_shortfall == pytest.approx(250.0)

    # Without terminal price, defaults to arrival price
    report_default = parent.compute_implementation_shortfall(terminal_price=None)
    assert report_default.terminal_price == 200.00
    assert report_default.opportunity_cost == 0.0
    assert report_default.total_shortfall == 0.0


def test_parent_order_overfill_guard() -> None:
    """Verify cumulative child fills exceeding total quantity raises MassConservationException."""
    parent = ParentOrder(
        parent_id="overfill-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.00,
    )
    parent.record_child_fill(child_id="c-1", quantity=60.0, price=150.00)

    # Overfill by 0.01
    with pytest.raises(MassConservationException) as exc:
        parent.record_child_fill(child_id="c-2", quantity=40.01, price=150.00)
    assert exc.value.code == ERR_SOR_MASS_CONSERVATION_BREACH

    # Exact fill within 1e-7 tolerance is allowed
    parent.record_child_fill(child_id="c-2-ok", quantity=40.0, price=150.00)
    assert parent.is_completed

    # Any further fill is rejected
    with pytest.raises(MassConservationException) as exc2:
        parent.record_child_fill(child_id="c-3", quantity=0.1, price=150.00)
    assert exc2.value.code == ERR_SOR_MASS_CONSERVATION_BREACH


def test_parent_order_timeout_detection() -> None:
    """Verify timeout exception when execution horizon expires with open leaves."""
    start_ns = 1_000_000_000
    max_duration_sec = 60.0  # 60s -> 60_000_000_000 ns
    parent = ParentOrder(
        parent_id="time-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.00,
        max_duration_seconds=max_duration_sec,
        start_time_ns=start_ns,
    )
    # Check within horizon (50s later)
    parent.check_timeout(now_ns=start_ns + 50_000_000_000)

    # Check exactly at horizon (60s later) -> not yet strictly exceeded
    parent.check_timeout(now_ns=start_ns + 60_000_000_000)

    # Check beyond horizon (61s later)
    with pytest.raises(AlgorithmTimeoutException) as exc:
        parent.check_timeout(now_ns=start_ns + 61_000_000_000)
    assert exc.value.code == ERR_SOR_ALGORITHM_TIMEOUT

    # Completed parent order does NOT timeout even if well past horizon
    parent.record_child_fill(child_id="c-all", quantity=100.0, price=150.00)
    assert parent.is_completed
    parent.check_timeout(now_ns=start_ns + 999_000_000_000)


def test_parent_order_check_timeout_validation() -> None:
    """Verify check_timeout rejects boolean, non-integer, or past timestamps."""
    parent = ParentOrder(
        parent_id="time-val-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10.0,
        arrival_price=100.0,
        start_time_ns=1_000,
    )
    # Reject boolean
    with pytest.raises(NonFiniteInputException):
        parent.check_timeout(now_ns=True)  # type: ignore[arg-type]

    # Reject float
    with pytest.raises(NonFiniteInputException):
        parent.check_timeout(now_ns=2000.5)  # type: ignore[arg-type]

    # Reject timestamp preceding start_time_ns
    with pytest.raises(InvalidSORInputException):
        parent.check_timeout(now_ns=500)


def test_parent_order_defensive_copies() -> None:
    """Verify child_fills property returns a defensive copy."""
    parent = ParentOrder(
        parent_id="def-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10.0,
        arrival_price=100.0,
    )
    parent.record_child_fill(child_id="c-1", quantity=5.0, price=100.0)
    fills = parent.child_fills
    assert len(fills) == 1

    # Mutate returned list
    fills.clear()
    assert len(parent.child_fills) == 1


def test_parent_order_multiple_child_fills_vwap_and_fees() -> None:
    """Verify VWAP execution price and fees aggregation across multiple fills."""
    parent = ParentOrder(
        parent_id="multi-001",
        symbol="TSLA",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=250.00,
    )
    parent.record_child_fill(
        child_id="c-1", quantity=20.0, price=250.00, fee=0.50, spread_slippage=0.02
    )
    parent.record_child_fill(
        child_id="c-2", quantity=30.0, price=251.00, fee=0.75, spread_slippage=0.03
    )
    parent.record_child_fill(
        child_id="c-3", quantity=50.0, price=252.00, fee=1.25, spread_slippage=0.05
    )

    assert parent.is_completed
    expected_vwap = (20.0 * 250.00 + 30.0 * 251.00 + 50.0 * 252.00) / 100.0
    assert parent.average_execution_price == pytest.approx(expected_vwap)
    assert parent.total_fees == pytest.approx(0.50 + 0.75 + 1.25)

    report = parent.compute_implementation_shortfall(terminal_price=253.00)
    assert report.spread_slippage == pytest.approx(0.02 + 0.03 + 0.05)
    assert report.total_shortfall == pytest.approx(
        report.delay_cost + report.price_impact + report.fees_paid + report.opportunity_cost
    )


def test_parent_order_perold_additive_identity_stress_test() -> None:
    """Adversarial stress test: verify exact Perold identity across diverse market scenarios."""
    scenarios = [
        # (side, qty, arrival, decision, terminal, [(qty, price, fee)])
        (
            OrderSide.BUY,
            1000.0,
            50.0,
            50.05,
            50.20,
            [(300.0, 50.10, 2.0), (300.0, 50.15, 2.0), (200.0, 50.12, 1.5)],
        ),
        (OrderSide.BUY, 500.0, 100.0, 99.80, 99.50, [(250.0, 99.90, 1.0), (250.0, 99.70, 1.0)]),
        (
            OrderSide.SELL,
            1000.0,
            50.0,
            49.95,
            49.80,
            [(300.0, 49.90, 2.0), (300.0, 49.85, 2.0), (200.0, 49.88, 1.5)],
        ),
        (
            OrderSide.SELL,
            500.0,
            100.0,
            100.20,
            100.50,
            [(250.0, 100.10, 1.0), (250.0, 100.30, 1.0)],
        ),
    ]
    for idx, (side, total_q, p_arr, p_dec, p_term, fills) in enumerate(scenarios):
        parent = ParentOrder(
            parent_id=f"stress-{idx}",
            symbol="XYZ",
            side=side,
            total_quantity=total_q,
            arrival_price=p_arr,
            decision_price=p_dec,
        )
        for c_idx, (q, p, fee) in enumerate(fills):
            parent.record_child_fill(child_id=f"c-{c_idx}", quantity=q, price=p, fee=fee)

        report = parent.compute_implementation_shortfall(terminal_price=p_term)
        # Verify INV-SOR-005: additive decomposition holds within 1e-7 tolerance
        sum_components = (
            report.delay_cost + report.price_impact + report.fees_paid + report.opportunity_cost
        )
        assert math.isclose(report.total_shortfall, sum_components, abs_tol=1e-7)

        # Verify bps consistency
        benchmark = total_q * p_arr
        expected_total_bps = (report.total_shortfall / benchmark) * 10_000.0
        assert math.isclose(report.total_shortfall_bps, expected_total_bps, abs_tol=1e-7)


def test_parent_order_properties_and_repr() -> None:
    """Verify all ParentOrder getters and __repr__ formatting."""
    parent = ParentOrder(
        parent_id="p-props",
        symbol="NVDA",
        side=OrderSide.BUY,
        total_quantity=50.0,
        arrival_price=100.0,
        decision_price=100.5,
        max_duration_seconds=120.0,
        start_time_ns=500,
    )
    assert parent.parent_id == "p-props"
    assert parent.symbol == "NVDA"
    assert parent.side == OrderSide.BUY
    assert parent.total_quantity == 50.0
    assert parent.arrival_price == 100.0
    assert parent.decision_price == 100.5
    assert parent.max_duration_seconds == 120.0
    assert parent.start_time_ns == 500
    rep = repr(parent)
    assert "ParentOrder(id='p-props'" in rep
    assert "NVDA" in rep
    assert "BUY" in rep


def test_parent_order_fill_timestamp_causality() -> None:
    """Verify record_child_fill rejects timestamp preceding start_time_ns."""
    parent = ParentOrder(
        parent_id="p-time",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.0,
        start_time_ns=1_000_000,
    )
    # Valid timestamp >= start_time_ns
    parent.record_child_fill(child_id="c-ok", quantity=10.0, price=150.0, timestamp_ns=1_000_001)

    # Invalid timestamp < start_time_ns
    with pytest.raises(InvalidSORInputException) as exc:
        parent.record_child_fill(child_id="c-bad", quantity=10.0, price=150.0, timestamp_ns=500_000)
    assert exc.value.code == ERR_SOR_NON_FINITE_INPUT


def test_parent_order_compute_is_terminal_price_validation() -> None:
    """Verify compute_implementation_shortfall rejects invalid terminal_price."""
    parent = ParentOrder(
        parent_id="p-term",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.0,
    )
    # Reject bool
    with pytest.raises(NonFiniteInputException):
        parent.compute_implementation_shortfall(terminal_price=True)  # type: ignore[arg-type]

    # Reject non-positive
    with pytest.raises(NonFiniteInputException):
        parent.compute_implementation_shortfall(terminal_price=0.0)

    with pytest.raises(NonFiniteInputException):
        parent.compute_implementation_shortfall(terminal_price=-10.0)

    # Reject NaN
    with pytest.raises(NonFiniteInputException):
        parent.compute_implementation_shortfall(terminal_price=float("nan"))


def test_implementation_shortfall_report_edge_validations() -> None:
    """Verify edge validation branches on ImplementationShortfallReport."""
    # Negative filled quantity
    with pytest.raises(NonFiniteInputException):
        ImplementationShortfallReport(
            delay_cost=0.0,
            price_impact=0.0,
            spread_slippage=0.0,
            fees_paid=0.0,
            opportunity_cost=0.0,
            total_shortfall=0.0,
            total_shortfall_bps=0.0,
            delay_cost_bps=0.0,
            price_impact_bps=0.0,
            opportunity_cost_bps=0.0,
            fees_bps=0.0,
            filled_quantity=-1.0,  # invalid
            unfilled_quantity=10.0,
            average_price=100.0,
            arrival_price=100.0,
            decision_price=100.0,
            terminal_price=100.0,
        )

    # Non-positive arrival price
    with pytest.raises(NonFiniteInputException):
        ImplementationShortfallReport(
            delay_cost=0.0,
            price_impact=0.0,
            spread_slippage=0.0,
            fees_paid=0.0,
            opportunity_cost=0.0,
            total_shortfall=0.0,
            total_shortfall_bps=0.0,
            delay_cost_bps=0.0,
            price_impact_bps=0.0,
            opportunity_cost_bps=0.0,
            fees_bps=0.0,
            filled_quantity=10.0,
            unfilled_quantity=0.0,
            average_price=100.0,
            arrival_price=0.0,  # invalid
            decision_price=100.0,
            terminal_price=100.0,
        )

    # Non-positive average price when filled_quantity > 0
    with pytest.raises(NonFiniteInputException):
        ImplementationShortfallReport(
            delay_cost=0.0,
            price_impact=0.0,
            spread_slippage=0.0,
            fees_paid=0.0,
            opportunity_cost=0.0,
            total_shortfall=0.0,
            total_shortfall_bps=0.0,
            delay_cost_bps=0.0,
            price_impact_bps=0.0,
            opportunity_cost_bps=0.0,
            fees_bps=0.0,
            filled_quantity=10.0,
            unfilled_quantity=0.0,
            average_price=0.0,  # invalid when filled
            arrival_price=100.0,
            decision_price=100.0,
            terminal_price=100.0,
        )


def test_parent_order_terminal_price_none_with_fills() -> None:
    """Verify compute_implementation_shortfall when terminal_price is None and fills exist."""
    parent = ParentOrder(
        parent_id="p-none-term",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.0,
    )
    parent.record_child_fill(child_id="c-1", quantity=60.0, price=152.0)
    # Fills exist, terminal_price is None -> resolves to average_execution_price (152.0)
    report = parent.compute_implementation_shortfall(terminal_price=None)
    assert report.terminal_price == 152.0
    assert report.average_price == 152.0
