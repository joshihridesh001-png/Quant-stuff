"""Comprehensive unit and integration tests for Unified RiskOrchestrator façade.

Governing Standards:
- Rules.md:
  - Rule 1: Four-tier line annotations and explicit contract invariant testing.
  - Rule 2: Zero-execution deterministic diagnostic codes (ERR-RSK-001..008, ERR-HB-001).
  - Rule 3: Quality gates (100% pass rate, strict typing, >= 90% statement coverage on risk_orchestrator.py).
  - Rule 4: Mandatory adversarial red-teaming, non-finite/bool guards, sub-20us SLA, zero iterative numerical solvers.
- Invariants:
  - INV-RSK-001..007: Pre-Trade Risk Firewall Bounds & Input Sanitization
  - INV-RSK-008: Atomic Emergency Kill Switch Mass Cancellation & Order Lockout
  - Automated Tripwire Coupling: Watchdog disconnect & Market data drawdown breaches trigger kill switch
  - Atomic Leaves Accounting: Pending leaves incremented before dispatch and rolled back on failure
  - Sub-20us Hot-Path Latency SLA: Combined validation and leaves reservation in < 20 microseconds
"""

from __future__ import annotations

import asyncio
import time
from typing import Final
from unittest.mock import AsyncMock

import pytest

from quant.execution.gateway import ExecutionGateway, PaperExecutionGateway
from quant.execution.heartbeat import (
    ConnectionStatus,
    HeartbeatConfig,
    HeartbeatWatchdog,
)
from quant.execution.kill_switch import (
    EmergencyKillSwitch,
    InvalidAdminTokenException,
    PanicMode,
    PanicTriggerReason,
)
from quant.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderState,
    OrderType,
)
from quant.execution.risk import (
    ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED,
    ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED,
    ERR_RSK_FAT_FINGER_NOTIONAL,
    ERR_RSK_FAT_FINGER_QUANTITY,
    ERR_RSK_INSUFFICIENT_MARGIN,
    ERR_RSK_KILL_SWITCH_ACTIVE,
    ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
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
    RiskLimits,
)
from quant.execution.risk_orchestrator import (
    ERR_ORCHESTRATOR_DISCONNECTED,
    ERR_ORCHESTRATOR_DRAWDOWN,
    ERR_ORCHESTRATOR_KILL_ACTIVE,
    ERR_ORCHESTRATOR_NON_FINITE,
    RiskOrchestrator,
)
from quant.execution.sor import SmartOrderRouter
from quant.execution.venues import ConsolidatedQuote, VenueProfile, VenueType

_TEST_ADMIN: Final[str] = "TEST_ADMIN_TOKEN_SECRET_123"


# ============================================================================
# Helpers & Fixtures
# ============================================================================


def make_test_limits(
    max_order_notional: float = 100_000.0,
    max_order_qty: float = 1_000.0,
    max_gross_leverage: float = 4.0,
    max_net_leverage: float = 2.0,
    max_concentration_nav_pct: float = 0.50,
    max_intraday_drawdown_pct: float = 0.05,
    min_free_margin: float = 0.0,
) -> RiskLimits:
    """Create test RiskLimits value object."""
    return RiskLimits(
        max_order_notional=max_order_notional,
        max_order_qty=max_order_qty,
        max_gross_leverage=max_gross_leverage,
        max_net_leverage=max_net_leverage,
        max_concentration_nav_pct=max_concentration_nav_pct,
        max_intraday_drawdown_pct=max_intraday_drawdown_pct,
        min_free_margin=min_free_margin,
    )


def make_test_state(
    cash: float = 100_000.0,
    positions: dict[str, float] | None = None,
    pending_leaves: dict[str, float] | None = None,
    current_prices: dict[str, float] | None = None,
    initial_equity: float = 100_000.0,
    peak_equity: float = 100_000.0,
) -> PortfolioRiskState:
    """Create test PortfolioRiskState object."""
    pos = positions if positions is not None else {}
    leaves = pending_leaves if pending_leaves is not None else {}
    prices = current_prices if current_prices is not None else {"AAPL": 100.0, "MSFT": 200.0}
    return PortfolioRiskState(
        cash=cash,
        positions=pos,
        pending_leaves=leaves,
        current_prices=prices,
        initial_equity=initial_equity,
        peak_equity=peak_equity,
    )


def make_orchestrator(
    limits: RiskLimits | None = None,
    state: PortfolioRiskState | None = None,
    router: SmartOrderRouter | None = None,
    admin_token: str = _TEST_ADMIN,
) -> RiskOrchestrator:
    """Helper to construct a fully configured RiskOrchestrator instance."""
    lim = limits if limits is not None else make_test_limits()
    st = state if state is not None else make_test_state()
    firewall = PreTradeRiskFirewall(limits=lim)
    kill_switch = EmergencyKillSwitch(admin_token=admin_token)
    return RiskOrchestrator(
        firewall=firewall,
        kill_switch=kill_switch,
        state=st,
        router=router,
        admin_token=admin_token,
    )


# ============================================================================
# Section 1: Construction, Properties & Input Sanitization
# ============================================================================


def test_orchestrator_initialization_and_properties() -> None:
    """Verify RiskOrchestrator initialization and property accessors."""
    lim = make_test_limits()
    st = make_test_state()
    fw = PreTradeRiskFirewall(limits=lim)
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN)
    orc = RiskOrchestrator(firewall=fw, kill_switch=ks, state=st, admin_token=_TEST_ADMIN)

    assert orc.firewall is fw
    assert orc.kill_switch is ks
    assert orc.state is st
    assert orc.router is None
    assert orc.default_gateway_id is None
    assert orc.gateways == {}
    assert orc.watchdogs == {}
    assert orc.is_kill_switch_active is False
    assert orc.is_kill_switch_armed is True
    assert orc.is_kill_switch_disarmed is False


def test_orchestrator_init_input_sanitization() -> None:
    """Verify RiskOrchestrator constructor rejects invalid types and booleans."""
    lim = make_test_limits()
    st = make_test_state()
    fw = PreTradeRiskFirewall(limits=lim)
    ks = EmergencyKillSwitch(admin_token=_TEST_ADMIN)

    # Invalid firewall
    with pytest.raises(NonFiniteRiskInputException) as exc1:
        RiskOrchestrator(firewall="not_firewall", kill_switch=ks, state=st)  # type: ignore[arg-type]
    assert exc1.value.code == ERR_ORCHESTRATOR_NON_FINITE

    # Invalid kill_switch
    with pytest.raises(NonFiniteRiskInputException) as exc2:
        RiskOrchestrator(firewall=fw, kill_switch="not_ks", state=st)  # type: ignore[arg-type]
    assert exc2.value.code == ERR_ORCHESTRATOR_NON_FINITE

    # Invalid state
    with pytest.raises(NonFiniteRiskInputException) as exc3:
        RiskOrchestrator(firewall=fw, kill_switch=ks, state="not_state")  # type: ignore[arg-type]
    assert exc3.value.code == ERR_ORCHESTRATOR_NON_FINITE

    # Invalid router
    with pytest.raises(NonFiniteRiskInputException) as exc4:
        RiskOrchestrator(firewall=fw, kill_switch=ks, state=st, router="not_router")  # type: ignore[arg-type]
    assert exc4.value.code == ERR_ORCHESTRATOR_NON_FINITE

    # Invalid admin token: bool or empty
    with pytest.raises(NonFiniteRiskInputException) as exc5:
        RiskOrchestrator(firewall=fw, kill_switch=ks, state=st, admin_token=True)  # type: ignore[arg-type]
    assert exc5.value.code == ERR_ORCHESTRATOR_NON_FINITE

    with pytest.raises(NonFiniteRiskInputException) as exc6:
        RiskOrchestrator(firewall=fw, kill_switch=ks, state=st, admin_token="   ")
    assert exc6.value.code == ERR_ORCHESTRATOR_NON_FINITE


def test_diagnostic_fault_codes_catalog() -> None:
    """Verify Task 4 diagnostic fault constants conform to Rule 2 catalog."""
    assert ERR_ORCHESTRATOR_DISCONNECTED == "ERR-HB-001"
    assert ERR_ORCHESTRATOR_KILL_ACTIVE == "ERR-RSK-008"
    assert ERR_ORCHESTRATOR_DRAWDOWN == "ERR-RSK-006"
    assert ERR_ORCHESTRATOR_NON_FINITE == "ERR-RSK-007"


# ============================================================================
# Section 2: Gateway & Watchdog Management
# ============================================================================


@pytest.mark.asyncio
async def test_register_and_unregister_gateway_lifecycle() -> None:
    """Verify gateway registration, retrieval, default assignment, and unregistration."""
    orc = make_orchestrator()
    gw1 = PaperExecutionGateway(initial_balance=500_000.0)
    gw2 = PaperExecutionGateway(initial_balance=500_000.0)
    await gw1.connect()
    await gw2.connect()

    # Register first gateway -> becomes default
    gid1 = orc.register_gateway(gw1, gateway_id="venue_1")
    assert gid1 == "venue_1"
    assert orc.default_gateway_id == "venue_1"
    assert orc.get_gateway("venue_1") is gw1
    assert "venue_1" in orc.gateways

    # Register second gateway with is_default=True
    gid2 = orc.register_gateway(gw2, gateway_id="venue_2", is_default=True)
    assert gid2 == "venue_2"
    assert orc.default_gateway_id == "venue_2"
    assert orc.get_gateway("venue_2") is gw2

    # Set default gateway manually
    orc.set_default_gateway("venue_1")
    assert orc.default_gateway_id == "venue_1"
    with pytest.raises(KeyError):
        orc.set_default_gateway("non_existent")

    # Unregister default gateway
    orc.unregister_gateway("venue_1")
    assert orc.get_gateway("venue_1") is None
    assert "venue_1" not in orc.gateways
    # Default re-assigned to remaining gateway
    assert orc.default_gateway_id == "venue_2"

    # Unregister second gateway
    orc.unregister_gateway("venue_2")
    assert orc.default_gateway_id is None
    assert orc.gateways == {}

    # Unregistering non-existent gateway raises KeyError
    with pytest.raises(KeyError):
        orc.unregister_gateway("venue_2")

    # Unregistering with invalid ID raises ValueError
    with pytest.raises(ValueError):
        orc.unregister_gateway("  ")
    with pytest.raises(ValueError):
        orc.unregister_gateway(True)  # type: ignore[arg-type]


def test_register_gateway_invalid_inputs() -> None:
    """Verify register_gateway rejects non-gateway objects or invalid watchdogs."""
    orc = make_orchestrator()
    with pytest.raises(TypeError):
        orc.register_gateway("not_a_gateway")  # type: ignore[arg-type]

    gw = PaperExecutionGateway()
    with pytest.raises(TypeError):
        orc.register_gateway(gw, watchdog="not_a_watchdog")  # type: ignore[arg-type]


# ============================================================================
# Section 3: Nominal Order Clearance & Execution
# ============================================================================


@pytest.mark.asyncio
async def test_nominal_buy_order_route_and_fill() -> None:
    """Verify nominal BUY order pre-trade clearance, leaves reservation, and fill state update."""
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st)
    gw = PaperExecutionGateway(initial_balance=100_000.0, fee_bps=0.0, slippage_bps=0.0)
    await gw.connect()
    gw.set_market_price("AAPL", 150.0)
    orc.register_gateway(gw, gateway_id="paper_gw")

    order = Order(
        cl_ord_id="ord-buy-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        price=150.0,
    )

    # Initial state assertions
    assert st.pending_leaves == {}
    assert st.positions.get("AAPL", 0.0) == 0.0

    # Route order
    rep = await orc.route_order(order, current_price=150.0)
    assert isinstance(rep, ExecutionReport)
    assert rep.exec_type == OrderState.FILLED
    assert rep.cum_quantity == 100.0
    assert rep.last_price == 150.0

    # State verification: leaves reconciled, cash debited, position credited
    assert st.pending_leaves.get("AAPL", 0.0) == 0.0
    assert st.positions["AAPL"] == 100.0
    assert st.cash == 100_000.0 - (100.0 * 150.0)


@pytest.mark.asyncio
async def test_nominal_sell_order_route_and_fill() -> None:
    """Verify nominal SELL order pre-trade clearance and state reconciliation."""
    st = make_test_state(
        cash=85_000.0,
        positions={"AAPL": 100.0},
        current_prices={"AAPL": 150.0},
        initial_equity=100_000.0,
        peak_equity=100_000.0,
    )
    orc = make_orchestrator(state=st)
    gw = PaperExecutionGateway(initial_balance=85_000.0, fee_bps=0.0, slippage_bps=0.0)
    await gw.connect()
    gw.set_market_price("AAPL", 150.0)
    orc.register_gateway(gw, gateway_id="paper_gw")

    order = Order(
        cl_ord_id="ord-sell-001",
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=50.0,
        price=150.0,
    )

    rep = await orc.route_order(order, current_price=150.0)
    assert isinstance(rep, ExecutionReport)
    assert rep.exec_type == OrderState.FILLED
    assert rep.cum_quantity == 50.0

    # State verification: position reduced, cash increased, leaves clean
    assert st.positions["AAPL"] == 50.0
    assert st.cash == 85_000.0 + (50.0 * 150.0)
    assert st.pending_leaves.get("AAPL", 0.0) == 0.0


@pytest.mark.asyncio
async def test_limit_order_resting_leaves_remain_reserved() -> None:
    """Verify non-marketable LIMIT order keeps open leaves reserved in portfolio state."""
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st)
    gw = PaperExecutionGateway(initial_balance=100_000.0)
    await gw.connect()
    # Market price is 150; limit buy is 140 -> rests on book (OrderState.NEW)
    gw.set_market_price("AAPL", 150.0)
    orc.register_gateway(gw, gateway_id="paper_gw")

    order = Order(
        cl_ord_id="ord-limit-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=140.0,
    )

    rep = await orc.route_order(order, current_price=150.0)
    assert isinstance(rep, ExecutionReport)
    assert rep.exec_type == OrderState.NEW

    # Pending leaves must remain reserved for resting order
    assert st.pending_leaves["AAPL"] == 50.0
    assert st.positions.get("AAPL", 0.0) == 0.0


# ============================================================================
# Section 4: Pre-Trade Risk Firewall Rejection (Gateway Untouched)
# ============================================================================


@pytest.mark.asyncio
async def test_pre_trade_firewall_fat_finger_rejections() -> None:
    """Verify fat-finger notional and quantity breaches reject before gateway dispatch."""
    lim = make_test_limits(max_order_notional=50_000.0, max_order_qty=500.0)
    orc = make_orchestrator(limits=lim)
    mock_gw = AsyncMock(spec=ExecutionGateway)
    mock_gw.is_connected = True
    orc.register_gateway(mock_gw, gateway_id="mock_gw")

    # 1. Fat finger quantity breach (qty 600 > max 500)
    order_qty_breach = Order(
        cl_ord_id="ff-qty",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=600.0,
        price=50.0,
    )
    with pytest.raises(FatFingerQuantityException) as exc1:
        await orc.route_order(order_qty_breach)
    assert exc1.value.code == ERR_RSK_FAT_FINGER_QUANTITY
    mock_gw.submit_order.assert_not_called()
    assert orc.state.pending_leaves == {}

    # 2. Fat finger notional breach (qty 400 * price 150 = 60,000 > max 50,000)
    order_notional_breach = Order(
        cl_ord_id="ff-notional",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=400.0,
        price=150.0,
    )
    with pytest.raises(FatFingerNotionalException) as exc2:
        await orc.route_order(order_notional_breach)
    assert exc2.value.code == ERR_RSK_FAT_FINGER_NOTIONAL
    mock_gw.submit_order.assert_not_called()
    assert orc.state.pending_leaves == {}


@pytest.mark.asyncio
async def test_pre_trade_firewall_leverage_and_concentration_rejections() -> None:
    """Verify leverage, concentration, and margin breaches reject before gateway dispatch."""
    mock_gw = AsyncMock(spec=ExecutionGateway)
    mock_gw.is_connected = True

    # 1. Leverage breach: cash 100k, max gross leverage 0.50 -> order 60k is 0.60x (> 0.50x)
    # Free margin is 40k >= 0 (margin passes, leverage fails)
    lim_lev = make_test_limits(
        max_gross_leverage=0.50,
        max_net_leverage=5.0,
        max_concentration_nav_pct=2.0,
        max_order_notional=1_000_000.0,
        max_order_qty=10_000.0,
    )
    st_lev = make_test_state(cash=100_000.0, initial_equity=100_000.0, peak_equity=100_000.0)
    orc_lev = make_orchestrator(limits=lim_lev, state=st_lev)
    orc_lev.register_gateway(mock_gw, gateway_id="mock_gw")

    order_lev = Order(
        cl_ord_id="lev-breach",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=600.0,
        price=100.0,
    )
    with pytest.raises(LeverageLimitExceededException) as exc1:
        await orc_lev.route_order(order_lev)
    assert exc1.value.code == ERR_RSK_LEVERAGE_LIMIT_EXCEEDED
    mock_gw.submit_order.assert_not_called()

    # 2. Concentration breach: cash 100k, max concentration 0.40 -> order 50k is 0.50 (> 0.40)
    lim_conc = make_test_limits(
        max_gross_leverage=5.0,
        max_net_leverage=5.0,
        max_concentration_nav_pct=0.40,
        max_order_notional=1_000_000.0,
        max_order_qty=10_000.0,
    )
    st_conc = make_test_state(cash=100_000.0, initial_equity=100_000.0, peak_equity=100_000.0)
    orc_conc = make_orchestrator(limits=lim_conc, state=st_conc)
    orc_conc.register_gateway(mock_gw, gateway_id="mock_gw")

    order_conc = Order(
        cl_ord_id="conc-breach",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=500.0,
        price=100.0,
    )
    with pytest.raises(ConcentrationLimitExceededException) as exc2:
        await orc_conc.route_order(order_conc)
    assert exc2.value.code == ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED
    mock_gw.submit_order.assert_not_called()

    # 3. Margin sufficiency breach: Buying power $10,000; order requires $15,000
    lim_margin = make_test_limits(
        max_gross_leverage=10.0,
        max_concentration_nav_pct=1.0,
        max_order_notional=1_000_000.0,
        max_order_qty=10_000.0,
    )
    st_margin = make_test_state(cash=10_000.0, initial_equity=10_000.0, peak_equity=10_000.0)
    orc_margin = make_orchestrator(limits=lim_margin, state=st_margin)
    orc_margin.register_gateway(mock_gw, gateway_id="mock_gw")
    order_margin = Order(
        cl_ord_id="margin-breach",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=150.0,
        price=100.0,
    )
    with pytest.raises(InsufficientMarginRiskException) as exc3:
        await orc_margin.route_order(order_margin)
    assert exc3.value.code == ERR_RSK_INSUFFICIENT_MARGIN
    mock_gw.submit_order.assert_not_called()


@pytest.mark.asyncio
async def test_pre_trade_firewall_drawdown_rejection() -> None:
    """Verify that existing drawdown breach blocks order validation immediately."""
    lim = make_test_limits(max_intraday_drawdown_pct=0.05)
    # Peak equity 100,000, current cash 90,000 -> 10% drawdown >= 5% limit
    st = make_test_state(cash=90_000.0, initial_equity=100_000.0, peak_equity=100_000.0)
    orc = make_orchestrator(limits=lim, state=st)

    order = Order(
        cl_ord_id="dd-breach",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=100.0,
    )
    with pytest.raises(DrawdownLimitExceededException) as exc:
        await orc.route_order(order)
    assert exc.value.code == ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED


# ============================================================================
# Section 5: Atomic Leaves Rollback on Gateway Exceptions & Rejections
# ============================================================================


@pytest.mark.asyncio
async def test_leaves_rollback_on_gateway_exception() -> None:
    """Verify atomic leaves rollback when gateway raises a network/socket exception."""
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 100.0})
    orc = make_orchestrator(state=st)

    failing_gw = AsyncMock(spec=ExecutionGateway)
    failing_gw.is_connected = True
    failing_gw.submit_order.side_effect = ConnectionResetError("Exchange socket terminated")
    orc.register_gateway(failing_gw, gateway_id="failing_gw")

    order = Order(
        cl_ord_id="fail-ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=100.0,
    )

    # Submission must fail and cleanly rollback leaves
    with pytest.raises(ConnectionResetError):
        await orc.route_order(order)

    # Leaves must be rolled back completely to 0
    assert "AAPL" not in st.pending_leaves
    assert st.pending_leaves == {}


@pytest.mark.asyncio
async def test_leaves_rollback_on_gateway_rejected_report() -> None:
    """Verify atomic leaves rollback when gateway returns OrderState.REJECTED."""
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 100.0})
    orc = make_orchestrator(state=st)

    rejected_report = ExecutionReport(
        report_id="rep-rej-1",
        cl_ord_id="rej-ord-001",
        exchange_order_id="ex-rej-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        exec_type=OrderState.REJECTED,
        last_quantity=0.0,
        last_price=0.0,
        cum_quantity=0.0,
        leaves_quantity=0.0,
        cum_quote_amount=0.0,
        average_price=0.0,
        fee=0.0,
        timestamp_ns=time.time_ns(),
        text="Exchange rule 404: Symbol halted",
    )
    rejecting_gw = AsyncMock(spec=ExecutionGateway)
    rejecting_gw.is_connected = True
    rejecting_gw.submit_order.return_value = rejected_report
    orc.register_gateway(rejecting_gw, gateway_id="rejecting_gw")

    order = Order(
        cl_ord_id="rej-ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=40.0,
        price=100.0,
    )

    rep = await orc.route_order(order)
    assert isinstance(rep, ExecutionReport)
    assert rep.exec_type == OrderState.REJECTED

    # Leaves must be rolled back
    assert "AAPL" not in st.pending_leaves
    assert st.pending_leaves == {}


# ============================================================================
# Section 6: SmartOrderRouter Integration & Partial Leaves Cleanup
# ============================================================================


@pytest.mark.asyncio
async def test_smart_order_router_route_slice_integration() -> None:
    """Verify route_order dispatches to SmartOrderRouter when quote is provided."""
    venues = {
        "DARK_1": VenueProfile(
            venue_id="DARK_1",
            venue_type=VenueType.DARK_POOL,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            avg_latency_ms=1.0,
        ),
        "LIT_1": VenueProfile(
            venue_id="LIT_1",
            venue_type=VenueType.LIT_EXCHANGE,
            maker_fee_bps=1.0,
            taker_fee_bps=2.0,
            avg_latency_ms=0.5,
        ),
    }
    router = SmartOrderRouter(venues=venues)
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st, router=router)

    gw = PaperExecutionGateway(initial_balance=100_000.0)
    await gw.connect()
    gw.set_market_price("AAPL", 150.05)
    orc.register_gateway(gw, gateway_id="paper_gw")

    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=time.time_ns(),
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )

    order = Order(
        cl_ord_id="sor-order-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        price=150.05,
    )

    reports = await orc.route_order(order, quote=quote)
    assert isinstance(reports, list)
    assert len(reports) >= 1
    assert reports[0].exec_type == OrderState.FILLED
    assert reports[0].cum_quantity == 100.0

    # State updated and leaves clean
    assert st.positions["AAPL"] == 100.0
    assert st.pending_leaves.get("AAPL", 0.0) == 0.0


# ============================================================================
# Section 7: Automated Tripwire Coupling
# ============================================================================


@pytest.mark.asyncio
async def test_automated_tripwire_watchdog_disconnect() -> None:
    """Verify HeartbeatWatchdog disconnect edge automatically trips EmergencyKillSwitch."""
    orc = make_orchestrator()
    gw = PaperExecutionGateway()
    await gw.connect()

    cfg = HeartbeatConfig(
        heartbeat_interval_seconds=0.1,
        timeout_seconds=0.2,
        max_consecutive_misses=2,
    )
    wd = HeartbeatWatchdog(
        gateway_id="watchdog_gw",
        config=cfg,
        initial_timestamp_ns=1_000_000_000,
    )
    orc.register_gateway(gw, watchdog=wd, gateway_id="watchdog_gw")
    assert orc.get_watchdog("watchdog_gw") is wd

    # Trigger watchdog timeout silence via check_liveness (2.0s elapsed > 0.2s timeout)
    wd.check_liveness(current_timestamp_ns=3_000_000_000)
    assert wd.status == ConnectionStatus.DISCONNECTED

    # Allow background event loop task to run panic trigger
    await asyncio.sleep(0.01)

    # Kill switch must now be PANIC_TRIGGERED with GATEWAY_DISCONNECT reason
    assert orc.is_kill_switch_active is True
    assert orc.kill_switch.last_event is not None
    assert orc.kill_switch.last_event.trigger_reason == PanicTriggerReason.GATEWAY_DISCONNECT
    assert orc.kill_switch.last_event.trigger_source == "watchdog_gw"

    # Order submission must now be locked out
    order = Order(
        cl_ord_id="locked-ord",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
    )
    with pytest.raises(KillSwitchActiveException) as exc:
        await orc.route_order(order)
    assert exc.value.code == ERR_RSK_KILL_SWITCH_ACTIVE


@pytest.mark.asyncio
async def test_explicit_handle_gateway_disconnect() -> None:
    """Verify handle_gateway_disconnect directly triggers panic and records telemetry."""
    orc = make_orchestrator()
    event = await orc.handle_gateway_disconnect(
        gateway_id="fix_broker_ny",
        elapsed_seconds=3.5,
        timestamp_ns=2_000_000_000,
    )
    assert orc.is_kill_switch_active is True
    assert event.trigger_reason == PanicTriggerReason.GATEWAY_DISCONNECT
    assert event.trigger_source == "fix_broker_ny"
    assert "3.500s elapsed" in event.details


@pytest.mark.asyncio
async def test_automated_tripwire_market_data_drawdown_breach() -> None:
    """Verify update_market_price automatically trips EmergencyKillSwitch on drawdown breach."""
    # NAV = 100,000; hold 1000 AAPL @ $100; cash = 0. max drawdown = 5% ($5,000 drop)
    st = make_test_state(
        cash=0.0,
        positions={"AAPL": 1000.0},
        current_prices={"AAPL": 100.0},
        initial_equity=100_000.0,
        peak_equity=100_000.0,
    )
    lim = make_test_limits(max_intraday_drawdown_pct=0.05)
    orc = make_orchestrator(limits=lim, state=st)

    # Nominal price tick: drops to $98 (2% drawdown < 5%) -> nominal
    await orc.update_market_price("AAPL", 98.0)
    assert st.intraday_drawdown == pytest.approx(0.02)
    assert orc.is_kill_switch_active is False

    # Crash price tick: drops to $90 (10% drawdown >= 5%) -> automated panic trigger!
    await orc.update_market_price("AAPL", 90.0)
    assert st.intraday_drawdown == pytest.approx(0.10)
    assert orc.is_kill_switch_active is True
    assert orc.kill_switch.last_event is not None
    assert orc.kill_switch.last_event.trigger_reason == PanicTriggerReason.DRAWDOWN_BREACH
    assert orc.kill_switch.last_event.trigger_source == "MarketData:AAPL"

    # Subsequent tick when kill switch already active does not crash or double-trigger
    await orc.update_market_price("AAPL", 85.0)
    assert orc.is_kill_switch_active is True


# ============================================================================
# Section 8: Administrative Controls (Arm / Disarm / Reset)
# ============================================================================


@pytest.mark.asyncio
async def test_administrative_panic_trigger_and_reset() -> None:
    """Verify trigger_emergency_panic, reset_kill_switch, and token verification."""
    orc = make_orchestrator(admin_token=_TEST_ADMIN)

    event = await orc.trigger_emergency_panic(
        reason=PanicTriggerReason.MANUAL_OPERATOR,
        source="chief_risk_officer",
        panic_mode=PanicMode.CANCEL_ONLY,
        details="Manual risk review halt",
    )
    assert orc.is_kill_switch_active is True
    assert event.trigger_source == "chief_risk_officer"

    # Reset with invalid token raises InvalidAdminTokenException
    with pytest.raises(InvalidAdminTokenException):
        orc.reset_kill_switch("WRONG_TOKEN")

    # Reset with correct token re-arms system
    orc.reset_kill_switch(_TEST_ADMIN)
    assert orc.is_kill_switch_active is False
    assert orc.is_kill_switch_armed is True


def test_administrative_disarm_and_arm() -> None:
    """Verify disarm_kill_switch and arm_kill_switch state transitions."""
    orc = make_orchestrator(admin_token=_TEST_ADMIN)

    # Disarm
    orc.disarm_kill_switch(_TEST_ADMIN)
    assert orc.is_kill_switch_disarmed is True
    assert orc.is_kill_switch_armed is False

    # Re-arm
    orc.arm_kill_switch(_TEST_ADMIN)
    assert orc.is_kill_switch_armed is True
    assert orc.is_kill_switch_disarmed is False

    # Invalid tokens
    with pytest.raises(InvalidAdminTokenException):
        orc.disarm_kill_switch("BAD_TOKEN")
    with pytest.raises(InvalidAdminTokenException):
        orc.arm_kill_switch("BAD_TOKEN")


# ============================================================================
# Section 9: Sub-20us Hot-Path Latency SLA Benchmark
# ============================================================================


def test_hot_path_orchestration_latency_sla() -> None:
    """Verify combined pre-trade clearance, leaves reservation, and kill validation executes in < 20us."""
    st = make_test_state(cash=1_000_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st)

    order = Order(
        cl_ord_id="perf-test-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=150.0,
    )

    # Warmup
    for _ in range(50):
        orc.kill_switch.validate_submission()
        orc.firewall.validate_order(order, st, current_price=150.0)

    iterations = 2000
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        # Hot-path sequence: validate submission + validate risk firewall + reserve leaves
        orc.kill_switch.validate_submission()
        orc.firewall.validate_order(order, st, current_price=150.0)
        st.pending_leaves["AAPL"] = st.pending_leaves.get("AAPL", 0.0) + order.quantity
        # Rollback for loop repeatability
        st.pending_leaves["AAPL"] -= order.quantity
    elapsed_ns = time.perf_counter_ns() - start_ns

    avg_latency_us = (elapsed_ns / iterations) / 1_000.0
    # Must strictly beat 20 microseconds SLA (typically < 3us)
    assert avg_latency_us < 20.0, f"Hot path overhead {avg_latency_us:.2f}us exceeds 20us SLA"


# ============================================================================
# Section 10: Adversarial Red-Teaming & Error Invariants
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_red_teaming_missing_gateways() -> None:
    """Verify route_order raises RuntimeError when no execution gateways are registered."""
    orc = make_orchestrator()
    order = Order(
        cl_ord_id="no-gw-ord",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        price=100.0,
    )

    with pytest.raises(RuntimeError) as exc1:
        await orc.route_order(order)
    assert "No execution gateways registered" in str(exc1.value)

    # Unknown explicit target gateway
    gw = PaperExecutionGateway()
    orc.register_gateway(gw, gateway_id="gw_alpha")
    with pytest.raises(RuntimeError) as exc2:
        await orc.route_order(order, target_gateway_id="gw_beta")
    assert "Target execution gateway 'gw_beta' is not registered" in str(exc2.value)


@pytest.mark.asyncio
async def test_adversarial_non_finite_price_ticks() -> None:
    """Verify update_market_price strictly rejects NaNs, infinities, booleans, and negative values."""
    orc = make_orchestrator()

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("", 150.0)

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("AAPL", float("nan"))

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("AAPL", float("inf"))

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("AAPL", -150.0)

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("AAPL", 0.0)

    with pytest.raises(NonFiniteRiskInputException):
        await orc.update_market_price("AAPL", True)  # type: ignore[arg-type]


def test_zero_iterative_solvers_invariant() -> None:
    """Verify Rule 4 invariant: zero iterative numerical solvers in risk orchestrator."""
    import inspect

    import quant.execution.risk_orchestrator as ro_module

    source = inspect.getsource(ro_module)
    assert "scipy.optimize" not in source
    assert "np.linalg.solve" not in source
    assert "scipy" not in source


def test_unregister_gateway_with_attached_watchdog() -> None:
    """Verify unregistering a gateway with an attached watchdog deregisters listeners cleanly."""
    orc = make_orchestrator()
    gw = PaperExecutionGateway()
    wd = HeartbeatWatchdog(gateway_id="wd_cleanup_gw")
    orc.register_gateway(gw, watchdog=wd, gateway_id="wd_cleanup_gw")

    assert "wd_cleanup_gw" in orc.watchdogs
    assert "wd_cleanup_gw" in orc.gateways

    # Unregister
    orc.unregister_gateway("wd_cleanup_gw")
    assert "wd_cleanup_gw" not in orc.watchdogs
    assert "wd_cleanup_gw" not in orc.gateways
    assert orc.get_watchdog("wd_cleanup_gw") is None


def test_watchdog_disconnect_sync_execution_outside_event_loop() -> None:
    """Verify watchdog timeout callback executes synchronously when no event loop is running."""
    orc = make_orchestrator()
    gw = PaperExecutionGateway()
    cfg = HeartbeatConfig(timeout_seconds=0.1, heartbeat_interval_seconds=0.05)
    wd = HeartbeatWatchdog(gateway_id="sync_gw", config=cfg, initial_timestamp_ns=1_000_000_000)
    orc.register_gateway(gw, watchdog=wd, gateway_id="sync_gw")

    # Call check_liveness synchronously in non-async test
    wd.check_liveness(current_timestamp_ns=3_000_000_000)
    assert wd.status == ConnectionStatus.DISCONNECTED
    # Kill switch should be PANIC_TRIGGERED via asyncio.run branch
    assert orc.is_kill_switch_active is True
    assert orc.kill_switch.last_event is not None
    assert orc.kill_switch.last_event.trigger_reason == PanicTriggerReason.GATEWAY_DISCONNECT


@pytest.mark.asyncio
async def test_smart_order_router_partial_fill_leaves_cleanup() -> None:
    """Verify unfulfilled residual leaves are cleaned up when router returns partial fill."""
    venues = {
        "LIT_1": VenueProfile(
            venue_id="LIT_1",
            venue_type=VenueType.LIT_EXCHANGE,
            maker_fee_bps=1.0,
            taker_fee_bps=2.0,
            avg_latency_ms=0.5,
        ),
    }
    router = SmartOrderRouter(venues=venues)
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st, router=router)

    # Mock gateway that returns a partial fill (40 shares out of 100)
    mock_gw = AsyncMock(spec=ExecutionGateway)
    mock_gw.is_connected = True
    partial_rep = ExecutionReport(
        report_id="rep-part-1",
        cl_ord_id="sor-lit-LIT_1-12345",
        exchange_order_id="ex-part-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        exec_type=OrderState.PARTIALLY_FILLED,
        last_quantity=40.0,
        last_price=150.0,
        cum_quantity=40.0,
        leaves_quantity=0.0,
        cum_quote_amount=6000.0,
        average_price=150.0,
        fee=1.2,
        timestamp_ns=time.time_ns(),
        text="Partial fill",
    )
    mock_gw.submit_order.return_value = partial_rep
    orc.register_gateway(mock_gw, gateway_id="mock_lit_gw")

    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=time.time_ns(),
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )

    order = Order(
        cl_ord_id="sor-part-order",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        price=150.10,
    )

    reports = await orc.route_order(order, quote=quote)
    assert len(reports) == 1
    assert reports[0].cum_quantity == 40.0

    # Leaves must be 0 (40 reconciled by update_fill, 60 rolled back by residual cleaner)
    assert st.pending_leaves.get("AAPL", 0.0) == 0.0
    assert st.positions["AAPL"] == 40.0


def test_resolve_gateway_fallback_to_any_registered() -> None:
    """Verify _resolve_gateway falls back to first gateway if default_gateway_id is None."""
    orc = make_orchestrator()
    gw = PaperExecutionGateway()
    orc.register_gateway(gw, gateway_id="gw_sole")
    orc._default_gateway_id = None

    resolved = orc._resolve_gateway(None)
    assert resolved is gw


@pytest.mark.asyncio
async def test_route_order_with_explicit_target_gateway_id() -> None:
    """Verify route_order successfully resolves explicit target_gateway_id."""
    st = make_test_state(cash=100_000.0, current_prices={"AAPL": 150.0})
    orc = make_orchestrator(state=st)
    gw1 = PaperExecutionGateway(initial_balance=100_000.0, fee_bps=0.0, slippage_bps=0.0)
    gw2 = PaperExecutionGateway(initial_balance=100_000.0, fee_bps=0.0, slippage_bps=0.0)
    await gw1.connect()
    await gw2.connect()
    gw2.set_market_price("AAPL", 150.0)

    orc.register_gateway(gw1, gateway_id="gw_primary")
    orc.register_gateway(gw2, gateway_id="gw_secondary")

    order = Order(
        cl_ord_id="ord-target-gw",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        price=150.0,
    )
    rep = await orc.route_order(order, target_gateway_id="gw_secondary", current_price=150.0)
    assert isinstance(rep, ExecutionReport)
    assert rep.exec_type == OrderState.FILLED
    assert rep.cum_quantity == 10.0
