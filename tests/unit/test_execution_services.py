"""Comprehensive unit tests for ExecutionService and RiskService.

Purpose:
    Verifies that the application service layer correctly orchestrates parent orders,
    algorithmic meta-order schedulers (Poisson TWAP, Volume Adaptive VWAP, Arrival Price),
    pre-trade risk firewall boundaries, panic kill switch triggering/resetting, and
    Perold (1988) implementation shortfall TCA attribution.

Invariants Enforced:
    - INV-SOR-001: Parent-child mass conservation strictly maintained across all slices.
    - INV-SOR-005: Perold Implementation Shortfall additive identity holds within 1e-7.
    - INV-RSK-008: Emergency kill switch lockouts and constant-time token verification.
"""

from __future__ import annotations

import pytest

from quant.api.v1.schemas import (
    ParentOrderCreateRequest,
    RiskLimitsUpdateRequest,
)
from quant.execution.gateway import PaperExecutionGateway
from quant.execution.heartbeat import HeartbeatWatchdog
from quant.execution.kill_switch import EmergencyKillSwitch, PanicTriggerReason
from quant.execution.risk import (
    NonFiniteRiskInputException,
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskLimits,
)
from quant.execution.risk_orchestrator import RiskOrchestrator
from quant.execution.venues import (
    ConsolidatedQuote,
    InvalidSORInputException,
    NonFiniteInputException,
)
from quant.services.execution_service import ExecutionService
from quant.services.risk_service import RiskService


@pytest.fixture
def risk_orchestrator() -> RiskOrchestrator:
    limits = RiskLimits(
        max_order_notional=500_000.0,
        max_order_qty=50_000.0,
        max_gross_leverage=2.0,
        max_net_leverage=1.0,
        max_concentration_nav_pct=0.50,
        max_intraday_drawdown_pct=0.05,
        min_free_margin=50_000.0,
    )
    state = PortfolioRiskState(cash=1_000_000.0, initial_equity=1_000_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)
    kill_switch = EmergencyKillSwitch(admin_token="SECRET_ADMIN_TOKEN")
    orchestrator = RiskOrchestrator(
        firewall=firewall,
        kill_switch=kill_switch,
        state=state,
        admin_token="SECRET_ADMIN_TOKEN",
    )
    gateway = PaperExecutionGateway(slippage_bps=1.0, fee_bps=0.5, latency_ms=0.0)
    gateway._is_connected = True
    gateway.set_market_price("AAPL", 184.50)
    watchdog = HeartbeatWatchdog(gateway_id="PAPER_BROKER")
    orchestrator.register_gateway(
        gateway, watchdog=watchdog, gateway_id="PAPER_BROKER", is_default=True
    )
    return orchestrator


@pytest.fixture
def mock_quote() -> ConsolidatedQuote:
    return ConsolidatedQuote(
        symbol="AAPL",
        bid_price=184.48,
        bid_quantity=1000.0,
        ask_price=184.52,
        ask_quantity=1200.0,
        timestamp_ns=1_700_000_000_000_000_000,
        venue_depths={"NASDAQ": (1000.0, 1200.0)},
    )


@pytest.fixture
def execution_service(risk_orchestrator: RiskOrchestrator) -> ExecutionService:
    gateway = risk_orchestrator.gateways["PAPER_BROKER"]
    return ExecutionService(orchestrator=risk_orchestrator, gateway=gateway)


@pytest.fixture
def risk_service(risk_orchestrator: RiskOrchestrator) -> RiskService:
    return RiskService(orchestrator=risk_orchestrator)


# ============================================================================
# ExecutionService Tests
# ============================================================================


def test_execution_service_init_validation() -> None:
    gateway = PaperExecutionGateway()
    with pytest.raises(NonFiniteInputException):
        ExecutionService(orchestrator=None, gateway=gateway)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_submit_parent_order_twap_sync(
    execution_service: ExecutionService, mock_quote: ConsolidatedQuote
) -> None:
    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="BUY",
        order_type="LIMIT",
        quantity=500.0,
        price=184.55,
        algorithm="POISSON_TWAP",
        horizon_seconds=10.0,
        num_slices=4,
        algo_params={"timing_jitter_pct": 0.1, "volume_jitter_pct": 0.1, "rng_seed": 42},
    )

    parent_order = await execution_service.submit_parent_order_sync(req, quote=mock_quote)

    assert parent_order.symbol == "AAPL"
    assert parent_order.total_quantity == 500.0
    assert parent_order.filled_quantity == 500.0
    assert parent_order.is_completed is True
    assert parent_order.average_execution_price > 0.0

    # Shortfall TCA report
    report = execution_service.get_shortfall_report(parent_order.parent_id, terminal_price=184.55)
    assert report is not None
    assert (
        abs(
            report.total_shortfall
            - (report.delay_cost + report.price_impact + report.fees_paid + report.opportunity_cost)
        )
        <= 1e-7
    )
    assert report.filled_quantity == 500.0

    # DTO Response
    resp = execution_service.to_order_response(parent_order)
    assert resp.order_id == parent_order.parent_id
    assert resp.filled_quantity == 500.0
    assert len(resp.child_fills) == 4

    shortfall_resp = execution_service.to_shortfall_response(report)
    assert shortfall_resp.order_id == parent_order.parent_id
    assert shortfall_resp.is_additive_conserved is True


@pytest.mark.asyncio
async def test_submit_parent_order_vwap_sync(
    execution_service: ExecutionService, mock_quote: ConsolidatedQuote
) -> None:
    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="BUY",
        order_type="LIMIT",
        quantity=300.0,
        price=184.55,
        algorithm="VOLUME_ADAPTIVE_VWAP",
        horizon_seconds=10.0,
        num_slices=3,
        algo_params={
            "historical_volume_profile": [1000.0, 1000.0, 1000.0],
            "max_participation_rate": 0.15,
        },
    )

    parent_order = await execution_service.submit_parent_order_sync(req, quote=mock_quote)
    assert parent_order.total_quantity == 300.0
    assert parent_order.filled_quantity == 300.0
    assert parent_order.is_completed is True


@pytest.mark.asyncio
async def test_submit_parent_order_arrival_price_sync(
    execution_service: ExecutionService, mock_quote: ConsolidatedQuote
) -> None:
    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="BUY",
        order_type="LIMIT",
        quantity=400.0,
        price=184.55,
        algorithm="ARRIVAL_PRICE",
        horizon_seconds=10.0,
        num_slices=4,
        algo_params={"volatility": 0.02, "urgency_parameter": 1.0},
    )

    parent_order = await execution_service.submit_parent_order_sync(req, quote=mock_quote)
    assert parent_order.total_quantity == 400.0
    assert parent_order.filled_quantity == 400.0


@pytest.mark.asyncio
async def test_submit_parent_order_direct_market(
    execution_service: ExecutionService, mock_quote: ConsolidatedQuote
) -> None:
    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="SELL",
        order_type="MARKET",
        quantity=150.0,
        algorithm="DIRECT_MARKET",
        horizon_seconds=5.0,
        num_slices=1,
    )

    parent_order = await execution_service.submit_parent_order_sync(req, quote=mock_quote)
    assert parent_order.total_quantity == 150.0
    assert parent_order.filled_quantity == 150.0


@pytest.mark.asyncio
async def test_order_query_and_cancellation(
    execution_service: ExecutionService, mock_quote: ConsolidatedQuote
) -> None:
    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="BUY",
        order_type="LIMIT",
        quantity=100.0,
        price=184.50,
        algorithm="POISSON_TWAP",
        horizon_seconds=60.0,
        num_slices=5,
    )

    order = execution_service.submit_parent_order(req, quote=mock_quote)
    order_id = order.parent_id

    # Query single order
    queried = execution_service.get_order(order_id)
    assert queried is not None
    assert queried.parent_id == order_id

    # List orders
    all_orders = execution_service.list_orders(symbol="AAPL", limit=10)
    assert len(all_orders) >= 1
    assert all_orders[0].parent_id == order_id

    # Cancel order
    cancelled = execution_service.cancel_order(order_id)
    assert cancelled is True

    # Unknown order cancel
    assert execution_service.cancel_order("UNKNOWN-ORDER-ID") is False


@pytest.mark.asyncio
async def test_submission_lockout_under_kill_switch(
    execution_service: ExecutionService,
    risk_orchestrator: RiskOrchestrator,
    mock_quote: ConsolidatedQuote,
) -> None:
    risk_orchestrator.kill_switch.arm("SECRET_ADMIN_TOKEN")
    # Trigger kill switch
    await risk_orchestrator.kill_switch.trigger_panic(
        reason=PanicTriggerReason.MANUAL_OPERATOR,
        source="TEST",
        details="Test Panic",
    )

    req = ParentOrderCreateRequest(
        symbol="AAPL",
        side="BUY",
        order_type="LIMIT",
        quantity=100.0,
        price=184.50,
        algorithm="POISSON_TWAP",
    )

    with pytest.raises(InvalidSORInputException):
        execution_service.submit_parent_order(req, quote=mock_quote)


# ============================================================================
# RiskService Tests
# ============================================================================


def test_risk_service_init_validation() -> None:
    with pytest.raises(NonFiniteRiskInputException):
        RiskService(orchestrator=None)  # type: ignore[arg-type]


def test_risk_status_and_limits(risk_service: RiskService) -> None:
    status = risk_service.get_risk_status()
    assert status.nav == 1_000_000.0
    assert status.cash == 1_000_000.0
    assert status.gross_leverage == 0.0
    assert status.is_kill_switch_active is False

    limits = risk_service.get_risk_limits()
    assert limits.max_order_notional == 500_000.0
    assert limits.max_gross_leverage == 2.0


def test_update_risk_limits(risk_service: RiskService) -> None:
    update_req = RiskLimitsUpdateRequest(
        max_order_notional=600_000.0,
        max_gross_leverage=1.8,
    )
    new_limits = risk_service.update_risk_limits(update_req)
    assert new_limits.max_order_notional == 600_000.0
    assert new_limits.max_gross_leverage == 1.8
    assert new_limits.max_order_qty == 50_000.0  # Unchanged


@pytest.mark.asyncio
async def test_trigger_panic_and_reset(risk_service: RiskService) -> None:
    # 1. Trigger panic
    event = await risk_service.trigger_panic("Emergency rogue fill detected")
    assert event is not None
    assert risk_service.get_risk_status().is_kill_switch_active is True

    # 2. Reset with invalid token -> fails
    assert risk_service.reset_kill_switch("WRONG_TOKEN") is False
    assert risk_service.get_risk_status().is_kill_switch_active is True

    # 3. Reset with correct token -> succeeds
    assert risk_service.reset_kill_switch("SECRET_ADMIN_TOKEN") is True
    assert risk_service.get_risk_status().is_kill_switch_active is False


def test_gateway_health_monitoring(risk_service: RiskService) -> None:
    health_list = risk_service.get_gateway_health()
    assert len(health_list) == 1
    assert health_list[0].gateway_id == "PAPER_BROKER"
    assert health_list[0].is_connected is True

    # Filter by specific gateway
    filtered = risk_service.get_gateway_health("PAPER_BROKER")
    assert len(filtered) == 1

    # Filter by unknown gateway
    none_found = risk_service.get_gateway_health("UNKNOWN_BROKER")
    assert len(none_found) == 0


@pytest.mark.asyncio
async def test_update_market_price(risk_service: RiskService) -> None:
    await risk_service.update_market_price("AAPL", 185.00)
    assert risk_service.orchestrator.state.current_prices["AAPL"] == 185.00
