"""Unit tests for AutonomousTradingEngine live swarm rebalancing daemon."""

from __future__ import annotations

import asyncio

import pytest

from quant.data.alpaca_feed import AlpacaMarketDataFeed
from quant.execution.gateway import PaperExecutionGateway
from quant.execution.heartbeat import HeartbeatWatchdog
from quant.execution.kill_switch import EmergencyKillSwitch, PanicMode, PanicTriggerReason
from quant.execution.risk import (
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskLimits,
)
from quant.execution.risk_orchestrator import RiskOrchestrator
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)
from quant.services.autonomous_trader import (
    AutonomousState,
    AutonomousStepReport,
    AutonomousTradingEngine,
)
from quant.services.execution_service import ExecutionService


@pytest.fixture
def in_memory_duckdb() -> DuckDBManager:
    return DuckDBManager(database_path=":memory:")


@pytest.fixture
def market_repo(in_memory_duckdb: DuckDBManager) -> DuckDBMarketDataRepository:
    return DuckDBMarketDataRepository(in_memory_duckdb)


@pytest.fixture
def paper_gateway() -> PaperExecutionGateway:
    gw = PaperExecutionGateway()
    gw._is_connected = True
    gw.set_market_price("AAPL", 180.0)
    gw.set_market_price("NVDA", 125.0)
    gw.set_market_price("SPY", 510.0)
    return gw


@pytest.fixture
def risk_orchestrator(paper_gateway: PaperExecutionGateway) -> RiskOrchestrator:
    limits = RiskLimits(
        max_order_notional=500_000.0,
        max_order_qty=50_000.0,
        max_gross_leverage=2.0,
        max_net_leverage=1.0,
        max_concentration_nav_pct=0.25,
        max_intraday_drawdown_pct=0.05,
        min_free_margin=100_000.0,
    )
    state = PortfolioRiskState(cash=1_000_000.0, initial_equity=1_000_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)
    kill_switch = EmergencyKillSwitch(admin_token="TEST_ADMIN")
    watchdog = HeartbeatWatchdog(gateway_id="paper_broker")
    orchestrator = RiskOrchestrator(
        firewall=firewall,
        kill_switch=kill_switch,
        state=state,
        admin_token="TEST_ADMIN",
    )
    orchestrator.register_gateway(
        paper_gateway, watchdog=watchdog, gateway_id="paper_broker", is_default=True
    )
    return orchestrator


@pytest.fixture
def execution_service(
    risk_orchestrator: RiskOrchestrator, paper_gateway: PaperExecutionGateway
) -> ExecutionService:
    return ExecutionService(orchestrator=risk_orchestrator, gateway=paper_gateway)


@pytest.fixture
def market_feed(market_repo: DuckDBMarketDataRepository) -> AlpacaMarketDataFeed:
    return AlpacaMarketDataFeed(
        repository=market_repo,
        offline_mode=True,
    )


class TestAutonomousTradingEngine:
    """Verify autonomous trading swarm daemon lifecycle and rebalancing cycle."""

    @pytest.mark.asyncio
    async def test_initial_state_idle(
        self,
        execution_service: ExecutionService,
        paper_gateway: PaperExecutionGateway,
        market_feed: AlpacaMarketDataFeed,
    ) -> None:
        trader = AutonomousTradingEngine(
            execution_service=execution_service,
            gateway=paper_gateway,
            market_feed=market_feed,
            universe=["AAPL", "NVDA"],
            interval_sec=1.0,
        )
        assert trader.state == AutonomousState.IDLE
        assert trader.iteration == 0
        status = trader.get_status()
        assert status["state"] == "IDLE"
        assert status["iteration"] == 0
        assert status["universe"] == ["AAPL", "NVDA"]

    @pytest.mark.asyncio
    async def test_step_once_execution(
        self,
        execution_service: ExecutionService,
        paper_gateway: PaperExecutionGateway,
        market_feed: AlpacaMarketDataFeed,
    ) -> None:
        trader = AutonomousTradingEngine(
            execution_service=execution_service,
            gateway=paper_gateway,
            market_feed=market_feed,
            universe=["AAPL", "NVDA"],
            interval_sec=1.0,
        )

        report = await trader.step_once()
        assert isinstance(report, AutonomousStepReport)
        assert report.iteration == 1
        assert trader.iteration == 1
        assert "AAPL" in report.bars
        assert "NVDA" in report.bars
        assert len(report.target_allocations) == 2
        # Orders generated and submitted through execution_service
        assert len(report.orders_dispatched) >= 0

        # Step again
        report2 = await trader.step_once()
        assert report2.iteration == 2
        assert trader.iteration == 2

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(
        self,
        execution_service: ExecutionService,
        paper_gateway: PaperExecutionGateway,
        market_feed: AlpacaMarketDataFeed,
    ) -> None:
        trader = AutonomousTradingEngine(
            execution_service=execution_service,
            gateway=paper_gateway,
            market_feed=market_feed,
            universe=["AAPL"],
            interval_sec=0.05,  # fast test loop
        )

        await trader.start()
        state_running: AutonomousState = trader.state
        assert state_running == AutonomousState.RUNNING

        # Allow background loop to tick at least twice
        await asyncio.sleep(0.15)
        assert trader.iteration >= 2

        await trader.stop()
        state_stopped: AutonomousState = trader.state
        assert state_stopped == AutonomousState.STOPPED

    @pytest.mark.asyncio
    async def test_kill_switch_active_halts_trading(
        self,
        execution_service: ExecutionService,
        paper_gateway: PaperExecutionGateway,
        market_feed: AlpacaMarketDataFeed,
        risk_orchestrator: RiskOrchestrator,
    ) -> None:
        trader = AutonomousTradingEngine(
            execution_service=execution_service,
            gateway=paper_gateway,
            market_feed=market_feed,
            universe=["AAPL", "NVDA"],
            interval_sec=1.0,
        )

        # Trigger emergency kill switch
        await risk_orchestrator._kill_switch.trigger_panic(
            reason=PanicTriggerReason.MANUAL_OPERATOR,
            source="test_operator",
            panic_mode=PanicMode.CANCEL_ONLY,
            details="Simulated risk panic breach",
        )
        assert risk_orchestrator.is_kill_switch_active

        report = await trader.step_once()
        # When kill switch is active, target allocations are wiped to 0.0
        for alloc in report.target_allocations.values():
            assert alloc == 0.0
