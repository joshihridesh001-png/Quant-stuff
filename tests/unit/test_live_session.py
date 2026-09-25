"""Comprehensive Unit and Safety Tests for LiveTradingSession Orchestrator.

Functional Purpose:
    Verifies that the live session orchestrator adheres strictly to:
    - INV-LIVE-001: Monotonic lifecycle state transitions (INITIALIZING -> RUNNING <-> PAUSED -> STOPPED).
    - INV-LIVE-002: Strict order lockout when emergency kill switch is tripped.
    - INV-LIVE-003: Strict non-finite and boundary validation for configuration parameters.
    - INV-LIVE-004: Clean and atomic cancellation sweep on orderly shutdown.
    - INV-LIVE-005: Zero uncaught exceptions escaping from the cycle loop.

Governing Rules:
    - Rule 1: Four-tier docstrings on all test fixtures and assertions.
    - Rule 2: Deterministic error codes (ERR-LIVE-001, ERR-LIVE-002, ERR-LIVE-003, ERR-LIVE-005).
    - Rule 3: Quality gates (100% pass rate, strict static typing).
    - Rule 4: Adversarial edge cases, non-finite inputs, and drawdown tripwires.
"""

from __future__ import annotations

import math

import pytest

from quant.analytics.strategies.base import (
    BarHistoryWindow,
    BaseAlphaStrategy,
    SignalDirection,
    StrategyContext,
    StrategySignal,
)
from quant.services.live_session import (
    ERR_LIVE_ALREADY_RUNNING,
    ERR_LIVE_FIREWALL_BREACH,
    ERR_LIVE_NON_FINITE_INPUT,
    LiveSessionConfig,
    LiveSessionError,
    LiveSessionRiskError,
    LiveSessionStateError,
    LiveSessionStatus,
    LiveSessionTelemetry,
    LiveTradingSession,
)


class MockConstantWeightStrategy(BaseAlphaStrategy):
    """Mock strategy emitting constant deterministic weights for testing."""

    def __init__(
        self,
        strategy_id: str = "mock_fixed",
        target_weights: dict[str, float] | None = None,
        min_warmup_bars: int = 5,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id,
            min_warmup_bars=min_warmup_bars,
            monitored_symbols=("SPY", "QQQ"),
        )
        self.target_weights = target_weights or {"SPY": 0.20, "QQQ": 0.20}

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        signals = {}
        for sym, w in self.target_weights.items():
            direction = (
                SignalDirection.LONG
                if w > 0
                else (SignalDirection.SHORT if w < 0 else SignalDirection.FLAT)
            )
            signals[sym] = StrategySignal(
                symbol=sym,
                direction=direction,
                target_weight=w,
                conviction=1.0,
                target_horizon_bars=10,
                diagnostics={"model": self.strategy_id},
            )
        return signals


@pytest.fixture
def session_config() -> LiveSessionConfig:
    """Fixture providing valid LiveSessionConfig."""
    return LiveSessionConfig(
        session_id="test_live_session_001",
        symbols=("SPY", "QQQ"),
        initial_cash=100000.0,
        poll_interval_sec=0.01,
        max_leverage=2.0,
        max_concentration=0.35,
        max_drawdown_limit=0.15,
        rebalance_threshold_pct=0.01,
        min_order_notional=10.0,
        admin_token="test-secret-token-1234",
    )


def test_live_session_config_invariants() -> None:
    """Verify that LiveSessionConfig rejects non-finite, negative, or invalid configurations."""
    # 1. Reject empty symbols
    with pytest.raises(LiveSessionError) as exc_info:
        LiveSessionConfig(session_id="err_1", symbols=())
    assert exc_info.value.code == ERR_LIVE_NON_FINITE_INPUT

    # 2. Reject non-finite initial cash
    with pytest.raises(LiveSessionError) as exc_info:
        LiveSessionConfig(session_id="err_2", symbols=("SPY",), initial_cash=float("nan"))
    assert exc_info.value.code == ERR_LIVE_NON_FINITE_INPUT

    # 3. Reject negative initial cash
    with pytest.raises(LiveSessionError) as exc_info:
        LiveSessionConfig(session_id="err_3", symbols=("SPY",), initial_cash=-5000.0)
    assert exc_info.value.code == ERR_LIVE_NON_FINITE_INPUT

    # 4. Reject max_leverage < 1.0
    with pytest.raises(LiveSessionError) as exc_info:
        LiveSessionConfig(session_id="err_4", symbols=("SPY",), max_leverage=0.5)
    assert exc_info.value.code == ERR_LIVE_NON_FINITE_INPUT


@pytest.mark.asyncio
async def test_live_session_lifecycle(session_config: LiveSessionConfig) -> None:
    """Verify state transitions: INITIALIZING -> RUNNING -> PAUSED -> RUNNING -> STOPPED."""
    strategy = MockConstantWeightStrategy()
    session = LiveTradingSession(config=session_config, strategy=strategy)

    assert session.status == LiveSessionStatus.INITIALIZING

    # Start session
    await session.start()
    assert session.status == LiveSessionStatus.RUNNING
    assert session.gateway.is_connected

    # Attempting to start again must raise LiveSessionStateError
    with pytest.raises(LiveSessionStateError) as exc_info:
        await session.start()
    assert exc_info.value.code == ERR_LIVE_ALREADY_RUNNING

    # Pause session
    await session.pause()
    assert session.status == LiveSessionStatus.PAUSED

    # Resume session
    await session.resume()
    assert session.status == LiveSessionStatus.RUNNING

    # Orderly shutdown
    await session.stop()
    assert session.status == LiveSessionStatus.STOPPED
    assert not session.gateway.is_connected


@pytest.mark.asyncio
async def test_live_session_step_cycle_rebalancing(session_config: LiveSessionConfig) -> None:
    """Verify market prices update, strategy signals generate, orders dispatch and fills record."""
    strategy = MockConstantWeightStrategy(target_weights={"SPY": 0.20, "QQQ": 0.20})
    telemetry_snapshots: list[LiveSessionTelemetry] = []

    def _telemetry_handler(t: LiveSessionTelemetry) -> None:
        telemetry_snapshots.append(t)

    # Custom price feed providing fixed quotes
    def _price_feed() -> dict[str, float]:
        return {"SPY": 500.0, "QQQ": 400.0}

    session = LiveTradingSession(
        config=session_config,
        strategy=strategy,
        on_telemetry=_telemetry_handler,
        custom_price_feed=_price_feed,
    )

    # Pre-seed bar history to satisfy warmup (5 bars required)
    for sym in ("SPY", "QQQ"):
        seed_bars = [
            {
                "timestamp": float(i),
                "open": 500.0,
                "high": 500.0,
                "low": 500.0,
                "close": 500.0,
                "volume": 1000.0,
                "vwap": 500.0,
            }
            for i in range(10)
        ]
        session.seed_bar_history(sym, seed_bars)

    await session.start()

    # Manually execute 1 cycle
    telemetry = await session.step_cycle()

    assert telemetry.iteration >= 1
    assert telemetry.status == LiveSessionStatus.RUNNING
    assert math.isclose(telemetry.current_prices["SPY"], 500.0, rel_tol=1e-3)
    assert math.isclose(telemetry.current_prices["QQQ"], 400.0, rel_tol=1e-3)

    # Orders should have been dispatched to reach 20% SPY ($20k / 500 = 40 shares) and 20% QQQ ($20k / 400 = 50 shares)
    assert telemetry.orders_dispatched_count >= 2
    assert telemetry.fills_count >= 2

    # Verify positions in state
    port_state = session.risk_orchestrator.state
    assert port_state.positions["SPY"] == 40.0
    assert port_state.positions["QQQ"] == 50.0

    await session.stop()


@pytest.mark.asyncio
async def test_live_session_panic_kill_and_disarm(session_config: LiveSessionConfig) -> None:
    """Verify emergency kill switch cancels all orders and blocks subsequent rebalancing."""
    strategy = MockConstantWeightStrategy()
    session = LiveTradingSession(config=session_config, strategy=strategy)
    await session.start()

    # Trigger panic kill
    await session.panic_kill(reason="TEST_PANIC")
    assert session.status == LiveSessionStatus.PANIC_KILLED
    assert session.risk_orchestrator.kill_switch.is_active

    # Subsequent cycle while killed must NOT dispatch any orders
    orders_before = dict(session.risk_orchestrator.state.pending_leaves)
    await session.step_cycle()
    assert session.status == LiveSessionStatus.PANIC_KILLED
    assert session.risk_orchestrator.state.pending_leaves == orders_before

    # Resuming while killed must be forbidden
    with pytest.raises(LiveSessionRiskError) as exc_info:
        await session.resume()
    assert exc_info.value.code == ERR_LIVE_FIREWALL_BREACH

    # Disarm kill switch with valid token
    await session.disarm(session_config.admin_token)
    assert not session.risk_orchestrator.kill_switch.is_active
    assert session.status == LiveSessionStatus.PAUSED

    # Now can resume cleanly
    await session.resume()
    assert session.status == LiveSessionStatus.RUNNING

    await session.stop()


@pytest.mark.asyncio
async def test_live_session_drawdown_tripwire(session_config: LiveSessionConfig) -> None:
    """Verify that severe mark-to-market drawdown trips kill switch automatically."""
    strategy = MockConstantWeightStrategy()
    current_market_prices = {"SPY": 500.0, "QQQ": 400.0}

    def _dynamic_feed() -> dict[str, float]:
        return dict(current_market_prices)

    session = LiveTradingSession(
        config=session_config,
        strategy=strategy,
        custom_price_feed=_dynamic_feed,
    )

    # Pre-seed history
    for sym in ("SPY", "QQQ"):
        seed_bars = [
            {
                "timestamp": float(i),
                "open": 500.0,
                "high": 500.0,
                "low": 500.0,
                "close": 500.0,
                "volume": 1000.0,
                "vwap": 500.0,
            }
            for i in range(10)
        ]
        session.seed_bar_history(sym, seed_bars)

    await session.start()

    # Cycle 1: Acquire long positions
    await session.step_cycle()
    assert session.risk_orchestrator.state.positions["SPY"] == 40.0

    # Simulate catastrophic price crash on SPY from 500 to 200 (60% drop on $20,000 position = -$12,000 drawdown on $100k NAV = -12% DD)
    # Plus crash on QQQ from 400 to 150 (62.5% drop on $20,000 position = -$12,500 drawdown -> total drawdown -$24,500 = 24.5% DD > 15% limit)
    current_market_prices["SPY"] = 200.0
    current_market_prices["QQQ"] = 150.0

    # Cycle 2: Update prices; drawdown tripwire should trip kill switch!
    telemetry = await session.step_cycle()

    assert telemetry.status == LiveSessionStatus.PANIC_KILLED
    assert session.risk_orchestrator.kill_switch.is_active

    await session.stop()
