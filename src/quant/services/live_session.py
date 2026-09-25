"""Turnkey Continuous Live Paper Trading Session Orchestrator.

Functional Purpose:
    Provides the continuous asynchronous event loop orchestrating live market data ingestion,
    regime-conditioned alpha strategy evaluation, pre-trade risk firewall gating, smart order
    routing, execution gateway matching, double-entry tax-lot bookkeeping, and terminal/web
    telemetry streaming.

Explicit Dependency Tracking:
    - asyncio: Non-blocking periodic scheduling loop and task cancellation safety.
    - math: Strict non-finite float verification (math.isfinite).
    - time: Nanosecond and millisecond epoch benchmarking.
    - typing: Final constants, Protocols, and immutable dataclass configurations.
    - quant.analytics.strategies.base: IAlphaStrategy, BarHistoryWindow, StrategyContext, StrategySignal.
    - quant.execution.gateway: ExecutionGateway, PaperExecutionGateway.
    - quant.execution.models: ExecutionReport, Order, OrderSide, OrderState, OrderType.
    - quant.execution.risk: ERR_RSK_KILL_SWITCH_ACTIVE, KillSwitchActiveException, PortfolioRiskState, RiskLimits.
    - quant.execution.risk_orchestrator: RiskOrchestrator.
    - quant.services.ledger_service: LedgerService (optional persistent ledger).

Structural Relationship:
    - Master application coordinator for Phase 17 (Turnkey Live Trading).
    - Bridges Market Data providers, Strategy Library (Phase 14), Execution & Risk Orchestrator (Phase 6),
      and Persistent Ledger (Phase 16) into an autonomous, resilient production runtime.
    - Consumed by CLI runner (scripts/run_live_trader.py) and Terminal HUD (src/quant/execution/terminal_hud.py).

Defensive Invariants:
    - INV-LIVE-001: Monotonic state transitions (INITIALIZING -> RUNNING <-> PAUSED -> STOPPING -> STOPPED).
    - INV-LIVE-002: Strict kill-switch lockout: Zero orders dispatched when kill switch is active.
    - INV-LIVE-003: Non-finite input rejection on prices, quantities, cash, and leverage.
    - INV-LIVE-004: Atomic orderly shutdown: All working orders cancelled before session termination.
    - INV-LIVE-005: Zero uncaught exceptions in event loop: Any cycle fault is logged and captured in telemetry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import signal
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

from quant.analytics.strategies.base import (
    BarHistoryWindow,
    IAlphaStrategy,
    StrategyContext,
    StrategySignal,
)
from quant.execution.gateway import ExecutionGateway, PaperExecutionGateway
from quant.execution.kill_switch import EmergencyKillSwitch, PanicMode, PanicTriggerReason
from quant.execution.models import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
)
from quant.execution.risk import (
    DrawdownLimitExceededException,
    KillSwitchActiveException,
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskLimits,
)
from quant.execution.risk_orchestrator import RiskOrchestrator
from quant.services.ledger_service import LedgerService

logger = logging.getLogger(__name__)

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_LIVE_ALREADY_RUNNING: Final[str] = "ERR-LIVE-001"
ERR_LIVE_INVALID_STATE: Final[str] = "ERR-LIVE-002"
ERR_LIVE_FIREWALL_BREACH: Final[str] = "ERR-LIVE-003"
ERR_LIVE_GATEWAY_DISCONNECTED: Final[str] = "ERR-LIVE-004"
ERR_LIVE_NON_FINITE_INPUT: Final[str] = "ERR-LIVE-005"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class LiveSessionError(Exception):
    """Base exception for live trading session faults."""

    def __init__(self, message: str, code: str = "ERR-LIVE-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class LiveSessionStateError(LiveSessionError):
    """Raised when an invalid state transition or concurrent run is attempted."""

    def __init__(self, message: str, code: str = ERR_LIVE_INVALID_STATE) -> None:
        super().__init__(message, code=code)


class LiveSessionRiskError(LiveSessionError):
    """Raised when a critical risk or firewall invariant is breached in live trading."""

    def __init__(self, message: str, code: str = ERR_LIVE_FIREWALL_BREACH) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Enums and Data Transfer Objects
# ============================================================================


class LiveSessionStatus(StrEnum):
    """Lifecycle states of the live trading session."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    PANIC_KILLED = "PANIC_KILLED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class LiveSessionConfig:
    """Immutable configuration parameters for live paper trading session.

    Functional Purpose:
        Specifies trading universe, capital constraints, risk thresholds, and execution friction.

    Defensive Invariants:
        initial_cash > 0, max_leverage >= 1.0, poll_interval_sec > 0.
    """

    session_id: str
    symbols: tuple[str, ...]
    initial_cash: float = 100000.0
    poll_interval_sec: float = 1.0
    max_leverage: float = 2.0
    max_concentration: float = 0.35
    max_drawdown_limit: float = 0.15
    rebalance_threshold_pct: float = 0.02
    min_order_notional: float = 50.0
    slippage_bps: float = 2.0
    fee_bps: float = 1.0
    admin_token: str = "live-admin-safety-token"

    def __post_init__(self) -> None:
        if not self.symbols:
            raise LiveSessionError(
                "symbols must contain at least one ticker symbol",
                code=ERR_LIVE_NON_FINITE_INPUT,
            )
        if (
            isinstance(self.initial_cash, bool)
            or self.initial_cash <= 0.0
            or not math.isfinite(self.initial_cash)
        ):
            raise LiveSessionError(
                f"initial_cash must be finite positive float, got {self.initial_cash!r}",
                code=ERR_LIVE_NON_FINITE_INPUT,
            )
        if (
            isinstance(self.max_leverage, bool)
            or self.max_leverage < 1.0
            or not math.isfinite(self.max_leverage)
        ):
            raise LiveSessionError(
                f"max_leverage must be finite >= 1.0, got {self.max_leverage!r}",
                code=ERR_LIVE_NON_FINITE_INPUT,
            )
        if (
            isinstance(self.poll_interval_sec, bool)
            or self.poll_interval_sec <= 0.0
            or not math.isfinite(self.poll_interval_sec)
        ):
            raise LiveSessionError(
                f"poll_interval_sec must be positive finite float, got {self.poll_interval_sec!r}",
                code=ERR_LIVE_NON_FINITE_INPUT,
            )


@dataclass(frozen=True, slots=True)
class LiveSessionTelemetry:
    """Real-time telemetry snapshot emitted on every session cycle.

    Functional Purpose:
        Immutable metrics bundle consumed by Terminal HUD, WebSockets, and audit monitors.
    """

    timestamp_ns: int
    iteration: int
    status: LiveSessionStatus
    portfolio_nav: float
    cash: float
    gross_leverage: float
    net_leverage: float
    drawdown_pct: float
    current_prices: dict[str, float]
    positions: dict[str, float]
    target_weights: dict[str, float]
    active_signals: dict[str, StrategySignal]
    orders_dispatched_count: int
    fills_count: int
    is_kill_switch_active: bool
    last_error: str | None = None


# ============================================================================
# Live Trading Session Orchestrator Class
# ============================================================================


class LiveTradingSession:
    """Master asynchronous live trading session orchestrator.

    Functional Purpose:
        Coordinates data ingestion, strategy signal generation, pre-trade risk clearance,
        smart order routing, simulated/broker execution, and double-entry ledger tracking.

    Explicit Dependency Tracking:
        - RiskOrchestrator for pre-trade firewall and emergency kill switch.
        - ExecutionGateway for broker communication.
        - IAlphaStrategy for alpha weight generation.
        - LedgerService for persistent tax-lot bookkeeping.

    Structural Relationship:
        Single central coordinator instantiated by CLI runners or web controllers.

    Defensive Invariants:
        - INV-LIVE-001: Thread-safe monotonic status transitions.
        - INV-LIVE-002: Strict order blocking under active emergency kill switch.
        - INV-LIVE-003: Graceful, idempotent shutdown via signal handlers.
    """

    __slots__ = (
        "_bar_buffers",
        "_config",
        "_custom_price_feed",
        "_dispatched_orders_count",
        "_fills_count",
        "_gateway",
        "_is_loop_running",
        "_iteration",
        "_last_error",
        "_ledger_service",
        "_loop_task",
        "_on_telemetry",
        "_risk_orchestrator",
        "_status",
        "_strategy",
    )

    def __init__(
        self,
        config: LiveSessionConfig,
        strategy: IAlphaStrategy,
        gateway: ExecutionGateway | None = None,
        risk_orchestrator: RiskOrchestrator | None = None,
        ledger_service: LedgerService | None = None,
        on_telemetry: Callable[[LiveSessionTelemetry], None] | None = None,
        custom_price_feed: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        """Initialize the LiveTradingSession with configured components."""
        self._config = config
        self._strategy = strategy
        self._status = LiveSessionStatus.INITIALIZING
        self._on_telemetry = on_telemetry
        self._custom_price_feed = custom_price_feed
        self._ledger_service = ledger_service
        self._iteration = 0
        self._dispatched_orders_count = 0
        self._fills_count = 0
        self._last_error: str | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._is_loop_running = False

        # In-memory bar history buffers per symbol: map symbol -> deque of dicts
        self._bar_buffers: dict[str, deque[dict[str, float]]] = {
            sym: deque(maxlen=max(250, self._strategy.min_warmup_bars + 50))
            for sym in self._config.symbols
        }

        # Initialize PaperExecutionGateway if none provided
        gw: ExecutionGateway
        if gateway is None:
            gw = PaperExecutionGateway(
                slippage_bps=config.slippage_bps,
                fee_bps=config.fee_bps,
            )
        else:
            gw = gateway
        self._gateway: ExecutionGateway = gw

        # Initialize RiskOrchestrator if none provided
        if risk_orchestrator is None:
            limits = RiskLimits(
                max_order_notional=100000.0,
                max_order_qty=10000.0,
                max_gross_leverage=config.max_leverage,
                max_net_leverage=config.max_leverage,
                max_concentration_nav_pct=config.max_concentration,
                max_intraday_drawdown_pct=config.max_drawdown_limit,
                min_free_margin=1000.0,
            )
            state = PortfolioRiskState(
                cash=config.initial_cash,
                current_prices={},
                positions=dict.fromkeys(config.symbols, 0.0),
                pending_leaves=dict.fromkeys(config.symbols, 0.0),
                peak_equity=config.initial_cash,
                initial_equity=config.initial_cash,
            )
            firewall = PreTradeRiskFirewall(limits=limits)
            kill_switch = EmergencyKillSwitch(admin_token=config.admin_token)
            self._risk_orchestrator = RiskOrchestrator(
                firewall=firewall,
                kill_switch=kill_switch,
                state=state,
                admin_token=config.admin_token,
            )
            self._risk_orchestrator.register_gateway(
                gateway=self._gateway,
                gateway_id="default",
                is_default=True,
            )
        else:
            self._risk_orchestrator = risk_orchestrator

    # ------------------------------------------------------------------------
    # Properties & Status Accessors
    # ------------------------------------------------------------------------

    @property
    def status(self) -> LiveSessionStatus:
        """Current operational status of the live trading session."""
        return self._status

    @property
    def config(self) -> LiveSessionConfig:
        """Immutable session configuration."""
        return self._config

    @property
    def strategy(self) -> IAlphaStrategy:
        """Configured alpha strategy instance."""
        return self._strategy

    @property
    def risk_orchestrator(self) -> RiskOrchestrator:
        """Central risk orchestrator façade."""
        return self._risk_orchestrator

    @property
    def gateway(self) -> ExecutionGateway:
        """Underlying broker execution gateway."""
        return self._gateway

    @property
    def iteration(self) -> int:
        """Current cycle iteration count."""
        return self._iteration

    # ------------------------------------------------------------------------
    # Warmup and Seed Data
    # ------------------------------------------------------------------------

    def seed_bar_history(
        self,
        symbol: str,
        bars: list[dict[str, float]],
    ) -> None:
        """Pre-populate the rolling bar buffer for a symbol to satisfy strategy warmup.

        Functional Purpose:
            Ensures alpha strategies can immediately evaluate signals on the first cycle.
        """
        if symbol not in self._bar_buffers:
            self._bar_buffers[symbol] = deque(maxlen=max(250, self._strategy.min_warmup_bars + 50))
        for b in bars:
            self._bar_buffers[symbol].append(b)

    # ------------------------------------------------------------------------
    # Lifecycle Control Methods
    # ------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the continuous asynchronous live trading loop.

        Functional Purpose:
            Connects broker gateway, transitions status to RUNNING, and launches background task.

        Defensive Invariants:
            INV-LIVE-001: Cannot start if already running.
        """
        if self._status == LiveSessionStatus.RUNNING or self._is_loop_running:
            raise LiveSessionStateError(
                f"Session '{self._config.session_id}' is already running",
                code=ERR_LIVE_ALREADY_RUNNING,
            )

        logger.info("Connecting execution gateway for session '%s'...", self._config.session_id)
        if not self._gateway.is_connected:
            await self._gateway.connect()

        self._status = LiveSessionStatus.RUNNING
        self._is_loop_running = True
        self._loop_task = asyncio.create_task(self._run_event_loop())
        logger.info("Live trading session '%s' started successfully", self._config.session_id)

    async def pause(self) -> None:
        """Pause trading signal generation without disconnecting gateways or canceling orders.

        Functional Purpose:
            Allows operators to temporarily freeze rebalancing while maintaining live risk tracking.
        """
        if self._status != LiveSessionStatus.RUNNING:
            return
        self._status = LiveSessionStatus.PAUSED
        logger.info("Live trading session '%s' paused", self._config.session_id)

    async def resume(self) -> None:
        """Resume trading signal generation from paused state."""
        if self._risk_orchestrator.kill_switch.is_active:
            raise LiveSessionRiskError(
                "Cannot resume session while Emergency Kill Switch is active",
                code=ERR_LIVE_FIREWALL_BREACH,
            )
        if self._status != LiveSessionStatus.PAUSED:
            return
        self._status = LiveSessionStatus.RUNNING
        logger.info("Live trading session '%s' resumed", self._config.session_id)

    async def panic_kill(self, reason: str = "OPERATOR_PANIC") -> None:
        """Trigger emergency kill switch: cancel all open orders and lock out new submissions.

        Functional Purpose:
            Immediate risk mitigation under abnormal conditions or operator override.
        """
        self._status = LiveSessionStatus.PANIC_KILLED
        logger.critical(
            "PANIC KILL triggered on live session '%s'! Reason: %s",
            self._config.session_id,
            reason,
        )
        await self._risk_orchestrator.kill_switch.trigger_panic(
            reason=PanicTriggerReason.MANUAL_OPERATOR,
            source=f"session_{self._config.session_id}",
            panic_mode=PanicMode.CANCEL_ONLY,
            details=reason,
        )

    async def disarm(self, admin_token: str) -> None:
        """Disarm the emergency kill switch and restore trading readiness.

        Functional Purpose:
            Operator clearance required after safety review.
        """
        self._risk_orchestrator.disarm_kill_switch(admin_token)
        self._status = LiveSessionStatus.PAUSED
        logger.info(
            "Live trading session '%s' kill switch disarmed. State set to PAUSED.",
            self._config.session_id,
        )

    async def stop(self) -> None:
        """Orderly shutdown: cancel working orders, disconnect gateways, and flush state.

        Functional Purpose:
            INV-LIVE-004: Clean termination guaranteeing zero orphaned working orders.
        """
        if self._status in (LiveSessionStatus.STOPPING, LiveSessionStatus.STOPPED):
            return

        self._status = LiveSessionStatus.STOPPING
        logger.info("Stopping live trading session '%s'...", self._config.session_id)

        # 1. Stop background loop
        self._is_loop_running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task

        # 2. Cancel all open orders across gateways
        try:
            logger.info("Canceling all open orders across gateways...")
            open_orders = await self._gateway.get_open_orders()
            for o in open_orders:
                with contextlib.suppress(Exception):
                    await self._gateway.cancel_order(o.cl_ord_id)
        except Exception as exc:
            logger.error("Exception during final cancel_orders: %s", exc)

        # 3. Disconnect gateway
        try:
            if self._gateway.is_connected:
                await self._gateway.disconnect()
        except Exception as exc:
            logger.error("Exception disconnecting gateway: %s", exc)

        self._status = LiveSessionStatus.STOPPED
        logger.info("Live trading session '%s' stopped cleanly", self._config.session_id)

    # ------------------------------------------------------------------------
    # Signal Trapping for Graceful Terminal Interruption
    # ------------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers in the running event loop for orderly shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _handle_signal() -> None:
            logger.warning("Interruption signal received! Initiating atomic session shutdown...")
            asyncio.create_task(self.stop())

        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(ValueError, NotImplementedError):
                    loop.add_signal_handler(sig, _handle_signal)
        else:
            # On Windows, signal handlers can be attached via signal.signal
            def _win_sig_handler(signum: int, frame: Any) -> None:
                _handle_signal()

            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, _win_sig_handler)
                signal.signal(signal.SIGTERM, _win_sig_handler)

    # ------------------------------------------------------------------------
    # Event Loop and Atomic Step Execution
    # ------------------------------------------------------------------------

    async def _run_event_loop(self) -> None:
        """Continuous execution loop invoking step_cycle at poll_interval_sec."""
        logger.info("Starting background event loop for session '%s'", self._config.session_id)
        while self._is_loop_running:
            try:
                await self.step_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Unexpected error in live session cycle: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.poll_interval_sec)

    async def step_cycle(self) -> LiveSessionTelemetry:
        """Execute a single atomic trading and risk evaluation cycle.

        Functional Purpose:
            1. Fetch / synthesize latest market prices.
            2. Update RiskOrchestrator mark-to-market valuations and drawdown tracking.
            3. Append latest bar to rolling history buffer.
            4. If RUNNING and kill switch inactive:
               - Construct BarHistoryWindow and StrategyContext.
               - Compute alpha signals.
               - Calculate delta position rebalancing.
               - Submit rebalance orders through RiskOrchestrator pre-trade firewall.
            5. Emit telemetry snapshot to listeners.

        Defensive Invariants:
            INV-LIVE-005: Zero uncaught exceptions escaping the cycle.
        """
        self._iteration += 1
        now_ns = time.time_ns()
        latest_prices: dict[str, float] = {}

        try:
            # 1. Acquire current market prices
            if self._custom_price_feed is not None:
                latest_prices = self._custom_price_feed()
            else:
                # Default fallback: inspect gateway prices or fallback to nominal 100.0
                for sym in self._config.symbols:
                    p = self._risk_orchestrator.state.current_prices.get(sym, 100.0)
                    latest_prices[sym] = p

            # 2. Update RiskOrchestrator prices & check tripwires
            for sym, price in latest_prices.items():
                if math.isfinite(price) and price > 0.0:
                    try:
                        await self._risk_orchestrator.update_market_price(sym, price)
                    except DrawdownLimitExceededException:
                        self._status = LiveSessionStatus.PANIC_KILLED
                        self._last_error = "Drawdown limit exceeded, kill switch tripped"
                        logger.critical("Drawdown limit breached during price update!")
                    if isinstance(self._gateway, PaperExecutionGateway):
                        matched_reports = self._gateway.set_market_price(sym, price)
                        for rep in matched_reports:
                            if rep.exec_type in (
                                OrderState.FILLED,
                                OrderState.PARTIALLY_FILLED,
                            ) or (rep.last_quantity is not None and rep.last_quantity > 0.0):
                                self._fills_count += 1
                                if rep.last_quantity is not None and rep.last_price is not None:
                                    port_state = self._risk_orchestrator.state
                                    signed_qty = (
                                        rep.last_quantity
                                        if rep.side == OrderSide.BUY
                                        else -rep.last_quantity
                                    )
                                    port_state.positions[sym] = (
                                        port_state.positions.get(sym, 0.0) + signed_qty
                                    )
                                    curr_leaves = port_state.pending_leaves.get(sym, 0.0)
                                    if rep.side == OrderSide.BUY:
                                        port_state.pending_leaves[sym] = max(
                                            0.0, curr_leaves - rep.last_quantity
                                        )
                                    else:
                                        port_state.pending_leaves[sym] = min(
                                            0.0, curr_leaves + rep.last_quantity
                                        )

            if (
                self._risk_orchestrator.kill_switch.is_active
                and self._status == LiveSessionStatus.RUNNING
            ):
                self._status = LiveSessionStatus.PANIC_KILLED
                self._last_error = "Kill switch tripped by risk orchestrator tripwire"

            # 3. Update rolling bar history buffer
            for sym, price in latest_prices.items():
                if math.isfinite(price) and price > 0.0:
                    self._append_bar_observation(sym, price)

            # 4. Strategy evaluation & order rebalancing (if RUNNING and NOT killed)
            active_signals: dict[str, StrategySignal] = {}
            target_weights: dict[str, float] = {}

            if (
                self._status == LiveSessionStatus.RUNNING
                and not self._risk_orchestrator.kill_switch.is_active
            ):
                history_window = self._build_history_window()
                port_state = self._risk_orchestrator.state
                equity = port_state.current_equity
                cash = port_state.cash

                if equity > 0.0:
                    context = StrategyContext(
                        current_timestamp=int(now_ns / 1_000_000_000),
                        current_prices=dict(port_state.current_prices),
                        current_positions=dict(port_state.positions),
                        total_equity=equity,
                        unencumbered_cash=cash,
                        regime_label="NORMAL",
                    )

                    # Compute strategy signals
                    try:
                        active_signals = self._strategy.compute_signals(history_window, context)
                    except Exception as exc:
                        logger.warning("Strategy compute_signals threw exception: %s", exc)

                    # Determine target weights and dispatch orders
                    for sym in self._config.symbols:
                        sig = active_signals.get(sym)
                        target_w = sig.target_weight if sig is not None else 0.0
                        target_weights[sym] = target_w

                        # Execute delta rebalancing
                        await self._rebalance_symbol(
                            symbol=sym,
                            target_weight=target_w,
                            current_price=latest_prices.get(sym, 0.0),
                            equity=equity,
                            port_state=port_state,
                        )

        except Exception as exc:
            self._last_error = str(exc)
            logger.error("Error during step_cycle: %s", exc)

        # 5. Snapshot telemetry
        state = self._risk_orchestrator.state
        nav = state.current_equity
        peak = state.peak_equity
        dd = (peak - nav) / peak if peak > 0.0 else 0.0

        telemetry = LiveSessionTelemetry(
            timestamp_ns=now_ns,
            iteration=self._iteration,
            status=self._status,
            portfolio_nav=nav,
            cash=state.cash,
            gross_leverage=state.gross_leverage,
            net_leverage=state.net_leverage,
            drawdown_pct=dd,
            current_prices=dict(state.current_prices),
            positions=dict(state.positions),
            target_weights=target_weights,
            active_signals=active_signals,
            orders_dispatched_count=self._dispatched_orders_count,
            fills_count=self._fills_count,
            is_kill_switch_active=self._risk_orchestrator.kill_switch.is_active,
            last_error=self._last_error,
        )

        if self._on_telemetry is not None:
            try:
                self._on_telemetry(telemetry)
            except Exception as exc:
                logger.warning("Telemetry callback threw exception: %s", exc)

        return telemetry

    # ------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------

    def _append_bar_observation(self, symbol: str, price: float) -> None:
        """Append or create a bar entry in the buffer for a symbol."""
        buf = self._bar_buffers[symbol]
        now_sec = time.time()
        buf.append(
            {
                "timestamp": now_sec,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
                "vwap": price,
            }
        )

    def _build_history_window(self) -> BarHistoryWindow:
        """Construct a BarHistoryWindow from buffered bars for strategy consumption."""
        data: dict[str, dict[str, np.ndarray]] = {}
        for sym, buf in self._bar_buffers.items():
            if not buf:
                # Provide at least 1 mock nominal bar to prevent empty dimension errors
                p = 100.0
                data[sym] = {
                    "timestamp": np.array([time.time()], dtype=np.float64),
                    "open": np.array([p], dtype=np.float64),
                    "high": np.array([p], dtype=np.float64),
                    "low": np.array([p], dtype=np.float64),
                    "close": np.array([p], dtype=np.float64),
                    "volume": np.array([1000.0], dtype=np.float64),
                    "vwap": np.array([p], dtype=np.float64),
                }
            else:
                data[sym] = {
                    "timestamp": np.array([b["timestamp"] for b in buf], dtype=np.float64),
                    "open": np.array([b["open"] for b in buf], dtype=np.float64),
                    "high": np.array([b["high"] for b in buf], dtype=np.float64),
                    "low": np.array([b["low"] for b in buf], dtype=np.float64),
                    "close": np.array([b["close"] for b in buf], dtype=np.float64),
                    "volume": np.array([b["volume"] for b in buf], dtype=np.float64),
                    "vwap": np.array([b["vwap"] for b in buf], dtype=np.float64),
                }
        return BarHistoryWindow(data=data, symbols=self._config.symbols)

    async def _rebalance_symbol(
        self,
        symbol: str,
        target_weight: float,
        current_price: float,
        equity: float,
        port_state: PortfolioRiskState,
    ) -> None:
        """Evaluate delta rebalancing and submit market order if threshold exceeded."""
        if current_price <= 0.0 or not math.isfinite(current_price):
            return

        current_pos = port_state.positions.get(symbol, 0.0)
        pending_leaves = port_state.pending_leaves.get(symbol, 0.0)
        effective_pos = current_pos + pending_leaves
        effective_notional = effective_pos * current_price
        effective_weight = effective_notional / equity if equity > 0.0 else 0.0

        weight_delta = target_weight - effective_weight
        if abs(weight_delta) < self._config.rebalance_threshold_pct:
            return

        target_notional = target_weight * equity
        target_quantity = target_notional / current_price
        delta_quantity = target_quantity - effective_pos

        order_qty = abs(round(delta_quantity))
        if order_qty <= 0.0:
            return

        order_notional = order_qty * current_price
        if order_notional < self._config.min_order_notional:
            return

        side = OrderSide.BUY if delta_quantity > 0 else OrderSide.SELL
        cl_ord_id = f"ord_{self._config.session_id}_{symbol}_{self._iteration}_{time.time_ns()}"
        order = Order(
            cl_ord_id=cl_ord_id,
            symbol=symbol,
            side=side,
            quantity=order_qty,
            order_type=OrderType.MARKET,
        )

        try:
            self._dispatched_orders_count += 1
            reports = await self._risk_orchestrator.route_order(order, current_price=current_price)
            report_list = reports if isinstance(reports, list) else [reports]
            for rep in report_list:
                if rep.exec_type in (OrderState.FILLED, OrderState.PARTIALLY_FILLED) or (
                    rep.last_quantity is not None and rep.last_quantity > 0.0
                ):
                    self._fills_count += 1
        except KillSwitchActiveException:
            logger.info("Order submission skipped: Emergency Kill Switch active")
        except Exception as exc:
            logger.warning("Order submission rejected by risk firewall or gateway: %s", exc)
