"""Unified Live Execution Safety, Real-Time Risk Monitor, and Emergency Orchestration Gateway.

Purpose:
    Establishes the unified RiskOrchestrator façade under Phase 6 Step 3 (Live Execution
    Quality & Safety Gateway). Integrates the deterministic PreTradeRiskFirewall,
    exchange HeartbeatWatchdog, EmergencyKillSwitch mass cancellation engine, and
    downstream execution gateways / SmartOrderRouter into a single thread-safe,
    ultra-low latency coordination layer.

Dependencies:
    - asyncio: Asynchronous coroutines, background task tracking, and non-blocking panic sweeps.
    - math: Strict non-finite scalar validation (math.isfinite).
    - time: Nanosecond-resolution timestamps (time.time_ns).
    - typing: Strict static typing annotations, Final constants, and protocol interfaces.
    - quant.execution.gateway: ExecutionGateway protocol for broker connectivity.
    - quant.execution.heartbeat: HeartbeatWatchdog and connection liveness tracking.
    - quant.execution.kill_switch: EmergencyKillSwitch, KillSwitchEvent, PanicTriggerReason, PanicMode.
    - quant.execution.models: Order, ExecutionReport, OrderSide, OrderState, OrderType.
    - quant.execution.risk: PreTradeRiskFirewall, PortfolioRiskState, RiskLimits, RiskError,
      KillSwitchActiveException, DrawdownLimitExceededException, NonFiniteRiskInputException,
      and diagnostic error code constants (ERR-RSK-001 through ERR-RSK-008).
    - quant.execution.sor: SmartOrderRouter for dark probing and lit waterfilling.
    - quant.execution.venues: ConsolidatedQuote for market depth routing.

Structural Relationship:
    - Sits as the single central entry point between high-level strategy schedulers
      (ExecutionScheduler, ParentOrder, algorithmic engines) and exchange transports.
    - Ingests: Outbound Orders, tick market price updates, connection status transitions.
    - Dispatches to: SmartOrderRouter or individual ExecutionGateway venues.
    - Couples tripwires:
        1. Gateway disconnect from HeartbeatWatchdog -> EmergencyKillSwitch.trigger_panic(GATEWAY_DISCONNECT).
        2. Intraday drawdown breach from update_market_price -> EmergencyKillSwitch.trigger_panic(DRAWDOWN_BREACH).
    - Enforces:
        1. Atomic leaves reservation before dispatch, with instant rollback on transport failure.
        2. Strict submission lockout whenever the emergency kill switch is active.
        3. Sub-20us hot-path orchestration overhead SLA.

Invariants Enforced:
    - INV-RSK-001 through INV-RSK-007: Pre-trade risk firewall boundary enforcement.
    - INV-RSK-008: Atomic kill switch mass cancellation and order lockout.
    - Atomic Leaves Accounting: Pending leaves incremented prior to dispatch and rolled back
      upon gateway failure or rejection, preventing phantom open interest drift.
    - Sub-20us Orchestration SLA: Combined hot-path pre-trade clearance and leaves reservation
      completes in < 20 microseconds.
    - Strict Input Sanitization: Non-finite values, booleans, and empty identifiers rejected.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Final

from quant.execution.gateway import ExecutionGateway
from quant.execution.heartbeat import HeartbeatWatchdog
from quant.execution.kill_switch import (
    EmergencyKillSwitch,
    KillSwitchEvent,
    PanicMode,
    PanicTriggerReason,
)
from quant.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderState,
)
from quant.execution.risk import (
    ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED,
    ERR_RSK_KILL_SWITCH_ACTIVE,
    ERR_RSK_NON_FINITE_INPUT,
    NonFiniteRiskInputException,
    PortfolioRiskState,
    PreTradeRiskFirewall,
)
from quant.execution.sor import SmartOrderRouter
from quant.execution.venues import ConsolidatedQuote

# Diagnostic fault code aliases
ERR_ORCHESTRATOR_DISCONNECTED: Final[str] = "ERR-HB-001"
ERR_ORCHESTRATOR_KILL_ACTIVE: Final[str] = ERR_RSK_KILL_SWITCH_ACTIVE
ERR_ORCHESTRATOR_DRAWDOWN: Final[str] = ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED
ERR_ORCHESTRATOR_NON_FINITE: Final[str] = ERR_RSK_NON_FINITE_INPUT


class RiskOrchestrator:
    """Unified live execution safety coordinator, real-time risk monitor, and emergency gateway façade.

    Slots:
        _firewall: Deterministic in-memory pre-trade risk firewall.
        _kill_switch: Emergency panic kill switch and mass cancellation engine.
        _state: Real-time portfolio valuation, inventory, leaves, and equity tracker.
        _router: Optional SmartOrderRouter for multi-venue algorithmic slicing.
        _admin_token: Cryptographic administrative secret for safety resets.
        _gateways: Registry of active execution gateways mapped by unique identifier.
        _watchdogs: Registry of exchange heartbeat watchdogs mapped by gateway identifier.
        _disconnect_callbacks: Registry of registered watchdog disconnect observer callbacks.
        _default_gateway_id: Identifier of the primary / default execution gateway.
        _background_tasks: Strong reference container preventing garbage collection of panic tasks.
    """

    __slots__ = (
        "_admin_token",
        "_background_tasks",
        "_default_gateway_id",
        "_disconnect_callbacks",
        "_firewall",
        "_gateways",
        "_kill_switch",
        "_router",
        "_state",
        "_watchdogs",
    )

    def __init__(
        self,
        firewall: PreTradeRiskFirewall,
        kill_switch: EmergencyKillSwitch,
        state: PortfolioRiskState,
        router: SmartOrderRouter | None = None,
        admin_token: str = "DEFAULT_ADMIN_TOKEN",
    ) -> None:
        """Initialize the unified RiskOrchestrator with required safety, risk, and state subsystems.

        Args:
            firewall: Injected PreTradeRiskFirewall instance.
            kill_switch: Injected EmergencyKillSwitch instance.
            state: Injected PortfolioRiskState instance.
            router: Optional SmartOrderRouter for multi-venue slicing.
            admin_token: Administrator authorization credential string.

        Raises:
            NonFiniteRiskInputException: If any subsystem instance or admin_token violates typing/boundaries.
        """
        # Functional Purpose: Construct unified orchestration façade and validate core subsystem dependencies.
        # Explicit Dependency Tracking: PreTradeRiskFirewall, EmergencyKillSwitch, PortfolioRiskState, SmartOrderRouter.
        # Structural Relationship: Top-level coordination entry point for live execution safety.
        # Defensive Invariant: Subsystems must be non-None valid instances; admin_token must be non-empty str.
        if not isinstance(firewall, PreTradeRiskFirewall):
            raise NonFiniteRiskInputException(
                f"firewall must be PreTradeRiskFirewall instance, got {type(firewall).__name__}",
                code=ERR_ORCHESTRATOR_NON_FINITE,
            )
        if not isinstance(kill_switch, EmergencyKillSwitch):
            raise NonFiniteRiskInputException(
                f"kill_switch must be EmergencyKillSwitch instance, got {type(kill_switch).__name__}",
                code=ERR_ORCHESTRATOR_NON_FINITE,
            )
        if not isinstance(state, PortfolioRiskState):
            raise NonFiniteRiskInputException(
                f"state must be PortfolioRiskState instance, got {type(state).__name__}",
                code=ERR_ORCHESTRATOR_NON_FINITE,
            )
        if router is not None and not isinstance(router, SmartOrderRouter):
            raise NonFiniteRiskInputException(
                f"router must be SmartOrderRouter or None, got {type(router).__name__}",
                code=ERR_ORCHESTRATOR_NON_FINITE,
            )
        if (
            isinstance(admin_token, bool)
            or not isinstance(admin_token, str)
            or not admin_token.strip()
        ):
            raise NonFiniteRiskInputException(
                f"admin_token must be non-empty string, got {admin_token!r}",
                code=ERR_ORCHESTRATOR_NON_FINITE,
            )

        self._firewall: PreTradeRiskFirewall = firewall
        self._kill_switch: EmergencyKillSwitch = kill_switch
        self._state: PortfolioRiskState = state
        self._router: SmartOrderRouter | None = router
        self._admin_token: str = admin_token.strip()

        self._gateways: dict[str, ExecutionGateway] = {}
        self._watchdogs: dict[str, HeartbeatWatchdog] = {}
        self._disconnect_callbacks: dict[str, Callable[[str, float, int], None]] = {}
        self._default_gateway_id: str | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------------
    # Subsystem Property Accessors
    # ------------------------------------------------------------------------

    @property
    def firewall(self) -> PreTradeRiskFirewall:
        """Access the underlying deterministic PreTradeRiskFirewall."""
        # Functional Purpose: Expose underlying risk firewall for parameter inspection.
        # Explicit Dependency Tracking: self._firewall.
        # Structural Relationship: Queried by risk audit monitors.
        # Defensive Invariant: Returns immutable or validated PreTradeRiskFirewall instance.
        return self._firewall

    @property
    def kill_switch(self) -> EmergencyKillSwitch:
        """Access the underlying EmergencyKillSwitch engine."""
        # Functional Purpose: Expose underlying kill switch for telemetry inspection.
        # Explicit Dependency Tracking: self._kill_switch.
        # Structural Relationship: Queried by health monitors and admin consoles.
        # Defensive Invariant: Returns EmergencyKillSwitch instance.
        return self._kill_switch

    @property
    def state(self) -> PortfolioRiskState:
        """Access the real-time PortfolioRiskState valuation and inventory tracker."""
        # Functional Purpose: Expose portfolio risk state for telemetry and dashboard reporting.
        # Explicit Dependency Tracking: self._state.
        # Structural Relationship: Queried by risk monitors and position reconciliation workers.
        # Defensive Invariant: Returns active PortfolioRiskState instance.
        return self._state

    @property
    def router(self) -> SmartOrderRouter | None:
        """Access the optional SmartOrderRouter instance, or None if direct gateway routing."""
        # Functional Purpose: Expose smart order router instance if configured.
        # Explicit Dependency Tracking: self._router.
        # Structural Relationship: Queried by algorithmic parent order coordinators.
        # Defensive Invariant: Returns SmartOrderRouter or None.
        return self._router

    @property
    def default_gateway_id(self) -> str | None:
        """Identifier of the primary / default execution gateway."""
        # Functional Purpose: Expose default gateway ID used when target_gateway_id is omitted.
        # Explicit Dependency Tracking: self._default_gateway_id.
        # Structural Relationship: Read by routing dispatch logic.
        # Defensive Invariant: Returns str identifier or None.
        return self._default_gateway_id

    @property
    def gateways(self) -> dict[str, ExecutionGateway]:
        """Shallow copy of registered execution gateways mapped by identifier."""
        # Functional Purpose: Provide safe read-only snapshot of active gateway venue registrations.
        # Explicit Dependency Tracking: self._gateways.
        # Structural Relationship: Read by venue diagnostics and health monitors.
        # Defensive Invariant: Returns dictionary shallow copy protecting internal registry.
        return dict(self._gateways)

    @property
    def watchdogs(self) -> dict[str, HeartbeatWatchdog]:
        """Shallow copy of registered connection watchdogs mapped by gateway identifier."""
        # Functional Purpose: Provide safe read-only snapshot of active transport watchdog monitors.
        # Explicit Dependency Tracking: self._watchdogs.
        # Structural Relationship: Read by connection health monitors.
        # Defensive Invariant: Returns dictionary shallow copy protecting internal registry.
        return dict(self._watchdogs)

    @property
    def is_kill_switch_active(self) -> bool:
        """Whether the emergency kill switch is currently in the PANIC_TRIGGERED state."""
        # Functional Purpose: Fast hot-path boolean predicate for active kill switch lockout.
        # Explicit Dependency Tracking: self._kill_switch.is_active.
        # Structural Relationship: Queried before order dispatch.
        # Defensive Invariant: True iff underlying kill switch is in PANIC_TRIGGERED state.
        return self._kill_switch.is_active

    @property
    def is_kill_switch_armed(self) -> bool:
        """Whether the emergency kill switch is currently in ARMED_STANDBY mode."""
        # Functional Purpose: Fast boolean predicate for operational readiness inspection.
        # Explicit Dependency Tracking: self._kill_switch.is_armed.
        # Structural Relationship: Queried during startup health checks.
        # Defensive Invariant: True iff underlying kill switch is in ARMED_STANDBY state.
        return self._kill_switch.is_armed

    @property
    def is_kill_switch_disarmed(self) -> bool:
        """Whether the emergency kill switch is currently in DISARMED maintenance mode."""
        # Functional Purpose: Fast boolean predicate for maintenance window inspection.
        # Explicit Dependency Tracking: self._kill_switch.is_disarmed.
        # Structural Relationship: Queried during scheduled maintenance.
        # Defensive Invariant: True iff underlying kill switch is in DISARMED state.
        return self._kill_switch.is_disarmed

    # ------------------------------------------------------------------------
    # Gateway & Watchdog Management
    # ------------------------------------------------------------------------

    def register_gateway(
        self,
        gateway: ExecutionGateway,
        watchdog: HeartbeatWatchdog | None = None,
        gateway_id: str | None = None,
        is_default: bool = False,
    ) -> str:
        """Register an execution gateway and optional connection watchdog with automatic tripwire coupling.

        Args:
            gateway: Object conforming to ExecutionGateway protocol.
            watchdog: Optional HeartbeatWatchdog monitoring transport liveness for this gateway.
            gateway_id: Optional explicit unique string identifier. If omitted, derived by kill_switch.
            is_default: Whether to designate this gateway as the default routing target.

        Returns:
            Resolved unique string identifier under which the gateway was registered.

        Raises:
            TypeError: If gateway does not implement ExecutionGateway or watchdog is invalid.
            ValueError: If gateway_id is invalid.
        """
        # Functional Purpose: Register venue gateway, configure kill switch sweeps, and couple watchdog tripwire.
        # Explicit Dependency Tracking: ExecutionGateway, HeartbeatWatchdog, EmergencyKillSwitch.register_gateway.
        # Structural Relationship: Called during execution engine startup or dynamic venue discovery.
        # Defensive Invariant: Attaches disconnect observer automatically firing PanicTriggerReason.GATEWAY_DISCONNECT.
        if not isinstance(gateway, ExecutionGateway):
            raise TypeError(
                f"Expected ExecutionGateway protocol implementation, got {type(gateway).__name__}"
            )
        if watchdog is not None and not isinstance(watchdog, HeartbeatWatchdog):
            raise TypeError(f"Expected HeartbeatWatchdog instance, got {type(watchdog).__name__}")

        # Register gateway with the emergency kill switch for mass order cancellation sweeps
        resolved_id = self._kill_switch.register_gateway(gateway, gateway_id=gateway_id)
        self._gateways[resolved_id] = gateway

        # If designated or first registered, set as default gateway
        if is_default or self._default_gateway_id is None:
            self._default_gateway_id = resolved_id

        # Couple connection watchdog disconnect tripwire if watchdog is provided
        if watchdog is not None:
            self._watchdogs[resolved_id] = watchdog

            # Build edge-triggered disconnect listener callback
            def _on_disconnect(gw_id: str, elapsed_sec: float, ts_ns: int) -> None:
                # Functional Purpose: Bridge synchronous watchdog timeout notification to async panic trigger.
                # Explicit Dependency Tracking: self._kill_switch.trigger_panic, PanicTriggerReason.GATEWAY_DISCONNECT.
                # Structural Relationship: Invoked on transport silence timeout.
                # Defensive Invariant: Schedules coroutine safely inside event loop or executes via asyncio.run.
                async def _trigger_disconnect_panic() -> None:
                    await self._kill_switch.trigger_panic(
                        reason=PanicTriggerReason.GATEWAY_DISCONNECT,
                        source=gw_id,
                        details=f"Watchdog silence timeout ({elapsed_sec:.3f}s elapsed)",
                        current_timestamp_ns=ts_ns,
                    )

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(_trigger_disconnect_panic())
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    # If called outside active event loop, execute synchronously
                    asyncio.run(_trigger_disconnect_panic())

            self._disconnect_callbacks[resolved_id] = _on_disconnect
            watchdog.register_disconnect_listener(_on_disconnect)

        return resolved_id

    def unregister_gateway(self, gateway_id: str) -> None:
        """Unregister an execution gateway and remove its associated watchdog disconnect tripwire.

        Args:
            gateway_id: Unique string identifier of the gateway to unregister.

        Raises:
            ValueError: If gateway_id is empty or invalid.
            KeyError: If gateway_id is not found in the registry.
        """
        # Functional Purpose: Safely decommission gateway venue and clean up watchdog tripwire listeners.
        # Explicit Dependency Tracking: EmergencyKillSwitch.unregister_gateway, HeartbeatWatchdog.unregister_disconnect_listener.
        # Structural Relationship: Called upon venue shutdown or broker session termination.
        # Defensive Invariant: Unregisters listener to prevent memory leaks or dangling panic triggers.
        if (
            isinstance(gateway_id, bool)
            or not isinstance(gateway_id, str)
            or not gateway_id.strip()
        ):
            raise ValueError(f"gateway_id must be a non-empty string, got {gateway_id!r}")
        gid = gateway_id.strip()

        if gid not in self._gateways:
            raise KeyError(f"Gateway '{gid}' is not registered in RiskOrchestrator.")

        # Clean up watchdog tripwire if attached
        if gid in self._watchdogs:
            wd = self._watchdogs.pop(gid)
            if gid in self._disconnect_callbacks:
                cb = self._disconnect_callbacks.pop(gid)
                wd.unregister_disconnect_listener(cb)

        # Remove from local registry and kill switch
        del self._gateways[gid]
        self._kill_switch.unregister_gateway(gid)

        # Reassign default gateway if unregistered was default
        if self._default_gateway_id == gid:
            self._default_gateway_id = next(iter(self._gateways.keys())) if self._gateways else None

    def get_gateway(self, gateway_id: str) -> ExecutionGateway | None:
        """Retrieve a registered ExecutionGateway by identifier, or None if not found."""
        # Functional Purpose: Lookup execution venue by string identifier.
        # Explicit Dependency Tracking: self._gateways.
        # Structural Relationship: Queried by order routing dispatch.
        # Defensive Invariant: Returns ExecutionGateway or None; no mutation.
        return self._gateways.get(gateway_id)

    def get_watchdog(self, gateway_id: str) -> HeartbeatWatchdog | None:
        """Retrieve a registered HeartbeatWatchdog by gateway identifier, or None if not found."""
        # Functional Purpose: Lookup connection watchdog by gateway string identifier.
        # Explicit Dependency Tracking: self._watchdogs.
        # Structural Relationship: Queried by connection health monitors.
        # Defensive Invariant: Returns HeartbeatWatchdog or None; no mutation.
        return self._watchdogs.get(gateway_id)

    def set_default_gateway(self, gateway_id: str) -> None:
        """Designate a registered gateway as the default routing target.

        Args:
            gateway_id: Identifier of an already registered execution gateway.

        Raises:
            KeyError: If gateway_id is not registered.
        """
        # Functional Purpose: Set default execution venue for outbound orders when target is unspecified.
        # Explicit Dependency Tracking: self._gateways, self._default_gateway_id.
        # Structural Relationship: Configured during session initialization.
        # Defensive Invariant: gateway_id must exist in self._gateways.
        if gateway_id not in self._gateways:
            raise KeyError(f"Gateway '{gateway_id}' is not registered.")
        self._default_gateway_id = gateway_id

    # ------------------------------------------------------------------------
    # Market Data Feed & Automated Drawdown Tripwire Coupling
    # ------------------------------------------------------------------------

    async def update_market_price(self, symbol: str, price: float) -> None:
        """Update mark-to-market valuation price and automatically trip emergency kill switch on drawdown breach.

        Args:
            symbol: Market ticker or instrument symbol (non-empty str).
            price: Prevailing tick or bar market valuation price (finite float > 0.0).

        Raises:
            NonFiniteRiskInputException: If symbol or price violates boundary constraints.
        """
        # Functional Purpose: Ingest streaming price ticks, update portfolio equity, and trip drawdown circuit breaker.
        # Explicit Dependency Tracking: PortfolioRiskState.update_price, EmergencyKillSwitch.trigger_panic.
        # Structural Relationship: Primary entry point for real-time market data ticks.
        # Defensive Invariant: Automatically fires PanicTriggerReason.DRAWDOWN_BREACH if drawdown >= max_intraday_drawdown_pct.
        # Update price in portfolio risk state (validates symbol, price finiteness, and positivity)
        self._state.update_price(symbol, price)

        # Evaluate real-time intraday drawdown against configured tripwire threshold
        current_drawdown = self._state.intraday_drawdown
        max_drawdown = self._firewall.limits.max_intraday_drawdown_pct

        if current_drawdown >= max_drawdown and not self._kill_switch.is_active:
            # Automated tripwire coupling: Drawdown breach -> EmergencyKillSwitch panic sweep
            await self._kill_switch.trigger_panic(
                reason=PanicTriggerReason.DRAWDOWN_BREACH,
                source=f"MarketData:{symbol}",
                details=(
                    f"Intraday drawdown {current_drawdown:.4%} breached tripwire "
                    f"threshold {max_drawdown:.4%} on tick price {price:.4f}"
                ),
            )

    async def handle_gateway_disconnect(
        self,
        gateway_id: str,
        elapsed_seconds: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> KillSwitchEvent:
        """Explicitly handle gateway transport disconnection event by triggering emergency panic.

        Args:
            gateway_id: Identifier of the disconnected gateway transport.
            elapsed_seconds: Duration in seconds of heartbeat silence.
            timestamp_ns: Nanosecond timestamp of disconnection event.

        Returns:
            KillSwitchEvent recording the emergency panic execution.
        """
        # Functional Purpose: Direct asynchronous entry point to trip emergency kill switch on gateway disconnect.
        # Explicit Dependency Tracking: EmergencyKillSwitch.trigger_panic, PanicTriggerReason.GATEWAY_DISCONNECT.
        # Structural Relationship: Invoked by connection monitor or test harness.
        # Defensive Invariant: Sub-50ms emergency cancellation broadcast across all venues.
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        return await self._kill_switch.trigger_panic(
            reason=PanicTriggerReason.GATEWAY_DISCONNECT,
            source=gateway_id,
            details=f"Transport silence timeout ({elapsed_seconds:.3f}s elapsed)",
            current_timestamp_ns=ts,
        )

    # ------------------------------------------------------------------------
    # Order Routing, Hot-Path Clearance & Atomic Leaves Accounting
    # ------------------------------------------------------------------------

    async def route_order(
        self,
        order: Order,
        target_gateway_id: str | None = None,
        current_price: float | None = None,
        quote: ConsolidatedQuote | None = None,
    ) -> ExecutionReport | list[ExecutionReport]:
        """Validate pre-trade risk, reserve open leaves atomically, and dispatch order to gateway or SOR.

        Execution Lifecycle:
            1. Kill Switch Submission Validation (< 10us): Rejects immediately if PANIC_TRIGGERED.
            2. Pre-Trade Risk Firewall Validation (< 10us): Enforces INV-RSK-001 through INV-RSK-007.
            3. Atomic Leaves Reservation: Increments signed pending leaves for symbol before dispatch.
            4. Dispatch: Routes to SmartOrderRouter.route_slice (if quote provided and target is None)
               or directly to target ExecutionGateway.submit_order.
            5. Reconciliation & Rollback:
               - On fill execution reports, updates state.update_fill (which reconciles filled leaves).
               - On unfulfilled remainders, gateway rejection, or transport exceptions, atomically
                 rolls back unfulfilled leaves, preventing phantom exposure corruption.

        Args:
            order: Outbound candidate domain Order entity.
            target_gateway_id: Optional specific gateway identifier. If omitted, uses router or default gateway.
            current_price: Optional prevailing reference price for MARKET orders.
            quote: Optional ConsolidatedQuote for multi-venue dark/lit algorithmic routing via SOR.

        Returns:
            Single ExecutionReport (from direct gateway) or list of ExecutionReports (from router).

        Raises:
            KillSwitchActiveException: If emergency kill switch is currently active (ERR-RSK-008).
            RiskError: Subclasses for fat-finger, leverage, concentration, margin, or drawdown violations.
            RuntimeError: If no execution gateway is registered or available.
        """
        # Functional Purpose: Enforce live execution safety hot-path SLA and atomic leaves conservation.
        # Explicit Dependency Tracking: validate_submission, validate_order, submit_order, route_slice.
        # Structural Relationship: Core dispatch gateway sitting between upstream strategy and downstream venues.
        # Defensive Invariant: Total orchestration overhead < 20us; atomic leaves rollback on any failure.

        # --------------------------------------------------------------------
        # Step 1: Kill Switch Hot-Path Submission Validation (< 10us)
        # --------------------------------------------------------------------
        self._kill_switch.validate_submission()

        # --------------------------------------------------------------------
        # Step 2: Pre-Trade Risk Firewall Validation (< 10us)
        # --------------------------------------------------------------------
        self._firewall.validate_order(
            order=order,
            state=self._state,
            current_price=current_price,
        )

        # --------------------------------------------------------------------
        # Step 3: Atomic Leaves Reservation Prior to Network Dispatch
        # --------------------------------------------------------------------
        # For BUY orders, pending leaves are positive (+qty); for SELL orders, negative (-qty)
        signed_leaves_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        sym = order.symbol
        self._state.pending_leaves[sym] = (
            self._state.pending_leaves.get(sym, 0.0) + signed_leaves_delta
        )

        # --------------------------------------------------------------------
        # Step 4: Dispatch to SOR (if quote provided and target is None) or Gateway
        # --------------------------------------------------------------------
        try:
            # Case A: Route via SmartOrderRouter if router is present, quote is provided, and target is None
            if self._router is not None and quote is not None and target_gateway_id is None:
                # Determine gateway for router execution
                selected_gw = self._resolve_gateway(target_gateway_id)
                reports = await self._router.route_slice(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    quote=quote,
                    gateway=selected_gw,
                    now_ns=time.time_ns(),
                )

                # Process fill reports and reconcile leaves
                total_filled_qty = 0.0
                for rep in reports:
                    if rep.exec_type in (OrderState.FILLED, OrderState.PARTIALLY_FILLED):
                        f_qty = rep.cum_quantity
                        f_price = (
                            rep.last_price
                            if (rep.last_price is not None and rep.last_price > 0.0)
                            else (
                                order.price
                                if (order.price is not None and order.price > 0.0)
                                else (current_price or self._state.current_prices.get(sym, 1.0))
                            )
                        )
                        # state.update_fill reconciles cash, positions, and decrements pending_leaves
                        self._state.update_fill(
                            symbol=sym,
                            filled_qty=f_qty,
                            price=f_price,
                            side=order.side,
                        )
                        total_filled_qty += f_qty

                # If unfilled remainder was discarded/cancelled by SOR, roll back unfulfilled leaves
                unfilled_qty = max(0.0, order.quantity - total_filled_qty)
                if unfilled_qty > 1e-7:
                    unfilled_signed = unfilled_qty if order.side == OrderSide.BUY else -unfilled_qty
                    self._state.pending_leaves[sym] = (
                        self._state.pending_leaves.get(sym, 0.0) - unfilled_signed
                    )
                    if abs(self._state.pending_leaves[sym]) <= 1e-12:
                        del self._state.pending_leaves[sym]

                return reports

            # Case B: Direct Gateway Submission
            target_gw = self._resolve_gateway(target_gateway_id)
            report = await target_gw.submit_order(order)

            # Reconcile fills and leaves based on execution report
            if report.exec_type == OrderState.REJECTED:
                # Gateway rejected order: complete atomic leaves rollback
                self._state.pending_leaves[sym] = (
                    self._state.pending_leaves.get(sym, 0.0) - signed_leaves_delta
                )
                if abs(self._state.pending_leaves[sym]) <= 1e-12:
                    del self._state.pending_leaves[sym]

            elif report.exec_type in (OrderState.FILLED, OrderState.PARTIALLY_FILLED):
                # Execution fill occurred: reconcile via state.update_fill
                f_qty = (
                    report.last_quantity
                    if (report.last_quantity is not None and report.last_quantity > 0.0)
                    else report.cum_quantity
                )
                f_price = (
                    report.last_price
                    if (report.last_price is not None and report.last_price > 0.0)
                    else (
                        order.price
                        if (order.price is not None and order.price > 0.0)
                        else (current_price or self._state.current_prices.get(sym, 1.0))
                    )
                )
                self._state.update_fill(
                    symbol=sym,
                    filled_qty=f_qty,
                    price=f_price,
                    side=order.side,
                )

            # (For OrderState.NEW, the order is resting on the book; leaves remain reserved)
            return report

        except Exception:
            # ----------------------------------------------------------------
            # Step 5: Defensive Leaves Rollback on Gateway / Router Exception
            # ----------------------------------------------------------------
            # If network error, socket timeout, or unhandled exception occurs, roll back reserved leaves
            self._state.pending_leaves[sym] = (
                self._state.pending_leaves.get(sym, 0.0) - signed_leaves_delta
            )
            if abs(self._state.pending_leaves[sym]) <= 1e-12:
                del self._state.pending_leaves[sym]
            raise

    def _resolve_gateway(self, target_gateway_id: str | None) -> ExecutionGateway:
        """Resolve the target ExecutionGateway instance or raise RuntimeError if none available."""
        # Functional Purpose: Deterministically resolve execution venue for outbound order dispatch.
        # Explicit Dependency Tracking: self._gateways, self._default_gateway_id.
        # Structural Relationship: Internal helper for route_order.
        # Defensive Invariant: Raises RuntimeError if designated gateway is missing or no gateways exist.
        if target_gateway_id is not None:
            gw = self._gateways.get(target_gateway_id)
            if gw is None:
                raise RuntimeError(
                    f"Target execution gateway '{target_gateway_id}' is not registered."
                )
            return gw

        if self._default_gateway_id is not None and self._default_gateway_id in self._gateways:
            return self._gateways[self._default_gateway_id]

        if self._gateways:
            return next(iter(self._gateways.values()))

        raise RuntimeError("No execution gateways registered in RiskOrchestrator.")

    # ------------------------------------------------------------------------
    # Emergency & Administrative Controls
    # ------------------------------------------------------------------------

    async def trigger_emergency_panic(
        self,
        reason: PanicTriggerReason | str,
        source: str,
        panic_mode: PanicMode | str = PanicMode.CANCEL_ONLY,
        details: str = "",
        current_timestamp_ns: int | None = None,
    ) -> KillSwitchEvent:
        """Trigger emergency panic: cancel all open orders, freeze schedulers, and lock submissions.

        Args:
            reason: Etiology classification (PanicTriggerReason or valid string).
            source: Subsystem or operator identifier triggering the panic.
            panic_mode: Response posture (CANCEL_ONLY or CANCEL_AND_FLATTEN).
            details: Contextual audit commentary.
            current_timestamp_ns: Explicit nanosecond timestamp (defaults to time.time_ns()).

        Returns:
            Immutable KillSwitchEvent documenting the panic sweep execution.
        """
        # Functional Purpose: Expose manual or automated emergency panic trigger through orchestrator façade.
        # Explicit Dependency Tracking: EmergencyKillSwitch.trigger_panic.
        # Structural Relationship: Invoked by risk officers, trading desks, or automated supervisors.
        # Defensive Invariant: Transitions state to PANIC_TRIGGERED; broadcasts multi-venue cancellations.
        return await self._kill_switch.trigger_panic(
            reason=reason,
            source=source,
            panic_mode=panic_mode,
            details=details,
            current_timestamp_ns=current_timestamp_ns,
        )

    def reset_kill_switch(self, admin_token: str) -> None:
        """Reset emergency kill switch from PANIC_TRIGGERED back to ARMED_STANDBY.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Administratively re-arm execution safety after post-mortem review.
        # Explicit Dependency Tracking: EmergencyKillSwitch.reset.
        # Structural Relationship: Invoked by Chief Risk Officer / Operator console.
        # Defensive Invariant: Constant-time authentication check; transitions state to ARMED_STANDBY.
        self._kill_switch.reset(admin_token)

    def disarm_kill_switch(self, admin_token: str) -> None:
        """Disarm emergency kill switch into DISARMED maintenance state.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Suspend safety tripwires for scheduled maintenance or testing.
        # Explicit Dependency Tracking: EmergencyKillSwitch.disarm.
        # Structural Relationship: Invoked during system maintenance windows.
        # Defensive Invariant: Constant-time authentication check; transitions state to DISARMED.
        self._kill_switch.disarm(admin_token)

    def arm_kill_switch(self, admin_token: str) -> None:
        """Arm emergency kill switch into ARMED_STANDBY monitoring state.

        Args:
            admin_token: Administrator authorization token matching configured secret.

        Raises:
            InvalidAdminTokenException: If admin_token does not match configured secret.
        """
        # Functional Purpose: Re-activate kill switch protection after maintenance disarm.
        # Explicit Dependency Tracking: EmergencyKillSwitch.arm.
        # Structural Relationship: Invoked when returning to live trading.
        # Defensive Invariant: Constant-time authentication check; transitions state to ARMED_STANDBY.
        self._kill_switch.arm(admin_token)
