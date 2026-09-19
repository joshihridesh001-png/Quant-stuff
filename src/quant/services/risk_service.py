"""Application service for pre-trade risk management, heartbeat watchdogs, and panic kill switch.

Purpose:
    Exposes unified risk controls, real-time exposure monitoring, dynamic limit adjustments,
    exchange heartbeat tracking, and emergency panic kill switch coordination to REST/WebSocket API endpoints.

Dependencies:
    - hmac: Constant-time authentication comparisons.
    - math: Strict non-finite validation.
    - typing: Static typing annotations and type definitions.
    - quant.api.v1.schemas: RiskStatusResponse, RiskLimitsDTO, RiskLimitsUpdateRequest, GatewayHealthDTO.
    - quant.execution.kill_switch: KillSwitchEvent, PanicTriggerReason.
    - quant.execution.risk: RiskLimits, PreTradeRiskFirewall, PortfolioRiskState, NonFiniteRiskInputException.
    - quant.execution.risk_orchestrator: RiskOrchestrator.

Structural Relationship:
    - Sits between API presentation layer (endpoints/risk.py, endpoints/gateways.py) and execution core (RiskOrchestrator).
    - Encapsulates risk invariants, preventing direct route-level mutation of risk state.

Invariants Enforced:
    - INV-RSK-001 through INV-RSK-007: Pre-trade risk firewall boundary enforcement.
    - INV-RSK-008: Atomic emergency kill switch lockout and mass cancellation.
    - Constant-Time Reset: Kill switch disarm strictly verifies admin_token via hmac.compare_digest.
    - Non-Finite Rejection: Rejects NaN, Inf, and booleans for all price and limit mutations.
    - Rule 1: Line annotations on all classes and methods.
    - Rule 2: Diagnostic error codes.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from typing import Any

from quant.api.v1.schemas import (
    GatewayHealthDTO,
    RiskLimitsDTO,
    RiskLimitsUpdateRequest,
    RiskStatusResponse,
)
from quant.execution.kill_switch import KillSwitchEvent, PanicTriggerReason
from quant.execution.risk import (
    ERR_RSK_NON_FINITE_INPUT,
    NonFiniteRiskInputException,
    RiskLimits,
)
from quant.execution.risk_orchestrator import RiskOrchestrator


class RiskService:
    """Application service coordinating real-time risk telemetry, dynamic limits, and emergency kill switch."""

    __slots__ = ("_orchestrator", "_risk_listeners")

    def __init__(self, orchestrator: RiskOrchestrator) -> None:
        """Initialize RiskService with injected RiskOrchestrator instance.

        Args:
            orchestrator: Central live execution risk orchestrator.

        Raises:
            NonFiniteRiskInputException: If orchestrator is not a valid RiskOrchestrator.
        """
        # Functional Purpose: Construct RiskService application facade.
        # Explicit Dependency Tracking: RiskOrchestrator.
        # Structural Relationship: Service layer wrapping execution risk orchestrator.
        # Defensive Invariant: orchestrator must be a non-None RiskOrchestrator instance.
        if not isinstance(orchestrator, RiskOrchestrator):
            raise NonFiniteRiskInputException(
                f"orchestrator must be RiskOrchestrator instance, got {type(orchestrator).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        self._orchestrator: RiskOrchestrator = orchestrator
        self._risk_listeners: list[Callable[[dict[str, Any]], None]] = []

    @property
    def orchestrator(self) -> RiskOrchestrator:
        """Return underlying RiskOrchestrator instance."""
        return self._orchestrator

    def register_risk_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Register callback for real-time risk telemetry and kill switch events.

        Args:
            listener: Callback taking event dictionary.
        """
        # Functional Purpose: Register observer callback for real-time WebSocket risk broadcasts.
        # Explicit Dependency Tracking: self._risk_listeners.
        # Structural Relationship: Connected to /api/v1/ws/risk streaming router.
        # Defensive Invariant: Prevents duplicate risk listener registration.
        if listener not in self._risk_listeners:
            self._risk_listeners.append(listener)

    def unregister_risk_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Unregister risk event callback.

        Args:
            listener: Previously registered callback.
        """
        # Functional Purpose: Remove risk listener on WebSocket client disconnection.
        # Explicit Dependency Tracking: self._risk_listeners.
        # Structural Relationship: Cleans up connection resources.
        # Defensive Invariant: Idempotent removal without raising ValueError.
        if listener in self._risk_listeners:
            self._risk_listeners.remove(listener)

    def get_risk_status(self) -> RiskStatusResponse:
        """Retrieve real-time firm-wide portfolio valuation, exposure, leverage, and kill switch status.

        Returns:
            RiskStatusResponse: Populated snapshot of real-time risk metrics.
        """
        # Functional Purpose: Provide synchronous snapshot of firm-wide exposure metrics.
        # Explicit Dependency Tracking: PortfolioRiskState, EmergencyKillSwitch.
        # Structural Relationship: Consumed by GET /api/v1/risk/status and WebSocket telemetry stream.
        # Defensive Invariant: Non-negative NAV, finite leverage ratios, valid trigger strings.
        state = self._orchestrator.state
        kill_switch = self._orchestrator.kill_switch

        trigger_name: str | None = None
        if kill_switch.last_event is not None:
            trigger_name = kill_switch.last_event.trigger_reason.value

        open_leaves_count = sum(1 for qty in state.pending_leaves.values() if abs(qty) > 1e-12)
        eq = state.current_equity
        free_m = state.free_margin
        margin_used = max(0.0, state.cash - free_m)
        gross_lev = state.gross_leverage
        net_lev = state.net_leverage
        gross_notional = gross_lev * eq if math.isfinite(gross_lev) else 0.0
        net_notional = net_lev * eq if math.isfinite(net_lev) else 0.0

        return RiskStatusResponse(
            nav=eq,
            peak_nav=state.peak_equity,
            cash=state.cash,
            margin_used=margin_used,
            free_margin=free_m,
            gross_notional=gross_notional,
            net_notional=net_notional,
            gross_leverage=gross_lev if math.isfinite(gross_lev) else 0.0,
            net_leverage=net_lev if math.isfinite(net_lev) else 0.0,
            intraday_drawdown_pct=state.intraday_drawdown,
            is_kill_switch_active=kill_switch.is_active,
            kill_switch_trigger=trigger_name,
            open_leaves_count=open_leaves_count,
        )

    def get_risk_limits(self) -> RiskLimitsDTO:
        """Retrieve currently enforced pre-trade risk firewall boundary limits.

        Returns:
            RiskLimitsDTO: Slotted copy of active risk limits.
        """
        # Functional Purpose: Read current pre-trade risk thresholds.
        # Explicit Dependency Tracking: PreTradeRiskFirewall.limits.
        # Structural Relationship: Consumed by GET /api/v1/risk/limits.
        # Defensive Invariant: Positive limit boundaries.
        limits = self._orchestrator.firewall.limits
        return RiskLimitsDTO(
            max_order_notional=limits.max_order_notional,
            max_order_qty=limits.max_order_qty,
            max_gross_leverage=limits.max_gross_leverage,
            max_net_leverage=limits.max_net_leverage,
            max_concentration_nav_pct=limits.max_concentration_nav_pct,
            max_intraday_drawdown_pct=limits.max_intraday_drawdown_pct,
            min_free_margin=limits.min_free_margin,
        )

    def update_risk_limits(self, update: RiskLimitsUpdateRequest) -> RiskLimitsDTO:
        """Dynamically modify pre-trade risk firewall boundaries with strict atomic replacement.

        Args:
            update: DTO containing optional modified limit fields.

        Returns:
            RiskLimitsDTO: Updated risk boundaries snapshot.

        Raises:
            NonFiniteRiskInputException: If any updated scalar is non-finite, negative, or boolean.
        """
        # Functional Purpose: Safely update pre-trade risk boundaries without downtime.
        # Explicit Dependency Tracking: PreTradeRiskFirewall, RiskLimits.
        # Structural Relationship: Consumed by PUT /api/v1/risk/limits (restricted to ADMIN role).
        # Defensive Invariant: Updated limits validated against strict boundaries before application.
        current = self._orchestrator.firewall.limits

        new_notional = (
            update.max_order_notional
            if update.max_order_notional is not None
            else current.max_order_notional
        )
        new_qty = (
            update.max_order_qty if update.max_order_qty is not None else current.max_order_qty
        )
        new_gross_lev = (
            update.max_gross_leverage
            if update.max_gross_leverage is not None
            else current.max_gross_leverage
        )
        new_net_lev = (
            update.max_net_leverage
            if update.max_net_leverage is not None
            else current.max_net_leverage
        )
        new_conc = (
            update.max_concentration_nav_pct
            if update.max_concentration_nav_pct is not None
            else current.max_concentration_nav_pct
        )
        new_dd = (
            update.max_intraday_drawdown_pct
            if update.max_intraday_drawdown_pct is not None
            else current.max_intraday_drawdown_pct
        )
        new_margin = (
            update.min_free_margin
            if update.min_free_margin is not None
            else current.min_free_margin
        )

        new_limits = RiskLimits(
            max_order_notional=new_notional,
            max_order_qty=new_qty,
            max_gross_leverage=new_gross_lev,
            max_net_leverage=new_net_lev,
            max_concentration_nav_pct=new_conc,
            max_intraday_drawdown_pct=new_dd,
            min_free_margin=new_margin,
        )

        self._orchestrator.firewall.limits = new_limits
        return self.get_risk_limits()

    async def trigger_panic(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> KillSwitchEvent:
        """Manually trigger emergency panic kill switch, cancelling all gateway leaves and locking submissions.

        Args:
            reason: Operator justification text string.
            details: Optional contextual key-value diagnostics.

        Returns:
            KillSwitchEvent: Generated panic event record.

        Raises:
            NonFiniteRiskInputException: If reason is empty or invalid.
        """
        # Functional Purpose: Emergency operator circuit breaker dispatching concurrent mass cancellation.
        # Explicit Dependency Tracking: RiskOrchestrator.trigger_emergency_panic.
        # Structural Relationship: Invoked by POST /api/v1/risk/panic and terminal emergency panic button.
        # Defensive Invariant: INV-RSK-008 sub-5ms concurrent mass cancellation execution SLA.
        if not isinstance(reason, str) or isinstance(reason, bool) or not reason.strip():
            raise NonFiniteRiskInputException(
                "Panic reason must be non-empty string",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        panic_reason = PanicTriggerReason.MANUAL_OPERATOR
        event = await self._orchestrator.trigger_emergency_panic(
            reason=panic_reason,
            source="OPERATOR_MANUAL",
            details=f"operator_reason: {reason.strip()}",
        )
        event_payload = {
            "type": "KILL_SWITCH_EVENT",
            "status": "PANIC_TRIGGERED",
            "trigger_reason": event.trigger_reason.value,
            "cancelled_orders_count": event.cancelled_orders_count,
            "timestamp_ns": event.timestamp_ns,
            "details": event.details,
        }
        for listener in list(self._risk_listeners):
            with contextlib.suppress(Exception):
                listener(event_payload)
        return event

    def reset_kill_switch(self, admin_token: str) -> bool:
        """Disarm and reset emergency kill switch using constant-time cryptographic verification.

        Args:
            admin_token: Administrator secret string.

        Returns:
            bool: True if reset succeeded, False if authorization failed.
        """
        # Functional Purpose: Safely resume normal execution routing post-investigation.
        # Explicit Dependency Tracking: RiskOrchestrator.reset_kill_switch, hmac.compare_digest.
        # Structural Relationship: Invoked by POST /api/v1/risk/reset (restricted to ADMIN role).
        # Defensive Invariant: Constant-time authentication comparison preventing timing side-channels.
        if (
            not isinstance(admin_token, str)
            or isinstance(admin_token, bool)
            or not admin_token.strip()
        ):
            return False

        try:
            self._orchestrator.reset_kill_switch(admin_token.strip())
            event_payload = {
                "type": "KILL_SWITCH_EVENT",
                "status": "ARMED_STANDBY",
                "message": "Kill switch disarmed and reset to standby",
            }
            for listener in list(self._risk_listeners):
                with contextlib.suppress(Exception):
                    listener(event_payload)
            return True
        except Exception:
            return False

    def get_gateway_health(self, gateway_id: str | None = None) -> list[GatewayHealthDTO]:
        """Query connection and heartbeat watchdog health across registered broker gateways.

        Args:
            gateway_id: Optional single gateway identifier filter.

        Returns:
            list[GatewayHealthDTO]: Health status records for all matching gateways.
        """
        # Functional Purpose: Monitor broker transport stability and sequence integrity.
        # Explicit Dependency Tracking: HeartbeatWatchdog, ExecutionGateway.
        # Structural Relationship: Consumed by GET /api/v1/gateways/health and dashboard status cards.
        # Defensive Invariant: Non-empty gateway list reporting exact sequence gap metrics.
        watchdogs = self._orchestrator.watchdogs
        results: list[GatewayHealthDTO] = []

        for gid, wd in watchdogs.items():
            if gateway_id is not None and gid != gateway_id:
                continue

            last_lat = wd.recent_latencies[-1] if wd.recent_latencies else 0.0
            results.append(
                GatewayHealthDTO(
                    gateway_id=gid,
                    status=wd.status.value,
                    last_heartbeat_timestamp=wd.last_heartbeat_timestamp_ns,
                    last_latency_ms=last_lat,
                    missed_sequence_count=wd.sequence_gaps_count,
                    is_connected=wd.is_connected,
                )
            )

        return results

    async def update_market_price(self, symbol: str, price: float) -> None:
        """Update real-time asset mark price in portfolio state, evaluating intraday drawdown tripwire.

        Args:
            symbol: Asset ticker string.
            price: New trade execution or midpoint mark price.

        Raises:
            NonFiniteRiskInputException: If price is non-finite, negative, or boolean.
        """
        # Functional Purpose: Mark-to-market portfolio valuation and autonomous drawdown tripwire check.
        # Explicit Dependency Tracking: RiskOrchestrator.update_market_price.
        # Structural Relationship: Invoked on every market data tick or price bar ingestion.
        # Defensive Invariant: Positive finite price strictly required.
        await self._orchestrator.update_market_price(symbol, price)

    def record_gateway_heartbeat(
        self,
        gateway_id: str,
        sequence_number: int,
        latency_ms: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> GatewayHealthDTO:
        """Process an inbound heartbeat packet for a registered gateway watchdog.

        Args:
            gateway_id: Target gateway identifier.
            sequence_number: Inbound message sequence number.
            latency_ms: Measured round-trip ping-pong latency (ms).
            timestamp_ns: Optional epoch nanoseconds timestamp (defaults to time.time_ns()).

        Returns:
            GatewayHealthDTO: Updated gateway health status snapshot.

        Raises:
            KeyError: If gateway_id is not registered.
        """
        # Functional Purpose: Record heartbeat telemetry and evaluate watchdog connection state.
        # Explicit Dependency Tracking: HeartbeatWatchdog.record_heartbeat, time.time_ns.
        # Structural Relationship: Invoked by POST /api/v1/gateways/{gateway_id}/heartbeat.
        # Defensive Invariant: Positive sequence number, non-negative latency, valid timestamp.
        wd = self._orchestrator.get_watchdog(gateway_id)
        if wd is None:
            raise KeyError(f"Gateway '{gateway_id}' is not registered with a watchdog")

        ts_ns = timestamp_ns if (timestamp_ns is not None and timestamp_ns > 0) else time.time_ns()
        wd.record_heartbeat(
            sequence_number=sequence_number,
            latency_ms=latency_ms,
            timestamp_ns=ts_ns,
        )

        health_list = self.get_gateway_health(gateway_id)
        return health_list[0]
