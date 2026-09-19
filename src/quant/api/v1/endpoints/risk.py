"""Risk monitor and emergency kill switch REST endpoints.

Purpose:
    Exposes real-time portfolio risk telemetry (NAV, gross/net leverage, margin sufficiency),
    pre-trade risk boundary inspection/updating, manual panic kill switch triggering, and
    authenticated administrative kill switch resetting.

Dependencies:
    - FastAPI APIRouter, Depends, HTTPException, status.
    - quant.api.dependencies: get_current_user, get_risk_service, require_role.
    - quant.api.v1.schemas: (
        KillSwitchResetRequest,
        PanicTriggerRequest,
        RiskLimitsDTO,
        RiskLimitsUpdateRequest,
        RiskStatusResponse,
    )
    - quant.services.risk_service: RiskService.
    - quant.execution.risk: NonFiniteRiskInputException.

Structural Relationship:
    - Mounted under /api/v1/risk in quant.main.
    - Connects frontend risk telemetry widgets and emergency desk controls to RiskService.

Invariants Enforced:
    - INV-RSK-008: Immediate execution lockout and multi-venue mass cancellation on panic trigger.
    - Constant-time secret comparison preventing timing side-channels during kill switch reset.
    - Role-Based Access Control (RBAC): Dynamic limit updates and resets strictly restricted to ADMIN role.
    - Rule 1: Four-tier line annotations on every function.
    - Rule 2: Diagnostic error codes and HTTP exception mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from quant.api.dependencies import get_current_user, get_risk_service, require_role
from quant.api.v1.schemas import (
    KillSwitchResetRequest,
    PanicTriggerRequest,
    RiskLimitsDTO,
    RiskLimitsUpdateRequest,
    RiskStatusResponse,
)
from quant.execution.risk import NonFiniteRiskInputException
from quant.services.risk_service import RiskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk Monitor & Kill Switch"])


@router.get(
    "/status",
    response_model=RiskStatusResponse,
    summary="Get real-time firm-wide portfolio risk metrics",
)
def get_risk_status(
    _: dict[str, Any] = Depends(get_current_user),
    service: RiskService = Depends(get_risk_service),
) -> RiskStatusResponse:
    """Retrieve real-time firm-wide portfolio valuation, exposure, leverage, and kill switch status.

    Args:
        _: Authenticated user claims context.
        service: Injected RiskService coordinator.

    Returns:
        RiskStatusResponse: Real-time risk metrics snapshot.
    """
    # Functional Purpose: Serve real-time exposure, leverage, and circuit breaker status.
    # Explicit Dependency Tracking: RiskService.get_risk_status.
    # Structural Relationship: Primary polling / telemetry endpoint for trading desk dashboards.
    # Defensive Invariant: Yields non-negative NAV, finite leverage ratios, and accurate kill switch state.
    return service.get_risk_status()


@router.get(
    "/limits",
    response_model=RiskLimitsDTO,
    summary="Get active pre-trade risk firewall boundaries",
)
def get_risk_limits(
    _: dict[str, Any] = Depends(get_current_user),
    service: RiskService = Depends(get_risk_service),
) -> RiskLimitsDTO:
    """Retrieve currently enforced pre-trade risk firewall boundary thresholds.

    Args:
        _: Authenticated user claims context.
        service: Injected RiskService coordinator.

    Returns:
        RiskLimitsDTO: Slotted copy of active risk limits.
    """
    # Functional Purpose: Inspect active pre-trade firewall parameters.
    # Explicit Dependency Tracking: RiskService.get_risk_limits.
    # Structural Relationship: Config viewing for risk officers and automated monitoring.
    # Defensive Invariant: All thresholds strictly positive scalars.
    return service.get_risk_limits()


@router.put(
    "/limits",
    response_model=RiskLimitsDTO,
    summary="Dynamically update pre-trade risk firewall boundaries (Admin only)",
)
def update_risk_limits(
    update: RiskLimitsUpdateRequest,
    _: dict[str, Any] = Depends(require_role(["ADMIN"])),
    service: RiskService = Depends(get_risk_service),
) -> RiskLimitsDTO:
    """Dynamically modify pre-trade risk boundaries without service interruption.

    Args:
        update: DTO containing optional modified limit fields.
        _: Authenticated administrator claims context.
        service: Injected RiskService coordinator.

    Returns:
        RiskLimitsDTO: Updated risk boundaries snapshot.

    Raises:
        HTTPException: 422 if boundary values are invalid or non-finite.
    """
    # Functional Purpose: Live risk ceiling modification during shifting market volatility.
    # Explicit Dependency Tracking: RiskService.update_risk_limits.
    # Structural Relationship: Invoked by Chief Risk Officer console.
    # Defensive Invariant: RBAC restricted to ADMIN role; atomic firewall limit replacement.
    try:
        return service.update_risk_limits(update)
    except NonFiniteRiskInputException as exc:
        logger.warning("Rejected risk limit update with invalid scalar: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


@router.post(
    "/panic",
    summary="Trigger emergency panic kill switch",
)
async def trigger_panic(
    payload: PanicTriggerRequest,
    _: dict[str, Any] = Depends(get_current_user),
    service: RiskService = Depends(get_risk_service),
) -> dict[str, Any]:
    """Manually trip emergency panic kill switch: cancels all open orders and locks submissions.

    Args:
        payload: Panic trigger request with operator justification.
        _: Authenticated user claims context.
        service: Injected RiskService coordinator.

    Returns:
        dict: Generated KillSwitchEvent audit confirmation.
    """
    # Functional Purpose: Emergency operator tripwire for instantaneous market isolation.
    # Explicit Dependency Tracking: RiskService.trigger_panic.
    # Structural Relationship: Hot emergency button on terminal dashboard HUD.
    # Defensive Invariant: INV-RSK-008 sub-5ms concurrent mass cancellation execution.
    try:
        event = await service.trigger_panic(reason=payload.reason, details=payload.details)
        return {
            "status": "PANIC_TRIGGERED",
            "trigger_reason": event.trigger_reason.value,
            "cancelled_orders_count": event.cancelled_orders_count,
            "timestamp_ns": event.timestamp_ns,
            "details": event.details,
        }
    except NonFiniteRiskInputException as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


@router.post(
    "/reset",
    summary="Disarm and reset emergency kill switch (Admin only)",
)
def reset_kill_switch(
    payload: KillSwitchResetRequest,
    _: dict[str, Any] = Depends(require_role(["ADMIN"])),
    service: RiskService = Depends(get_risk_service),
) -> dict[str, Any]:
    """Disarm and reset emergency kill switch back to ARMED_STANDBY using admin secret.

    Args:
        payload: Authenticated reset request containing cryptographic secret.
        _: Authenticated administrator claims context.
        service: Injected RiskService coordinator.

    Returns:
        dict: Reset status confirmation.

    Raises:
        HTTPException: 403 if admin token verification fails.
    """
    # Functional Purpose: Post-incident resumption of order routing.
    # Explicit Dependency Tracking: RiskService.reset_kill_switch.
    # Structural Relationship: Admin-only recovery operation.
    # Defensive Invariant: Constant-time authentication check preventing timing attacks.
    success = service.reset_kill_switch(payload.admin_token)
    if not success:
        logger.warning("Unauthorized kill switch reset attempt rejected")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid administrator secret token for kill switch reset",
        )
    return {
        "status": "ARMED_STANDBY",
        "success": True,
        "message": "Emergency kill switch successfully disarmed and re-armed to standby",
    }
