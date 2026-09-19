"""Broker gateway health monitoring and heartbeat watchdog REST endpoints.

Purpose:
    Exposes endpoints for inspecting execution gateway transport health, connection state,
    round-trip ping-pong latencies, and sequence gaps across registered brokers, plus an ingestion
    hook for transport heartbeat telemetry.

Dependencies:
    - FastAPI APIRouter, Depends, Query, HTTPException, status.
    - quant.api.dependencies: get_current_user, get_risk_service.
    - quant.api.v1.schemas: GatewayHealthDTO, HeartbeatPingRequest.
    - quant.services.risk_service: RiskService.

Structural Relationship:
    - Mounted under /api/v1/gateways in quant.main.
    - Consumed by frontend gateway status indicators and automated infrastructure monitors.

Invariants Enforced:
    - Transport latency and sequence gap tracking per gateway watchdog.
    - Rule 1: Four-tier line annotations on every function.
    - Rule 2: Diagnostic error codes and HTTP exception mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant.api.dependencies import get_current_user, get_risk_service
from quant.api.v1.schemas import GatewayHealthDTO, HeartbeatPingRequest
from quant.services.risk_service import RiskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gateways", tags=["Gateway Transport & Heartbeat"])


@router.get(
    "/health",
    response_model=list[GatewayHealthDTO],
    summary="Query broker gateway connectivity and watchdog health",
)
def get_gateway_health(
    gateway_id: str | None = Query(None, description="Optional single gateway identifier filter"),
    _: dict[str, Any] = Depends(get_current_user),
    service: RiskService = Depends(get_risk_service),
) -> list[GatewayHealthDTO]:
    """Retrieve connection and heartbeat watchdog health across registered broker gateways.

    Args:
        gateway_id: Optional single gateway identifier filter.
        _: Authenticated user claims context.
        service: Injected RiskService coordinator.

    Returns:
        list[GatewayHealthDTO]: Health status records for all matching gateways.
    """
    # Functional Purpose: Monitor transport stability, sequence continuity, and ping latencies.
    # Explicit Dependency Tracking: RiskService.get_gateway_health.
    # Structural Relationship: Health polling endpoint for broker cards on terminal UI.
    # Defensive Invariant: Non-empty result collection for registered gateways.
    return service.get_gateway_health(gateway_id=gateway_id)


@router.post(
    "/{gateway_id}/heartbeat",
    response_model=GatewayHealthDTO,
    summary="Record gateway transport heartbeat pulse",
)
def record_heartbeat(
    gateway_id: str,
    payload: HeartbeatPingRequest,
    _: dict[str, Any] = Depends(get_current_user),
    service: RiskService = Depends(get_risk_service),
) -> GatewayHealthDTO:
    """Process an inbound heartbeat packet, validate sequence numbers, and evaluate latency degradation.

    Args:
        gateway_id: Target exchange or broker gateway identifier.
        payload: Heartbeat pulse parameters (sequence number, latency, timestamp).
        _: Authenticated user claims context.
        service: Injected RiskService coordinator.

    Returns:
        GatewayHealthDTO: Updated health snapshot for the targeted gateway.

    Raises:
        HTTPException: 404 if gateway is not registered with a watchdog.
    """
    # Functional Purpose: Keep-alive watchdog ingestion updating sequence high-watermark.
    # Explicit Dependency Tracking: RiskService.record_gateway_heartbeat.
    # Structural Relationship: Ingested from FIX/WebSocket gateway transport reader loops.
    # Defensive Invariant: Positive sequence number and non-negative latency strictly verified.
    try:
        return service.record_gateway_heartbeat(
            gateway_id=gateway_id,
            sequence_number=payload.sequence_number,
            latency_ms=payload.latency_ms,
            timestamp_ns=payload.timestamp_ns,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gateway '{gateway_id}' is not registered with a watchdog",
        ) from exc
    except Exception as exc:
        logger.error("Failed recording gateway heartbeat: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Heartbeat recording failed: {exc}",
        ) from exc
