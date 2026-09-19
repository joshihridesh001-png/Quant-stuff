"""Live execution REST endpoints for algorithmic parent orders and TCA shortfall attribution.

Purpose:
    Exposes high-performance REST routes for submitting parent orders (Poisson TWAP,
    Volume Adaptive VWAP, Nonlinear Arrival Price, Direct Market), querying live order status,
    cancelling orders, and retrieving Perold (1988) implementation shortfall TCA reports.

Dependencies:
    - FastAPI APIRouter, Depends, Query, HTTPException, status.
    - quant.api.dependencies: get_current_user, get_execution_service.
    - quant.api.v1.schemas: ParentOrderCreateRequest, ParentOrderResponse, ImplementationShortfallResponse.
    - quant.services.execution_service: ExecutionService.
    - quant.execution.venues: InvalidSORInputException, NBBOViolationException.

Structural Relationship:
    - Presentation layer route controllers mounted under /api/v1/orders in quant.main.
    - Delegates domain orchestration to ExecutionService and RiskOrchestrator.

Invariants Enforced:
    - INV-RSK-008: Submissions locked out immediately if emergency kill switch is active.
    - INV-SOR-001: Parent-child mass conservation strictly maintained across all scheduled slices.
    - INV-SOR-005: Perold (1988) implementation shortfall additive identity holds within 1e-7.
    - Rule 1: Four-tier line annotations on every function.
    - Rule 2: Diagnostic error codes and HTTP exception mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant.api.dependencies import get_current_user, get_execution_service
from quant.api.v1.schemas import (
    ImplementationShortfallResponse,
    ParentOrderCreateRequest,
    ParentOrderResponse,
)
from quant.execution.venues import InvalidSORInputException, NBBOViolationException
from quant.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orders", tags=["Live Execution Orders"])


@router.post(
    "",
    response_model=ParentOrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit algorithmic parent order",
)
async def submit_order(
    request: ParentOrderCreateRequest,
    _: dict[str, Any] = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
) -> ParentOrderResponse:
    """Submit a parent order for algorithmic slicing and risk-cleared gateway dispatch.

    Args:
        request: Validated order configuration and strategy parameters.
        _: Authenticated user claims context.
        service: Injected ExecutionService coordinator.

    Returns:
        ParentOrderResponse: Registered parent order state snapshot.

    Raises:
        HTTPException: 400 if risk kill switch is active, 422 if input validation fails.
    """
    # Functional Purpose: Entry point for algorithmic execution orders across TWAP, VWAP, Arrival Price.
    # Explicit Dependency Tracking: ExecutionService.submit_parent_order, to_order_response.
    # Structural Relationship: Maps HTTP POST payload to parent order domain entity.
    # Defensive Invariant: Submissions rejected if kill switch is tripped (ERR-RSK-008).
    try:
        parent_order = service.submit_parent_order(request)
        return service.to_order_response(parent_order)
    except InvalidSORInputException as exc:
        logger.warning("Order submission rejected by risk firewall or SOR: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except NBBOViolationException as exc:
        logger.warning("Order submission rejected due to crossed/locked NBBO: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.error("Unexpected failure submitting parent order: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order submission failed: {exc}",
        ) from exc


@router.get(
    "",
    response_model=list[ParentOrderResponse],
    summary="List active and historic parent orders",
)
def list_orders(
    symbol: str | None = Query(None, description="Optional asset ticker filter"),
    is_closed: bool | None = Query(None, description="Optional completion status filter"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of orders to return"),
    offset: int = Query(0, ge=0, description="Result pagination offset"),
    _: dict[str, Any] = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
) -> list[ParentOrderResponse]:
    """Retrieve parent orders matching filter criteria.

    Args:
        symbol: Optional symbol filter string.
        is_closed: Optional completion status filter.
        limit: Maximum results count.
        offset: Skip count for pagination.
        _: Authenticated user claims context.
        service: Injected ExecutionService coordinator.

    Returns:
        list[ParentOrderResponse]: Collection of order snapshots.
    """
    # Functional Purpose: Query active and historic order book registry.
    # Explicit Dependency Tracking: ExecutionService.list_orders, to_order_response.
    # Structural Relationship: Consumed by UI order blotters and reconciliation jobs.
    # Defensive Invariant: Enforces pagination bounds (1 <= limit <= 500).
    orders = service.list_orders(symbol=symbol, is_closed=is_closed, limit=limit, offset=offset)
    return [service.to_order_response(ord_entity) for ord_entity in orders]


@router.get(
    "/{order_id}",
    response_model=ParentOrderResponse,
    summary="Get parent order details by identifier",
)
def get_order(
    order_id: str,
    _: dict[str, Any] = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
) -> ParentOrderResponse:
    """Retrieve single parent order by unique identifier.

    Args:
        order_id: Parent order unique string identifier.
        _: Authenticated user claims context.
        service: Injected ExecutionService coordinator.

    Returns:
        ParentOrderResponse: Populated parent order snapshot.

    Raises:
        HTTPException: 404 if order does not exist.
    """
    # Functional Purpose: Fetch single parent order state and child fill breakdown.
    # Explicit Dependency Tracking: ExecutionService.get_order, to_order_response.
    # Structural Relationship: Detail view for trade audit inspection.
    # Defensive Invariant: 404 Not Found raised on missing order.
    order = service.get_order(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent order '{order_id}' not found",
        )
    return service.to_order_response(order)


@router.delete(
    "/{order_id}",
    summary="Cancel active parent order and remaining slices",
)
def cancel_order(
    order_id: str,
    _: dict[str, Any] = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
) -> dict[str, Any]:
    """Cancel open parent order, stopping subsequent slice dispatch.

    Args:
        order_id: Parent order unique string identifier.
        _: Authenticated user claims context.
        service: Injected ExecutionService coordinator.

    Returns:
        dict: Cancellation confirmation status.

    Raises:
        HTTPException: 404 if order does not exist.
    """
    # Functional Purpose: Cancel pending parent order execution schedule.
    # Explicit Dependency Tracking: ExecutionService.cancel_order.
    # Structural Relationship: Invoked by trader order blotter cancellation button.
    # Defensive Invariant: Immediate cancellation registration; atomic abort of future child slices.
    cancelled = service.cancel_order(order_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent order '{order_id}' not found or already completed",
        )
    return {"order_id": order_id, "status": "CANCELLED"}


@router.get(
    "/{order_id}/shortfall",
    response_model=ImplementationShortfallResponse,
    summary="Get Perold implementation shortfall TCA report",
)
def get_shortfall_report(
    order_id: str,
    terminal_price: float | None = Query(
        None, gt=0.0, description="Optional terminal valuation price"
    ),
    _: dict[str, Any] = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
) -> ImplementationShortfallResponse:
    """Compute and retrieve Perold (1988) implementation shortfall TCA attribution.

    Args:
        order_id: Parent order identifier.
        terminal_price: Optional terminal horizon price.
        _: Authenticated user claims context.
        service: Injected ExecutionService coordinator.

    Returns:
        ImplementationShortfallResponse: Decomposed TCA attribution metrics.

    Raises:
        HTTPException: 404 if order does not exist.
    """
    # Functional Purpose: Post-trade transaction cost analysis (TCA) attribution.
    # Explicit Dependency Tracking: ExecutionService.get_shortfall_report, to_shortfall_response.
    # Structural Relationship: Consumed by post-trade analytics and model calibration.
    # Defensive Invariant: INV-SOR-005 additive shortfall conservation verified.
    order = service.get_order(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent order '{order_id}' not found",
        )

    report = service.get_shortfall_report(order_id, terminal_price=terminal_price)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to generate shortfall report for order '{order_id}'",
        )
    return service.to_shortfall_response(report, order=order)
