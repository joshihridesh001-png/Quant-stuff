"""Institutional Pre-Trade Decision Gate REST API endpoints.

Purpose:
    Exposes endpoints for testing orders against the 5-dimensional Bayesian pre-trade filter,
    inspecting recent evaluation audit logs, and querying active risk/regime thresholds.

Dependencies:
    - FastAPI APIRouter, Depends, HTTPException, Query, status.
    - quant.api.dependencies: get_current_user, get_pre_trade_gate.
    - quant.api.v1.schemas: PreTradeEvaluateRequest, PreTradeDecisionDTO, PreTradeStatusDTO, DimensionCheckDTO.
    - quant.execution.pre_trade_gate: PreTradeDecisionGate, PreTradeDecisionRequest, NonFiniteGateInputException.

Structural Relationship:
    - Mounted under /api/v1/pre-trade in quant.main.
    - Consumed by trading terminals, autonomous trading monitors, and external agent integrations.

Invariants Enforced:
    - INV-GATE-001 (Fail-Open on Exits)
    - INV-GATE-002 (5-Dimension Bayesian Log-Odds Evaluation)
    - Non-finite input protection with 422 HTTP mapping.
    - Rule 1: Four-tier line annotations on every function.
    - Rule 2: Diagnostic error codes mapping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant.api.dependencies import get_current_user, get_pre_trade_gate
from quant.api.v1.schemas import (
    DimensionCheckDTO,
    PreTradeDecisionDTO,
    PreTradeEvaluateRequest,
    PreTradeStatusDTO,
)
from quant.execution.pre_trade_gate import (
    NonFiniteGateInputException,
    PreTradeDecisionGate,
    PreTradeDecisionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pre-trade", tags=["Pre-Trade Decision Gate"])


@router.post(
    "/evaluate",
    response_model=PreTradeDecisionDTO,
    summary="Evaluate an order through the Bayesian pre-trade decision gate",
)
def evaluate_order(
    payload: PreTradeEvaluateRequest,
    _: dict[str, Any] = Depends(get_current_user),
    gate: PreTradeDecisionGate = Depends(get_pre_trade_gate),
) -> PreTradeDecisionDTO:
    """Evaluate candidate order parameters through the institutional 5-dimensional Bayesian filter.

    Args:
        payload: Proposed order details and market context.
        _: Authenticated user context.
        gate: Injected PreTradeDecisionGate instance.

    Returns:
        PreTradeDecisionDTO: Comprehensive audit verdict and dimension check breakdown.

    Raises:
        HTTPException: 422 if inputs are non-finite or domain bounds are violated.
    """
    # Functional Purpose: Provide instant, non-blocking pre-trade safety filtering for orders.
    # Explicit Dependency Tracking: PreTradeDecisionGate.evaluate.
    # Structural Relationship: Evaluates manual or programmatic orders before execution gateway dispatch.
    # Defensive Invariant: Non-finite inputs rejected with 422 status; exits fail-open.
    try:
        req = PreTradeDecisionRequest(
            order_id=f"MANUAL-{time.time_ns()}",
            symbol=payload.symbol,
            action=payload.action,
            quantity=payload.quantity,
            reference_price=payload.reference_price,
            timestamp_ns=time.time_ns(),
            is_position_exit=payload.is_position_exit,
            market_spread_bps=payload.market_spread_bps,
            order_book_imbalance=payload.order_book_imbalance,
            macro_yield_spread=payload.macro_yield_spread,
            trailing_volatility_pct=payload.trailing_volatility_pct,
            current_bar_return_pct=payload.current_bar_return_pct,
            consecutive_losses=payload.consecutive_losses,
            current_drawdown_pct=payload.current_drawdown_pct,
            cvar_95_pct=payload.cvar_95_pct,
            data_age_seconds=payload.data_age_seconds,
        )
        res = gate.evaluate(req)

        return PreTradeDecisionDTO(
            decision_id=res.decision_id,
            order_id=res.order_id,
            symbol=res.symbol,
            action=res.action,
            allowed=res.allowed,
            decision=res.decision.value,
            toxicity_probability=res.toxicity_probability,
            confidence=res.confidence,
            primary_code=res.primary_code,
            reason=res.reason,
            checks=[
                DimensionCheckDTO(
                    name=c.name.value,
                    passed=c.passed,
                    score=c.score,
                    details=c.details,
                )
                for c in res.checks
            ],
            latency_us=res.latency_us,
        )
    except NonFiniteGateInputException as exc:
        logger.warning("Rejected non-finite pre-trade evaluation request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


@router.get(
    "/decisions",
    response_model=list[PreTradeDecisionDTO],
    summary="Query rolling audit history of pre-trade evaluations",
)
def get_recent_decisions(
    limit: int = Query(50, ge=1, le=500, description="Max decisions to return"),
    _: dict[str, Any] = Depends(get_current_user),
    gate: PreTradeDecisionGate = Depends(get_pre_trade_gate),
) -> list[PreTradeDecisionDTO]:
    """Retrieve rolling in-memory audit log of recent pre-trade decisions.

    Args:
        limit: Number of recent decisions to return.
        _: Authenticated user context.
        gate: Injected PreTradeDecisionGate instance.

    Returns:
        list[PreTradeDecisionDTO]: Recent evaluation records.
    """
    # Functional Purpose: Serve recent pre-trade evaluations to terminal HUD and monitors.
    # Explicit Dependency Tracking: PreTradeDecisionGate.history.
    # Structural Relationship: Polled by Trading Terminal HUD Tab 6 / Tab 7.
    # Defensive Invariant: Bounded collection limit.
    raw_history = gate.history[-limit:]
    raw_history.reverse()
    return [
        PreTradeDecisionDTO(
            decision_id=res.decision_id,
            order_id=res.order_id,
            symbol=res.symbol,
            action=res.action,
            allowed=res.allowed,
            decision=res.decision.value,
            toxicity_probability=res.toxicity_probability,
            confidence=res.confidence,
            primary_code=res.primary_code,
            reason=res.reason,
            checks=[
                DimensionCheckDTO(
                    name=c.name.value,
                    passed=c.passed,
                    score=c.score,
                    details=c.details,
                )
                for c in res.checks
            ],
            latency_us=res.latency_us,
        )
        for res in raw_history
    ]


@router.get(
    "/status",
    response_model=PreTradeStatusDTO,
    summary="Query Pre-Trade Decision Gate operational thresholds",
)
def get_gate_status(
    _: dict[str, Any] = Depends(get_current_user),
    gate: PreTradeDecisionGate = Depends(get_pre_trade_gate),
) -> PreTradeStatusDTO:
    """Retrieve configuration thresholds and status metrics of the decision gate."""
    # Functional Purpose: Expose active Bayesian parameters and threshold boundaries.
    # Explicit Dependency Tracking: PreTradeDecisionGate constants.
    # Structural Relationship: Inspected by risk desk consoles.
    # Defensive Invariant: Finite valid configuration thresholds.
    return PreTradeStatusDTO(
        max_toxic_probability_threshold=gate.MAX_TOXIC_PROBABILITY_THRESHOLD,
        max_data_age_seconds=gate.MAX_DATA_AGE_SECONDS,
        max_consecutive_losses=gate.MAX_CONSECUTIVE_LOSSES,
        max_drawdown_threshold_pct=gate.MAX_DRAWDOWN_THRESHOLD_PCT,
        prior_toxic_prob=gate.PRIOR_TOXIC_PROB,
        prior_log_odds=gate.PRIOR_LOG_ODDS,
        decisions_count=len(gate.history),
    )
