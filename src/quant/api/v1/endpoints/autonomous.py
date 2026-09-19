"""Autonomous live trading swarm orchestrator REST endpoints.

Purpose:
    Exposes high-performance control and telemetry REST routes for starting, stopping,
    pausing, resuming, querying, and stepping the autonomous live trading swarm daemon.

Dependencies:
    - FastAPI APIRouter, Depends, status.
    - quant.api.dependencies: get_current_user, get_autonomous_trader.
    - quant.api.v1.schemas: AutonomousStatusDTO, AutonomousStepReportDTO.
    - quant.services.autonomous_trader: AutonomousTradingEngine.

Structural Relationship:
    - Presentation layer route controllers mounted under /api/v1/autonomous in quant.main.
    - Consumed by Institutional Trading Terminal HUD controls and external automation bots.

Invariants Enforced:
    - Thread-safe async task lifecycle transitions across IDLE, RUNNING, PAUSED, STOPPED.
    - Rule 1: Four-tier line annotations on every endpoint handler.
    - Rule 2: Standardized status DTOs and HTTP response codes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, status

from quant.api.dependencies import get_autonomous_trader, get_current_user
from quant.api.v1.schemas import AutonomousStatusDTO, AutonomousStepReportDTO
from quant.services.autonomous_trader import AutonomousTradingEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autonomous", tags=["Autonomous Live Trading Swarm"])


@router.get(
    "/status",
    response_model=AutonomousStatusDTO,
    summary="Query operational state and telemetry of the autonomous trading engine",
)
def get_autonomous_status(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStatusDTO:
    """Retrieve operational state, rebalancing iteration, universe, and allocations.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStatusDTO: Snapshot of daemon status and metrics.
    """
    # Functional Purpose: Provide live status and telemetry to trading terminal HUD.
    # Explicit Dependency Tracking: AutonomousTradingEngine properties.
    # Structural Relationship: Polled by frontend header badge and telemetry bar.
    # Defensive Invariant: Always returns valid state enum and non-negative iteration counter.
    return AutonomousStatusDTO(
        state=trader.state.value,
        iteration=trader.iteration,
        universe=trader.universe,
        target_allocations=trader.target_allocations,
        last_step=trader.last_report.__dict__ if trader.last_report else None,
    )


@router.post(
    "/start",
    response_model=AutonomousStatusDTO,
    status_code=status.HTTP_200_OK,
    summary="Start continuous background rebalancing loop",
)
async def start_autonomous_loop(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStatusDTO:
    """Initiate background autonomous trading loop clock.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStatusDTO: Updated status reflecting RUNNING state.
    """
    # Functional Purpose: Start continuous background rebalancing swarm.
    # Explicit Dependency Tracking: AutonomousTradingEngine.start.
    # Structural Relationship: Invoked by Start Autonomous Engine HUD button.
    # Defensive Invariant: Idempotent if already in RUNNING state.
    await trader.start()
    return AutonomousStatusDTO(
        state=trader.state.value,
        iteration=trader.iteration,
        universe=trader.universe,
        target_allocations=trader.target_allocations,
        last_step=trader.last_report.__dict__ if trader.last_report else None,
    )


@router.post(
    "/stop",
    response_model=AutonomousStatusDTO,
    status_code=status.HTTP_200_OK,
    summary="Stop continuous background rebalancing loop",
)
async def stop_autonomous_loop(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStatusDTO:
    """Halt background autonomous trading loop and cancel active background task.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStatusDTO: Updated status reflecting STOPPED state.
    """
    # Functional Purpose: Gracefully terminate background clock loop.
    # Explicit Dependency Tracking: AutonomousTradingEngine.stop.
    # Structural Relationship: Invoked by Stop Engine button or application shutdown.
    # Defensive Invariant: Cancels task and drains cancellation cleanly.
    await trader.stop()
    return AutonomousStatusDTO(
        state=trader.state.value,
        iteration=trader.iteration,
        universe=trader.universe,
        target_allocations=trader.target_allocations,
        last_step=trader.last_report.__dict__ if trader.last_report else None,
    )


@router.post(
    "/pause",
    response_model=AutonomousStatusDTO,
    status_code=status.HTTP_200_OK,
    summary="Pause autonomous trading loop without canceling worker task",
)
def pause_autonomous_loop(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStatusDTO:
    """Pause autonomous trading cycles while retaining background worker.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStatusDTO: Updated status reflecting PAUSED state.
    """
    # Functional Purpose: Temporarily suspend order generation while keeping task alive.
    # Explicit Dependency Tracking: AutonomousTradingEngine.pause.
    # Structural Relationship: Invoked by Pause button or temporary manual intervention.
    # Defensive Invariant: Immediate state transition to PAUSED.
    trader.pause()
    return AutonomousStatusDTO(
        state=trader.state.value,
        iteration=trader.iteration,
        universe=trader.universe,
        target_allocations=trader.target_allocations,
        last_step=trader.last_report.__dict__ if trader.last_report else None,
    )


@router.post(
    "/resume",
    response_model=AutonomousStatusDTO,
    status_code=status.HTTP_200_OK,
    summary="Resume autonomous trading from PAUSED state",
)
def resume_autonomous_loop(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStatusDTO:
    """Resume trading cycles from PAUSED state.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStatusDTO: Updated status reflecting RUNNING state.
    """
    # Functional Purpose: Resume execution ticks following pause.
    # Explicit Dependency Tracking: AutonomousTradingEngine.resume.
    # Structural Relationship: Invoked by Resume button.
    # Defensive Invariant: Only transitions if previously PAUSED.
    trader.resume()
    return AutonomousStatusDTO(
        state=trader.state.value,
        iteration=trader.iteration,
        universe=trader.universe,
        target_allocations=trader.target_allocations,
        last_step=trader.last_report.__dict__ if trader.last_report else None,
    )


@router.post(
    "/step",
    response_model=AutonomousStepReportDTO,
    status_code=status.HTTP_200_OK,
    summary="Execute single discrete rebalance cycle iteration",
)
async def step_autonomous_loop(
    _: dict[str, Any] = Depends(get_current_user),
    trader: AutonomousTradingEngine = Depends(get_autonomous_trader),
) -> AutonomousStepReportDTO:
    """Execute a single discrete rebalance step immediately.

    Args:
        _: Authenticated user claims context.
        trader: Injected AutonomousTradingEngine daemon.

    Returns:
        AutonomousStepReportDTO: Telemetry report from the completed cycle.
    """
    # Functional Purpose: Single-cycle execution for step-by-step verification and manual triggers.
    # Explicit Dependency Tracking: AutonomousTradingEngine.step_once.
    # Structural Relationship: Invoked by Single Step button on Terminal HUD.
    # Defensive Invariant: Atomically advances iteration counter and computes target allocations.
    report = await trader.step_once()
    return AutonomousStepReportDTO(
        iteration=report.iteration,
        timestamp_ns=report.timestamp_ns,
        universe=report.universe,
        target_allocations=report.target_allocations,
        current_positions=report.current_positions,
        orders_dispatched=report.orders_dispatched,
        duration_ms=report.duration_ms,
        haircut=report.haircut,
        is_kill_switch_active=report.is_kill_switch_active,
    )
