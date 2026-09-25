"""WebSocket streaming endpoints for live execution orders, fills, risk telemetry, and kill switch events.

Purpose:
    Provides full-duplex WebSocket streams for institutional trading terminals:
    - /api/v1/ws/executions (and /api/v1/ws/orders): Real-time parent order lifecycle updates,
      algorithmic slicing progress, and child order fills.
    - /api/v1/ws/risk: Real-time firm-wide risk metrics (NAV, margin, leverage, drawdown),
      gateway connectivity heartbeats, and emergency kill switch alert broadcasts.

Dependencies:
    - asyncio: Non-blocking asynchronous queues, tasks, and telemetry timers.
    - contextlib: Exception suppression for optional queue deliveries.
    - time: High-resolution timestamping.
    - fastapi: APIRouter, WebSocket, WebSocketDisconnect, Depends, status.
    - quant.api.dependencies: get_execution_service, get_risk_service.
    - quant.api.v1.schemas: ChildOrderDTO, ParentOrderResponse.
    - quant.execution.parent_order: ParentOrder.
    - quant.services.execution_service: ExecutionService.
    - quant.services.risk_service: RiskService.

Structural Relationship:
    - Presentation layer streaming transport mounted under /api/v1 in quant.main.
    - Bridges backend ExecutionService and RiskService event streams to trading_terminal.html.

Invariants Enforced:
    - Immediate initial state snapshot delivery upon successful WebSocket handshake.
    - Graceful connection teardown and listener unregistration on WebSocketDisconnect.
    - Non-blocking event queuing preventing slow clients from blocking engine execution.
    - Rule 1: Four-tier line annotations on every function and class.
    - Rule 2: Diagnostic error code logging and structured message envelopes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from quant.api.dependencies import get_execution_service, get_risk_service
from quant.api.v1.schemas import ChildOrderDTO
from quant.execution.parent_order import ParentOrder
from quant.services.execution_service import ExecutionService
from quant.services.risk_service import RiskService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket Streaming"])


# ============================================================================
# 1. Live Executions & Orders WebSocket Stream
# ============================================================================


@router.websocket("/ws/executions")
@router.websocket("/ws/orders")
async def websocket_executions(
    websocket: WebSocket,
    service: ExecutionService = Depends(get_execution_service),
) -> None:
    """Stream live parent order status changes, child slice executions, and order cancellations.

    Args:
        websocket: Inbound WebSocket client connection.
        service: Injected ExecutionService coordinator.
    """
    # Functional Purpose: Real-time execution blotter and child fill telemetry stream.
    # Explicit Dependency Tracking: ExecutionService order/fill listener interfaces.
    # Structural Relationship: Primary live order feed for trading terminal order book & blotter.
    # Defensive Invariant: Handshake sends full snapshot before listening for live deltas.
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/executions")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

    # 1. Send initial snapshot of recent orders
    try:
        recent_orders = service.list_orders(limit=50)
        snapshot_payload = {
            "type": "SNAPSHOT",
            "timestamp_ns": time.time_ns(),
            "orders": [service.to_order_response(o).model_dump() for o in recent_orders],
        }
        await websocket.send_json(snapshot_payload)
    except Exception as exc:
        logger.warning("Failed sending executions initial snapshot: %s", exc)
        await websocket.close()
        return

    # 2. Define reactive observer callbacks
    def on_order_event(order: ParentOrder, event_type: str) -> None:
        payload = {
            "type": "ORDER_UPDATE",
            "event": event_type,
            "timestamp_ns": time.time_ns(),
            "order": service.to_order_response(order).model_dump(),
        }
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(payload)

    def on_child_fill(parent_id: str, fill: ChildOrderDTO) -> None:
        payload = {
            "type": "CHILD_FILL",
            "parent_id": parent_id,
            "timestamp_ns": time.time_ns(),
            "fill": fill.model_dump(),
        }
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(payload)

    service.register_order_listener(on_order_event)
    service.register_fill_listener(on_child_fill)

    # 3. Concurrent read and write workers
    async def read_inbound() -> None:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and data.get("action") == "PING":
                await queue.put(
                    {
                        "type": "PONG",
                        "timestamp_ns": time.time_ns(),
                    }
                )

    async def write_outbound() -> None:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            queue.task_done()

    read_task = asyncio.create_task(read_inbound())
    write_task = asyncio.create_task(write_outbound())

    try:
        # Wait until either read or write fails / disconnects
        done, pending = await asyncio.wait(
            [read_task, write_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task_exc = task.exception()
            if task_exc and not isinstance(task_exc, (WebSocketDisconnect, asyncio.CancelledError)):
                logger.info("Executions WebSocket stream completed with: %s", task_exc)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/executions")
    finally:
        read_task.cancel()
        write_task.cancel()
        service.unregister_order_listener(on_order_event)
        service.unregister_fill_listener(on_child_fill)


# ============================================================================
# 2. Live Risk & Emergency Kill Switch WebSocket Stream
# ============================================================================


@router.websocket("/ws/risk")
async def websocket_risk(
    websocket: WebSocket,
    service: RiskService = Depends(get_risk_service),
) -> None:
    """Stream real-time portfolio risk telemetry, gateway status, and kill switch panic alerts.

    Args:
        websocket: Inbound WebSocket client connection.
        service: Injected RiskService coordinator.
    """
    # Functional Purpose: Real-time risk telemetry, gateway watchdog, and kill switch HUD stream.
    # Explicit Dependency Tracking: RiskService risk listener and status queries.
    # Structural Relationship: Powers trading terminal real-time risk gauge, P&L HUD, and kill switch badge.
    # Defensive Invariant: Dispatches initial snapshot followed by periodic 1.0s heartbeats and event alerts.
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/risk")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

    # 1. Send initial snapshot of risk state and gateways
    try:
        risk_status = service.get_risk_status()
        gateways = service.get_gateway_health()
        snapshot_payload = {
            "type": "SNAPSHOT",
            "timestamp_ns": time.time_ns(),
            "risk": risk_status.model_dump(),
            "gateways": [g.model_dump() for g in gateways],
        }
        await websocket.send_json(snapshot_payload)
    except Exception as exc:
        logger.warning("Failed sending risk initial snapshot: %s", exc)
        await websocket.close()
        return

    # 2. Define reactive observer callback for kill switch and limit events
    def on_risk_event(event_data: dict[str, Any]) -> None:
        risk_snap = service.get_risk_status()
        payload = {
            "type": "KILL_SWITCH_EVENT",
            "timestamp_ns": time.time_ns(),
            **event_data,
            "risk": risk_snap.model_dump(),
        }
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(payload)

    service.register_risk_listener(on_risk_event)

    # 3. Telemetry ticker task pushing periodic risk updates every 1.0s
    async def telemetry_ticker() -> None:
        while True:
            await asyncio.sleep(1.0)
            status_update = service.get_risk_status()
            gateways_update = service.get_gateway_health()
            payload = {
                "type": "RISK_UPDATE",
                "timestamp_ns": time.time_ns(),
                "risk": status_update.model_dump(),
                "gateways": [g.model_dump() for g in gateways_update],
            }
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    # 4. Inbound reader task handling PING/PONG
    async def read_inbound() -> None:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and data.get("action") == "PING":
                await queue.put(
                    {
                        "type": "PONG",
                        "timestamp_ns": time.time_ns(),
                    }
                )

    # 5. Outbound sender task draining queue
    async def write_outbound() -> None:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            queue.task_done()

    ticker_task = asyncio.create_task(telemetry_ticker())
    read_task = asyncio.create_task(read_inbound())
    write_task = asyncio.create_task(write_outbound())

    try:
        done, pending = await asyncio.wait(
            [ticker_task, read_task, write_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task_exc = task.exception()
            if task_exc and not isinstance(task_exc, (WebSocketDisconnect, asyncio.CancelledError)):
                logger.info("Risk WebSocket stream completed with: %s", task_exc)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/risk")
    finally:
        ticker_task.cancel()
        read_task.cancel()
        write_task.cancel()
        service.unregister_risk_listener(on_risk_event)
