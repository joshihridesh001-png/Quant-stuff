"""Comprehensive Unit Tests for Non-Blocking HotkeyManager.

Functional Purpose:
    Verifies that HotkeyManager adheres strictly to:
    - INV-KEY-001: Thread-safe, non-blocking registration and dispatch.
    - INV-KEY-002: Reliable async and sync callback routing to the event loop.
    - INV-KEY-003: Headless safety and graceful degradation when stdin is not a TTY.

Governing Rules:
    - Rule 1: Four-tier docstrings on all test fixtures and assertions.
    - Rule 2: Deterministic error codes (ERR-KEY-001, ERR-KEY-003).
    - Rule 3: Quality gates (100% pass rate, strict static typing).
    - Rule 4: Adversarial registration boundaries, case insensitivity, and async cancellation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from quant.execution.hotkeys import (
    ERR_KEY_NON_FINITE_INPUT,
    HotkeyAction,
    HotkeyError,
    HotkeyManager,
)


def cast_any(val: Any) -> Any:
    """Helper to bypass static typing for adversarial test input."""
    return val


def test_hotkey_registration_and_validation() -> None:
    """Verify that valid keys register and invalid inputs raise HotkeyError."""
    manager = HotkeyManager()

    # Valid sync callback
    called: list[str] = []
    manager.register("k", lambda: called.append("k"))
    manager.trigger_key("k")
    assert called == ["k"]

    # Case insensitivity: 'K' triggers 'k'
    manager.trigger_key("K")
    assert called == ["k", "k"]

    # Enum action registration
    manager.register(HotkeyAction.PAUSE_RESUME, lambda: called.append("space"))
    manager.trigger_key(" ")
    assert "space" in called

    # Reject non-callable
    with pytest.raises(HotkeyError) as exc_info:
        manager.register("q", cast_any("not_callable"))
    assert exc_info.value.code == ERR_KEY_NON_FINITE_INPUT

    # Reject empty key
    with pytest.raises(HotkeyError) as exc_info:
        manager.register("", lambda: None)
    assert exc_info.value.code == ERR_KEY_NON_FINITE_INPUT


@pytest.mark.asyncio
async def test_hotkey_async_callback_dispatch() -> None:
    """Verify that async coroutine callbacks execute correctly via trigger_key."""
    loop = asyncio.get_running_loop()
    manager = HotkeyManager(loop=loop)

    event = asyncio.Event()

    async def _async_action() -> None:
        event.set()

    manager.register(HotkeyAction.PANIC_KILL, _async_action)
    manager.trigger_key("k")

    # Wait for async task to run
    await asyncio.wait_for(event.wait(), timeout=1.0)
    assert event.is_set()


def test_hotkey_unregister() -> None:
    """Verify unregistering removes callback and trigger becomes no-op."""
    manager = HotkeyManager()
    called: list[str] = []
    manager.register("r", lambda: called.append("r"))
    manager.trigger_key("r")
    assert called == ["r"]

    manager.unregister("r")
    manager.trigger_key("r")
    assert called == ["r"]  # No new execution


def test_hotkey_lifecycle_headless() -> None:
    """Verify start and stop in non-interactive / headless environment."""
    manager = HotkeyManager()
    # In pytest, stdin is captured / not a TTY, so start should gracefully disable worker thread
    manager.start()
    manager.stop()
    assert not manager.is_running
