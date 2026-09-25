"""Non-Blocking Interactive Terminal Hotkey Handler for Live Trading Controls.

Functional Purpose:
    Provides non-blocking, cross-platform terminal key interception enabling real-time
    operator intervention during live trading sessions:
    - [SPACE]: Toggle session between RUNNING and PAUSED.
    - [K]: Trigger emergency panic kill switch (atomic mass order cancellation).
    - [R]: Disarm/re-arm pre-trade risk firewall after operator inspection.
    - [Q]: Orderly shutdown (cancel working orders and terminate cleanly).

Explicit Dependency Tracking:
    - asyncio: Thread-safe coroutine scheduling (asyncio.run_coroutine_threadsafe).
    - inspect: Inspection of sync vs async callbacks.
    - threading: Non-blocking background worker thread running terminal input loop.
    - sys, msvcrt, select: Platform-specific non-blocking character polling.

Structural Relationship:
    - Controller component for Phase 17 (Live Trading Station).
    - Attached to LiveTradingSession and terminal HUD in scripts/run_live_trader.py.

Defensive Invariants:
    - INV-KEY-001: Non-blocking terminal input: Never stalls or delays event loop execution.
    - INV-KEY-002: Thread-safe callback dispatch to running asyncio event loop.
    - INV-KEY-003: Headless safety: Degrades gracefully with zero exception when stdin is not a TTY.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any, Final

logger = logging.getLogger(__name__)

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_KEY_REGISTRATION_CONFLICT: Final[str] = "ERR-KEY-001"
ERR_KEY_INPUT_FAILURE: Final[str] = "ERR-KEY-002"
ERR_KEY_NON_FINITE_INPUT: Final[str] = "ERR-KEY-003"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class HotkeyError(Exception):
    """Base exception for terminal hotkey controller faults."""

    def __init__(self, message: str, code: str = "ERR-KEY-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


# ============================================================================
# Action Enumeration & Hotkey Manager
# ============================================================================


class HotkeyAction(StrEnum):
    """Standardized terminal hotkey identifiers."""

    PAUSE_RESUME = " "
    PANIC_KILL = "k"
    REARM = "r"
    SHUTDOWN = "q"


class HotkeyManager:
    """Cross-platform non-blocking terminal hotkey listener.

    Functional Purpose:
        Polls console keyboard input in a background daemon thread and dispatches
        actions to async callbacks on the active event loop without freezing UI or execution.

    Defensive Invariants:
        INV-KEY-001: Thread-safe non-blocking polling.
        INV-KEY-003: Graceful degradation in non-interactive/pipe/headless mode.
    """

    __slots__ = (
        "_callbacks",
        "_is_running",
        "_loop",
        "_stop_event",
        "_thread",
    )

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize the HotkeyManager."""
        self._loop = loop
        self._callbacks: dict[str, Callable[[], Coroutine[Any, Any, None] | None]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Indicate whether the hotkey polling thread is currently active."""
        return self._is_running

    def register(
        self,
        key: str | HotkeyAction,
        callback: Callable[[], Coroutine[Any, Any, None] | None],
    ) -> None:
        """Register a callback for a specific keystroke.

        Args:
            key: Single character (e.g. ' ', 'k', 'K', 'q', 'r') or HotkeyAction.
            callback: Sync or async callable taking zero arguments.

        Raises:
            HotkeyError: If callback is not callable or key is empty.
        """
        if not callable(callback):
            raise HotkeyError(
                f"Callback for key '{key}' must be callable",
                code=ERR_KEY_NON_FINITE_INPUT,
            )

        key_str = key.value if isinstance(key, HotkeyAction) else str(key)
        if not key_str:
            raise HotkeyError("Key string cannot be empty", code=ERR_KEY_NON_FINITE_INPUT)

        # Normalize key to lowercase (or space)
        norm_key = key_str.lower() if key_str != " " else " "
        self._callbacks[norm_key] = callback

    def unregister(self, key: str | HotkeyAction) -> None:
        """Deregister a hotkey callback."""
        key_str = key.value if isinstance(key, HotkeyAction) else str(key)
        norm_key = key_str.lower() if key_str != " " else " "
        self._callbacks.pop(norm_key, None)

    def start(self) -> None:
        """Start the background keyboard input polling thread if in interactive terminal."""
        if self._is_running:
            return

        if self._loop is None:
            with contextlib.suppress(RuntimeError):
                self._loop = asyncio.get_running_loop()

        # Check if stdin is an interactive TTY
        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        if not is_tty:
            logger.info("Non-interactive / headless environment detected. Hotkeys disabled.")
            return

        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._poll_input_worker,
            name="HotkeyListenerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("Hotkey listener thread started")

    def stop(self) -> None:
        """Stop the background keyboard input polling thread cleanly."""
        if not self._is_running:
            return

        self._stop_event.set()
        self._is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
            self._thread = None
        logger.info("Hotkey listener thread stopped")

    def trigger_key(self, key: str) -> None:
        """Programmatically trigger a hotkey action (useful for testing or external inputs)."""
        norm_key = key.lower() if key != " " else " "
        cb = self._callbacks.get(norm_key)
        if cb is not None:
            self._dispatch_callback(cb)

    # ------------------------------------------------------------------------
    # Internal Input Polling & Dispatch
    # ------------------------------------------------------------------------

    def _poll_input_worker(self) -> None:
        """Background thread target polling keyboard characters without blocking."""
        while not self._stop_event.is_set():
            try:
                key = self._read_single_char_nonblocking()
                if key:
                    self.trigger_key(key)
            except Exception as exc:
                logger.debug("Exception in hotkey poll loop: %s", exc)
            time.sleep(0.05)

    def _read_single_char_nonblocking(self) -> str | None:
        """Cross-platform non-blocking single character read."""
        if sys.platform == "win32":
            import msvcrt

            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode("utf-8", errors="ignore")
                except UnicodeDecodeError:
                    return None
            return None
        else:
            import select

            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                return sys.stdin.read(1)
            return None

    def _dispatch_callback(
        self,
        callback: Callable[[], Coroutine[Any, Any, None] | None],
    ) -> None:
        """Execute callback either as an async task on the loop or directly if sync."""
        try:
            if inspect.iscoroutinefunction(callback):
                if self._loop is not None and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(callback(), self._loop)
                else:
                    asyncio.run(callback())
            else:
                res = callback()
                if inspect.iscoroutine(res):
                    if self._loop is not None and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(res, self._loop)
                    else:
                        asyncio.run(res)
        except Exception as exc:
            logger.error("Error executing hotkey callback: %s", exc)
