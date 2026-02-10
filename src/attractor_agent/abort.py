"""Cooperative cancellation primitive shared across all 3 layers.

The AbortSignal is threaded through Attractor → Agent Loop → LLM SDK
to enable cooperative cancellation of long-running operations.

Design reference: Our Issue 4 design decision.
"""

from __future__ import annotations

from collections.abc import Callable


class AbortSignal:
    """Cooperative cancellation signal.

    Thread-safe flag that can be checked at the top of every loop
    iteration and before each handler/tool call. Once set, it cannot
    be unset.

    Usage::

        signal = AbortSignal()

        # In the loop:
        if signal.is_set:
            break

        # From the cancel endpoint:
        signal.set()
    """

    def __init__(self) -> None:
        self._is_set: bool = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def is_set(self) -> bool:
        """Whether the abort has been requested."""
        return self._is_set

    def set(self) -> None:
        """Request abort. Fires all registered callbacks."""
        if self._is_set:
            return
        self._is_set = True
        for cb in self._callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def on_abort(self, callback: Callable[[], None]) -> None:
        """Register a callback to fire when abort is requested.

        If already aborted, fires immediately.
        """
        self._callbacks.append(callback)
        if self._is_set:
            try:
                callback()
            except Exception:  # noqa: BLE001
                pass
