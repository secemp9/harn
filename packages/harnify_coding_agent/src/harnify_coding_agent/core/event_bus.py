"""Lightweight event bus for coding-agent runtime services."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

type EventHandler = Callable[[Any], Any]


class EventBus(Protocol):
    def emit(self, channel: str, data: Any) -> None: ...

    def on(self, channel: str, handler: EventHandler) -> Callable[[], None]: ...


class EventBusController(EventBus, Protocol):
    def clear(self) -> None: ...


@dataclass(slots=True)
class _EventBusController:
    _listeners: dict[str, list[EventHandler]] = field(default_factory=dict)

    def emit(self, channel: str, data: Any) -> None:
        for handler in list(self._listeners.get(channel, [])):
            try:
                result = handler(data)
            except Exception as error:
                _report_handler_error(channel, error)
                continue
            if inspect.isawaitable(result):
                _schedule_awaitable(channel, result)

    def on(self, channel: str, handler: EventHandler) -> Callable[[], None]:
        listeners = self._listeners.setdefault(channel, [])
        listeners.append(handler)

        def unsubscribe() -> None:
            current = self._listeners.get(channel)
            if current and handler in current:
                current.remove(handler)

        return unsubscribe

    def clear(self) -> None:
        self._listeners.clear()


async def _await_handler(channel: str, awaitable: Awaitable[Any]) -> None:
    try:
        await awaitable
    except Exception as error:
        _report_handler_error(channel, error)


def _schedule_awaitable(channel: str, awaitable: Awaitable[Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_handler(channel, awaitable))
        return
    loop.create_task(_await_handler(channel, awaitable))


def _report_handler_error(channel: str, error: Exception) -> None:
    print(f"Event handler error ({channel}): {error}", file=sys.stderr)


def create_event_bus() -> EventBusController:
    return _EventBusController()


createEventBus = create_event_bus

__all__ = [
    "EventBus",
    "EventBusController",
    "createEventBus",
    "create_event_bus",
]
