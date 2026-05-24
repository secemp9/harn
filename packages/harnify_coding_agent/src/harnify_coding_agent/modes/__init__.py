"""Shared mode exports for coding-agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .print_mode import PrintModeOptions, run_print_mode, runPrintMode

__all__ = ["PrintModeOptions", "runPrintMode", "run_print_mode"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import print_mode as _print_mode

        return getattr(_print_mode, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
