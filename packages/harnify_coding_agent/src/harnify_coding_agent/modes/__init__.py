"""Shared mode exports for coding-agent."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "InteractiveMode": ("harnify_coding_agent.modes.interactive.interactive_mode", "InteractiveMode"),
    "InteractiveModeOptions": ("harnify_coding_agent.modes.interactive.interactive_mode", "InteractiveModeOptions"),
    "PrintModeOptions": ("harnify_coding_agent.modes.print_mode", "PrintModeOptions"),
    "runPrintMode": ("harnify_coding_agent.modes.print_mode", "runPrintMode"),
    "ModelInfo": ("harnify_coding_agent.modes.rpc.rpc_client", "ModelInfo"),
    "RpcClient": ("harnify_coding_agent.modes.rpc.rpc_client", "RpcClient"),
    "RpcClientOptions": ("harnify_coding_agent.modes.rpc.rpc_client", "RpcClientOptions"),
    "RpcEventListener": ("harnify_coding_agent.modes.rpc.rpc_client", "RpcEventListener"),
    "runRpcMode": ("harnify_coding_agent.modes.rpc.rpc_mode", "runRpcMode"),
    "RpcCommand": ("harnify_coding_agent.modes.rpc.rpc_types", "RpcCommand"),
    "RpcResponse": ("harnify_coding_agent.modes.rpc.rpc_types", "RpcResponse"),
    "RpcSessionState": ("harnify_coding_agent.modes.rpc.rpc_types", "RpcSessionState"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _PUBLIC_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
