"""User-Agent helpers for harnify version checks."""

from __future__ import annotations

import platform
import sys


def get_harnify_user_agent(version: str) -> str:
    runtime = f"python/{platform.python_version()}"
    return f"harnify/{version} ({sys.platform}; {runtime}; {_arch()})"


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine in {"x86_64", "amd64"}:
        return "x64"
    return machine


getHarnifyUserAgent = get_harnify_user_agent

__all__ = ["getHarnifyUserAgent", "get_harnify_user_agent"]
