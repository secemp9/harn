from __future__ import annotations

import subprocess

import pytest

from harnify_coding_agent.core import resolve_config_value as module
from harnify_coding_agent.utils.shell import ShellConfig


def setup_function() -> None:
    module.clearConfigValueCache()


def test_resolve_config_value_exports_match_ts() -> None:
    assert module.__all__ == [
        "clearConfigValueCache",
        "resolveConfigValue",
        "resolveConfigValueOrThrow",
        "resolveConfigValueUncached",
        "resolveHeaders",
        "resolveHeadersOrThrow",
    ]


def test_windows_configured_shell_falls_back_only_for_missing_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "get_shell_config", lambda: ShellConfig(shell="missing.exe", args=["-c"]))

    def missing_shell(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError()

    monkeypatch.setattr(module.subprocess, "run", missing_shell)
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_execute_with_default_shell",
        lambda command: fallback_calls.append(command) or "fallback-value",
    )

    assert module.resolveConfigValueUncached("!echo hi") == "fallback-value"
    assert fallback_calls == ["echo hi"]


def test_windows_configured_shell_does_not_fallback_after_non_enoent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "get_shell_config", lambda: ShellConfig(shell="bash.exe", args=["-c"]))

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="bash.exe", timeout=10)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_execute_with_default_shell",
        lambda command: fallback_calls.append(command) or "fallback-value",
    )

    assert module.resolveConfigValueUncached("!echo hi") is None
    assert fallback_calls == []


def test_resolve_config_value_or_throw_uses_plain_exception() -> None:
    with pytest.raises(Exception) as exc_info:
        module.resolveConfigValueOrThrow("!exit 1", "demo value")

    assert exc_info.value.__class__ is Exception
    assert str(exc_info.value) == "Failed to resolve demo value from shell command: exit 1"
