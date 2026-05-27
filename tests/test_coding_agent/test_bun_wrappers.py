from __future__ import annotations

import importlib
import sys

from harnify_coding_agent.bun import cli as bun_cli
from harnify_coding_agent.bun import register_bedrock, restore_sandbox_env


def test_restore_sandbox_env_recovers_proc_environ_when_environment_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(restore_sandbox_env, "_read_proc_self_environ", lambda: "ALPHA=1\0BETA=two\0")
    monkeypatch.setattr(restore_sandbox_env.os, "environ", {})

    restore_sandbox_env.restore_sandbox_env()

    assert restore_sandbox_env.os.environ == {"ALPHA": "1", "BETA": "two"}


def test_restore_sandbox_env_leaves_existing_environment_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(restore_sandbox_env.os, "environ", {"KEEP": "value"})
    monkeypatch.setattr(restore_sandbox_env, "_read_proc_self_environ", lambda: "ALPHA=1\0")

    restore_sandbox_env.restore_sandbox_env()

    assert restore_sandbox_env.os.environ == {"KEEP": "value"}


def test_register_bedrock_forwards_module_override(monkeypatch) -> None:
    seen: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(register_bedrock, "bedrockProviderModule", sentinel)
    monkeypatch.setattr(register_bedrock, "setBedrockProviderModule", lambda module: seen.setdefault("module", module))

    register_bedrock.register_bedrock()

    assert seen["module"] is sentinel


def test_register_bedrock_module_exports_match_ts_surface() -> None:
    assert register_bedrock.__all__ == []


def test_bun_cli_main_runs_wrapper_steps_before_delegating(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_cli_main(argv: list[str] | None) -> int:
        seen["argv"] = argv
        return 23

    monkeypatch.setattr(bun_cli, "_set_process_title", lambda title: seen.setdefault("title", title))
    monkeypatch.setattr(bun_cli, "_suppress_runtime_warnings", lambda: seen.setdefault("warnings", True))
    monkeypatch.setattr(bun_cli, "restore_sandbox_env", lambda: seen.setdefault("restored", True))
    monkeypatch.setattr(bun_cli, "_import_register_bedrock_module", lambda: seen.setdefault("registered", True))
    monkeypatch.setattr(bun_cli, "_load_cli_main", lambda: fake_cli_main)
    assert bun_cli.main(["--demo"]) == 23
    assert seen == {
        "title": "harnify",
        "warnings": True,
        "restored": True,
        "registered": True,
        "argv": ["--demo"],
    }


def test_bun_cli_entrypoint_uses_module_name_guard(monkeypatch) -> None:
    monkeypatch.setattr(bun_cli, "_load_cli_main", lambda: lambda argv: 0)
    monkeypatch.setattr(bun_cli, "_import_register_bedrock_module", lambda: None)
    monkeypatch.setattr(bun_cli, "_set_process_title", lambda title: None)
    monkeypatch.setattr(bun_cli, "_suppress_runtime_warnings", lambda: None)
    monkeypatch.setattr(bun_cli, "restore_sandbox_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["prog"])

    assert bun_cli.main() == 0


def test_restore_sandbox_env_module_exports_match_ts_surface() -> None:
    assert restore_sandbox_env.__all__ == ["restoreSandboxEnv"]
