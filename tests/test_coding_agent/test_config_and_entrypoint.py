from __future__ import annotations

import os
import sys
from pathlib import Path

from harnify_coding_agent import cli as cli_package
from harnify_coding_agent import config


def test_config_metadata_defaults_match_package_configuration() -> None:
    assert config.PACKAGE_NAME == "harnify-coding-agent"
    assert config.APP_NAME == "pi"
    assert config.APP_TITLE == "π"
    assert config.VERSION == "0.1.0"
    assert Path(config.get_package_json_path()).name == "pyproject.toml"


def test_detect_install_method_handles_python_layouts() -> None:
    assert config._detect_install_method(
        "/home/test/.local/share/pipx/venvs/harnify-coding-agent/lib/python3.12/site-packages/harnify_coding_agent",
        "/home/test/.local/share/pipx/venvs/harnify-coding-agent/bin/python",
    ) == "pipx"
    assert config._detect_install_method(
        "/home/test/.local/share/uv/tools/harnify-coding-agent/lib/python3.12/site-packages/harnify_coding_agent",
        "/home/test/.local/share/uv/tools/harnify-coding-agent/bin/python",
    ) == "uv-tool"
    assert config._detect_install_method(
        "/usr/lib/python3/dist-packages/harnify_coding_agent",
        "/usr/bin/python3",
    ) == "pip"


def test_detect_install_method_reports_source_checkout_for_repo_tree() -> None:
    assert config.detect_install_method() == "source"


def test_self_update_commands_match_python_install_methods(monkeypatch) -> None:
    monkeypatch.setattr(config, "detect_install_method", lambda: "pipx")
    pipx_command = config.get_self_update_command(config.PACKAGE_NAME)
    assert pipx_command is not None
    assert pipx_command.display == "pipx upgrade harnify-coding-agent"

    monkeypatch.setattr(config, "detect_install_method", lambda: "uv-tool")
    uv_command = config.get_self_update_command(config.PACKAGE_NAME)
    assert uv_command is not None
    assert uv_command.display == "uv tool upgrade harnify-coding-agent"

    monkeypatch.setattr(config, "detect_install_method", lambda: "pip")
    pip_command = config.get_self_update_command(config.PACKAGE_NAME, python_command=["python", "-m", "pip"])
    assert pip_command is not None
    assert pip_command.display == "python -m pip install --upgrade harnify-coding-agent"
    assert config.get_update_instruction(config.PACKAGE_NAME) == (
        f"Run: {sys.executable} -m pip install --upgrade harnify-coding-agent"
    )


def test_self_update_fallback_mentions_source_checkout(monkeypatch) -> None:
    monkeypatch.setattr(config, "detect_install_method", lambda: "source")
    assert config.get_self_update_command(config.PACKAGE_NAME) is None
    instruction = config.get_self_update_unavailable_instruction(config.PACKAGE_NAME)
    assert "source checkout" in instruction
    assert "uv sync" in instruction


def test_cli_package_entrypoint_wraps_async_main(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_invoke(argv: list[str]) -> int:
        seen["argv"] = argv
        seen["env"] = os.environ.get("PI_CODING_AGENT")
        return 17

    monkeypatch.setattr(cli_package, "_invoke_main", fake_invoke)
    monkeypatch.delenv("PI_CODING_AGENT", raising=False)

    assert cli_package.main(["--demo"]) == 17
    assert seen["argv"] == ["--demo"]
    assert seen["env"] == "true"
