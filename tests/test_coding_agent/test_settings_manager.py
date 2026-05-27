from __future__ import annotations

import json
from pathlib import Path

import pytest
from harnify_coding_agent.core.http_dispatcher import DEFAULT_HTTP_IDLE_TIMEOUT_MS
from harnify_coding_agent.core import settings_manager as settings_manager_module
from harnify_coding_agent.core.settings_manager import SettingsManager


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_preserves_external_array_changes_on_unrelated_global_save(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    project_dir.mkdir()
    settings_path = agent_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"theme": "dark", "packages": ["npm:harnify-mcp-adapter"]}),
        encoding="utf-8",
    )

    manager = SettingsManager.create(str(project_dir), str(agent_dir))

    current = _read_json(settings_path)
    current["packages"] = []
    settings_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    manager.setTheme("light")
    await manager.flush()

    saved = _read_json(settings_path)
    assert saved["packages"] == []
    assert saved["theme"] == "light"


def test_migrates_legacy_settings_shapes() -> None:
    manager = SettingsManager.inMemory(
        {
            "queueMode": "all",
            "websockets": True,
            "skills": {
                "enableSkillCommands": False,
                "customDirectories": ["./skills"],
            },
            "retry": {"maxDelayMs": 1234},
        }
    )

    assert manager.getSteeringMode() == "all"
    assert manager.getTransport() == "websocket"
    assert manager.getEnableSkillCommands() is False
    assert manager.getSkillPaths() == ["./skills"]
    assert manager.getProviderRetrySettings()["maxRetryDelayMs"] == 1234


@pytest.mark.asyncio
async def test_project_settings_dir_created_only_on_write(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    project_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    assert not (project_dir / ".harnify").exists()
    assert manager.getTheme() == "dark"

    manager.setProjectPackages([{"source": "npm:test-pkg"}])
    await manager.flush()

    assert (project_dir / ".harnify").exists()
    assert _read_json(project_dir / ".harnify" / "settings.json")["packages"] == [{"source": "npm:test-pkg"}]


@pytest.mark.asyncio
async def test_http_idle_timeout_defaults_and_project_override(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    project_dir.mkdir()

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    assert manager.getHttpIdleTimeoutMs() == DEFAULT_HTTP_IDLE_TIMEOUT_MS

    (agent_dir / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": 300000}), encoding="utf-8")
    (project_dir / ".harnify").mkdir()
    (project_dir / ".harnify" / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": 0}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    assert manager.getHttpIdleTimeoutMs() == 0

    (project_dir / ".harnify" / "settings.json").unlink()
    (agent_dir / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": -1}), encoding="utf-8")
    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    with pytest.raises(Exception, match="Invalid httpIdleTimeoutMs setting"):
        manager.getHttpIdleTimeoutMs()


def test_settings_manager_exports_match_ts_surface() -> None:
    assert settings_manager_module.__all__ == [
        "CompactionSettings",
        "BranchSummarySettings",
        "ProviderRetrySettings",
        "RetrySettings",
        "TerminalSettings",
        "ImageSettings",
        "ThinkingBudgetsSettings",
        "MarkdownSettings",
        "WarningSettings",
        "TransportSetting",
        "PackageSource",
        "Settings",
        "SettingsScope",
        "SettingsStorage",
        "SettingsError",
        "FileSettingsStorage",
        "InMemorySettingsStorage",
        "SettingsManager",
    ]


def test_settings_manager_nullish_getters_match_ts_runtime() -> None:
    manager = SettingsManager.inMemory(
        {
            "transport": "",
            "doubleEscapeAction": "",
            "collapseChangelog": "",
            "retry": {"enabled": None},
            "terminal": {"showImages": None},
            "images": {"autoResize": None},
            "enableInstallTelemetry": None,
        }
    )

    assert manager.getTransport() == ""
    assert manager.getDoubleEscapeAction() == ""
    assert manager.getCollapseChangelog() == ""
    assert manager.getRetryEnabled() is True
    assert manager.getShowImages() is True
    assert manager.getImageAutoResize() is True
    assert manager.getEnableInstallTelemetry() is True


def test_settings_manager_preserves_ts_live_list_reference_semantics() -> None:
    manager = SettingsManager.inMemory()

    extension_paths = ["./ext-a"]
    manager.setExtensionPaths(extension_paths)
    extension_paths.append("./ext-b")
    assert manager.getExtensionPaths() == ["./ext-a", "./ext-b"]

    enabled_models = ["openai/*"]
    manager.setEnabledModels(enabled_models)
    enabled_models.append("anthropic/*")
    assert manager.getEnabledModels() == ["openai/*", "anthropic/*"]

    returned_models = manager.getEnabledModels()
    assert returned_models is enabled_models
    returned_models.append("google/*")
    assert manager.getEnabledModels() == ["openai/*", "anthropic/*", "google/*"]


@pytest.mark.asyncio
async def test_settings_manager_flush_waits_for_queued_write(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    project_dir.mkdir()

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    settings_path = agent_dir / "settings.json"

    manager.setTheme("light")
    assert not settings_path.exists()

    await manager.flush()

    assert _read_json(settings_path)["theme"] == "light"


@pytest.mark.asyncio
async def test_settings_manager_create_respects_explicit_empty_agent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_dir = tmp_path / "current"
    project_dir = tmp_path / "project"
    current_dir.mkdir()
    project_dir.mkdir()
    monkeypatch.chdir(current_dir)

    manager = SettingsManager.create(str(project_dir), "")
    manager.setTheme("light")
    await manager.flush()

    assert _read_json(current_dir / "settings.json")["theme"] == "light"
