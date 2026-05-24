from __future__ import annotations

import json
from pathlib import Path

import pytest
from harnify_coding_agent.core.http_dispatcher import DEFAULT_HTTP_IDLE_TIMEOUT_MS
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
        json.dumps({"theme": "dark", "packages": ["npm:pi-mcp-adapter"]}),
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
    assert not (project_dir / ".pi").exists()
    assert manager.getTheme() == "dark"

    manager.setProjectPackages([{"source": "npm:test-pkg"}])
    await manager.flush()

    assert (project_dir / ".pi").exists()
    assert _read_json(project_dir / ".pi" / "settings.json")["packages"] == [{"source": "npm:test-pkg"}]


@pytest.mark.asyncio
async def test_http_idle_timeout_defaults_and_project_override(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir()
    project_dir.mkdir()

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    assert manager.getHttpIdleTimeoutMs() == DEFAULT_HTTP_IDLE_TIMEOUT_MS

    (agent_dir / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": 300000}), encoding="utf-8")
    (project_dir / ".pi").mkdir()
    (project_dir / ".pi" / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": 0}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    assert manager.getHttpIdleTimeoutMs() == 0

    (project_dir / ".pi" / "settings.json").unlink()
    (agent_dir / "settings.json").write_text(json.dumps({"httpIdleTimeoutMs": -1}), encoding="utf-8")
    manager = SettingsManager.create(str(project_dir), str(agent_dir))
    with pytest.raises(ValueError, match="Invalid httpIdleTimeoutMs setting"):
        manager.getHttpIdleTimeoutMs()
