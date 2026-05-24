from __future__ import annotations

import json
from pathlib import Path

from harnify_coding_agent.config import APP_NAME
from harnify_coding_agent.migrations import (
    migrate_auth_to_auth_json,
    migrate_commands_to_prompts,
    migrate_sessions_from_agent_root,
    migrate_tools_to_bin,
    run_migrations,
)


def test_migrate_auth_to_auth_json_moves_oauth_and_api_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(tmp_path))
    (tmp_path / "oauth.json").write_text(json.dumps({"github-copilot": {"accessToken": "abc"}}), encoding="utf-8")
    (tmp_path / "settings.json").write_text(
        json.dumps({"apiKeys": {"openai": "sk-test"}, "theme": "dark"}),
        encoding="utf-8",
    )

    migrated = migrate_auth_to_auth_json()

    assert migrated == ["github-copilot", "openai"]
    auth = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    assert auth["github-copilot"]["type"] == "oauth"
    assert auth["openai"] == {"type": "api_key", "key": "sk-test"}
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert "apiKeys" not in settings
    assert (tmp_path / "oauth.json.migrated").exists()


def test_migrate_sessions_from_agent_root_moves_into_encoded_session_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(tmp_path))
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        '{"type":"session","cwd":"/tmp/project"}\n{"type":"user","message":"hi"}\n',
        encoding="utf-8",
    )

    migrate_sessions_from_agent_root()

    moved_files = list((tmp_path / "sessions").rglob("session.jsonl"))
    assert len(moved_files) == 1
    assert moved_files[0].read_text(encoding="utf-8").startswith('{"type":"session"')
    assert not session_file.exists()


def test_migrate_commands_to_prompts_renames_directory(tmp_path: Path) -> None:
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "review.md").write_text("review", encoding="utf-8")

    migrated = migrate_commands_to_prompts(str(tmp_path), "Project")

    assert migrated is True
    assert not commands_dir.exists()
    assert (tmp_path / "prompts" / "review.md").exists()


def test_migrate_tools_to_bin_moves_known_binaries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(tmp_path))
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "rg").write_text("binary", encoding="utf-8")

    migrate_tools_to_bin()

    assert (tmp_path / "bin" / "rg").exists()
    assert not (tools_dir / "rg").exists()


def test_run_migrations_collects_deprecation_warnings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(tmp_path / "agent"))
    agent_dir = tmp_path / "agent"
    (agent_dir / "hooks").mkdir(parents=True)
    project_config = tmp_path / ".pi" / "tools"
    project_config.mkdir(parents=True)
    (project_config / "custom.sh").write_text("echo hi", encoding="utf-8")

    result = run_migrations(str(tmp_path))

    assert result["migratedAuthProviders"] == []
    assert any("hooks/" in warning for warning in result["deprecationWarnings"])
    assert any("custom tools" in warning for warning in result["deprecationWarnings"])
