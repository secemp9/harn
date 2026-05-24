from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from harnify_coding_agent.core.package_manager import DefaultPackageManager
from harnify_coding_agent.core.resource_loader import DefaultResourceLoader
from harnify_coding_agent.core.settings_manager import SettingsManager


def _write_extension(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("async def default(api):\n    return None\n", encoding="utf-8")


def _write_package(package_root: Path, *, skill_name: str, prompt_name: str, theme_name: str) -> None:
    (package_root / "skills" / skill_name).mkdir(parents=True, exist_ok=True)
    (package_root / "prompts").mkdir(parents=True, exist_ok=True)
    (package_root / "themes").mkdir(parents=True, exist_ok=True)
    _write_extension(package_root / "extensions" / "index.py")
    (package_root / "skills" / skill_name / "SKILL.md").write_text(
        f"---\ndescription: {skill_name}\n---\n# {skill_name}",
        encoding="utf-8",
    )
    (package_root / "prompts" / f"{prompt_name}.md").write_text(
        f"---\ndescription: {prompt_name}\n---\n{prompt_name}",
        encoding="utf-8",
    )
    (package_root / "themes" / f"{theme_name}.json").write_text(
        json.dumps({"name": theme_name, "accent": "blue"}),
        encoding="utf-8",
    )
    (package_root / "package.json").write_text(
        json.dumps(
            {
                "pi": {
                    "extensions": ["extensions/index.py"],
                    "skills": [f"skills/{skill_name}"],
                    "prompts": [f"prompts/{prompt_name}.md"],
                    "themes": [f"themes/{theme_name}.json"],
                }
            }
        ),
        encoding="utf-8",
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def _commit_all(repo: Path, message: str) -> None:
    _git("add", ".", cwd=repo)
    _git(
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
        cwd=repo,
    )


def _create_git_package_remote(tmp_path: Path) -> tuple[str, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    _git("init", "--bare", str(remote))
    _git("init", "-b", "main", str(seed))
    _write_package(seed, skill_name="git-skill", prompt_name="git-prompt", theme_name="Git Theme")
    _commit_all(seed, "initial")
    _git("remote", "add", "origin", remote.as_uri(), cwd=seed)
    _git("push", "-u", "origin", "HEAD:main", cwd=seed)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
    return f"git:{remote.as_uri()}", seed


def _update_git_package_remote(seed: Path) -> None:
    (seed / "prompts" / "git-prompt.md").write_text(
        "---\ndescription: git-prompt\n---\nupdated git prompt",
        encoding="utf-8",
    )
    _commit_all(seed, "update")
    _git("push", "origin", "HEAD:main", cwd=seed)


def _write_fake_npm(script_path: Path, registry_path: Path) -> None:
    script_path.write_text(
        textwrap.dedent(
            f"""\
            from __future__ import annotations

            import json
            import os
            import shutil
            import sys
            from pathlib import Path

            REGISTRY_PATH = Path({str(registry_path)!r})


            def load_registry() -> dict[str, dict[str, str]]:
                return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


            def split_spec(spec: str) -> tuple[str, str | None]:
                if spec.startswith("@"):
                    second_at = spec.find("@", 1 + spec[1:].find("/") + 1)
                    if second_at > 0:
                        return spec[:second_at], spec[second_at + 1 :]
                    return spec, None
                if "@" not in spec:
                    return spec, None
                name, version = spec.rsplit("@", 1)
                return name, version


            def get_option(args: list[str], flag: str) -> str | None:
                if flag not in args:
                    return None
                index = args.index(flag)
                if index + 1 >= len(args):
                    return None
                return args[index + 1]


            def write_package(root: Path, name: str, version: str, meta: dict[str, str]) -> None:
                package_root = root / "node_modules" / name
                (package_root / "extensions").mkdir(parents=True, exist_ok=True)
                (package_root / "skills" / meta["skill"]).mkdir(parents=True, exist_ok=True)
                (package_root / "prompts").mkdir(parents=True, exist_ok=True)
                (package_root / "themes").mkdir(parents=True, exist_ok=True)
                (package_root / "extensions" / "index.py").write_text(
                    "async def default(api):\\n    return None\\n",
                    encoding="utf-8",
                )
                (package_root / "skills" / meta["skill"] / "SKILL.md").write_text(
                    f"---\\ndescription: {{meta['skill']}}\\n---\\n# {{meta['skill']}}",
                    encoding="utf-8",
                )
                (package_root / "prompts" / f"{{meta['prompt']}}.md").write_text(
                    f"---\\ndescription: {{meta['prompt']}}\\n---\\n{{meta['prompt']}} {{version}}",
                    encoding="utf-8",
                )
                (package_root / "themes" / f"{{meta['theme']}}.json").write_text(
                    json.dumps({{"name": meta["theme"], "accent": "green"}}),
                    encoding="utf-8",
                )
                (package_root / "package.json").write_text(
                    json.dumps(
                        {{
                            "name": name,
                            "version": version,
                            "pi": {{
                                "extensions": ["extensions/index.py"],
                                "skills": [f"skills/{{meta['skill']}}"],
                                "prompts": [f"prompts/{{meta['prompt']}}.md"],
                                "themes": [f"themes/{{meta['theme']}}.json"],
                            }},
                        }}
                    ),
                    encoding="utf-8",
                )


            def main() -> int:
                args = sys.argv[1:]
                if not args:
                    return 1
                command = args[0]
                registry = load_registry()
                if command == "view":
                    name = args[1]
                    print(json.dumps(registry[name]["latest"]))
                    return 0
                if command == "root" and args[1:] == ["-g"]:
                    root = Path(__file__).parent / "fake-global" / "node_modules"
                    root.mkdir(parents=True, exist_ok=True)
                    print(str(root))
                    return 0
                if command == "list":
                    print("[]")
                    return 0
                if command == "install":
                    prefix = get_option(args, "--prefix") or get_option(args, "--cwd")
                    if prefix is None:
                        raise SystemExit("missing install root")
                    root = Path(prefix)
                    specs = [arg for arg in args[1:] if not arg.startswith("-") and arg not in {{prefix}}]
                    for spec in specs:
                        name, version = split_spec(spec)
                        meta = registry[name]
                        resolved_version = meta["latest"] if version in (None, "latest") else version
                        write_package(root, name, resolved_version, meta)
                    return 0
                if command == "uninstall":
                    name = args[1]
                    prefix = get_option(args, "--prefix") or get_option(args, "--cwd")
                    if prefix is None:
                        raise SystemExit("missing uninstall root")
                    shutil.rmtree(Path(prefix) / "node_modules" / name, ignore_errors=True)
                    return 0
                raise SystemExit(f"unsupported fake npm command: {{args}}")


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ),
        encoding="utf-8",
    )


def _create_fake_npm_harness(tmp_path: Path) -> tuple[list[str], Path]:
    registry_path = tmp_path / "npm-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "pi-fake": {
                    "latest": "1.0.0",
                    "skill": "npm-skill",
                    "prompt": "npm-prompt",
                    "theme": "NPM Theme",
                }
            }
        ),
        encoding="utf-8",
    )
    script_path = tmp_path / "fake_npm.py"
    _write_fake_npm(script_path, registry_path)
    return [sys.executable, str(script_path)], registry_path


def test_package_manager_adds_removes_and_lists_local_sources(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    local_source = tmp_path / "extra-package"
    local_source.mkdir()

    settings_manager = SettingsManager.inMemory()
    manager = DefaultPackageManager(
        {"cwd": str(cwd), "agentDir": str(agent_dir), "settingsManager": settings_manager}
    )

    assert manager.addSourceToSettings(str(local_source)) is True
    assert manager.addSourceToSettings(str(local_source)) is False

    configured = manager.listConfiguredPackages()
    assert len(configured) == 1
    assert configured[0].scope == "user"
    assert configured[0].installedPath == str(local_source)

    assert manager.removeSourceFromSettings(str(local_source)) is True
    assert manager.removeSourceFromSettings(str(local_source)) is False
    assert manager.listConfiguredPackages() == []


@pytest.mark.asyncio
async def test_resource_loader_loads_project_and_cli_package_resources(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    project_package = cwd / ".harnify" / "packages" / "project-addon"
    cli_package = tmp_path / "cli-addon"
    _write_package(
        project_package,
        skill_name="project-skill",
        prompt_name="project-prompt",
        theme_name="Project Theme",
    )
    _write_package(
        cli_package,
        skill_name="cli-skill",
        prompt_name="cli-prompt",
        theme_name="CLI Theme",
    )

    settings_manager = SettingsManager.inMemory()
    settings_manager.setProjectPackages(["packages/project-addon"])

    loader = DefaultResourceLoader(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "settingsManager": settings_manager,
            "additionalExtensionPaths": [str(cli_package)],
        }
    )
    await loader.reload()

    skill_scopes = {skill.name: skill.sourceInfo.scope for skill in loader.getSkills()["skills"]}
    prompt_scopes = {prompt.name: prompt.sourceInfo.scope for prompt in loader.getPrompts()["prompts"]}
    theme_scopes = {theme.name: theme.sourceInfo.scope for theme in loader.getThemes()["themes"]}
    extension_scopes = {
        Path(extension.path).parent.parent.name: extension.sourceInfo.scope
        for extension in loader.getExtensions().extensions
    }

    assert skill_scopes["project-skill"] == "project"
    assert skill_scopes["cli-skill"] == "temporary"
    assert prompt_scopes["project-prompt"] == "project"
    assert prompt_scopes["cli-prompt"] == "temporary"
    assert theme_scopes["Project Theme"] == "project"
    assert theme_scopes["CLI Theme"] == "temporary"
    assert extension_scopes["project-addon"] == "project"
    assert extension_scopes["cli-addon"] == "temporary"


@pytest.mark.asyncio
async def test_package_manager_installs_updates_and_removes_git_sources(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    source, seed = _create_git_package_remote(tmp_path)

    settings_manager = SettingsManager.inMemory()
    settings_manager.setPackages([source])
    manager = DefaultPackageManager(
        {"cwd": str(cwd), "agentDir": str(agent_dir), "settingsManager": settings_manager}
    )

    resolved = await manager.resolve()

    installed_path = manager.getInstalledPath(source, "user")
    assert installed_path is not None
    assert Path(installed_path).exists()
    assert any(entry.path.endswith("extensions/index.py") for entry in resolved.extensions)
    assert any(entry.path.endswith("skills/git-skill/SKILL.md") for entry in resolved.skills)
    assert any(entry.path.endswith("prompts/git-prompt.md") for entry in resolved.prompts)
    assert manager.listConfiguredPackages()[0].installedPath == installed_path
    assert await manager.checkForAvailableUpdates() == []

    _update_git_package_remote(seed)
    updates = await manager.checkForAvailableUpdates()
    assert [(entry.source, entry.type, entry.scope) for entry in updates] == [(source, "git", "user")]

    await manager.update(source)
    assert "updated git prompt" in Path(installed_path, "prompts", "git-prompt.md").read_text(encoding="utf-8")

    await manager.remove(source)
    assert not Path(installed_path).exists()


@pytest.mark.asyncio
async def test_package_manager_resolves_updates_and_removes_npm_sources(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    npm_command, registry_path = _create_fake_npm_harness(tmp_path)

    settings_manager = SettingsManager.inMemory()
    settings_manager.setNpmCommand(npm_command)
    settings_manager.setPackages(["npm:pi-fake"])
    manager = DefaultPackageManager(
        {"cwd": str(cwd), "agentDir": str(agent_dir), "settingsManager": settings_manager}
    )

    resolved = await manager.resolve()
    installed_path = manager.getInstalledPath("npm:pi-fake", "user")

    assert installed_path is not None
    assert Path(installed_path, "package.json").exists()
    assert any(entry.path.endswith("skills/npm-skill/SKILL.md") for entry in resolved.skills)
    assert any(entry.path.endswith("prompts/npm-prompt.md") for entry in resolved.prompts)
    assert manager.listConfiguredPackages()[0].installedPath == installed_path
    assert Path(installed_path, "package.json").read_text(encoding="utf-8")
    assert await manager.checkForAvailableUpdates() == []

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["pi-fake"]["latest"] = "2.0.0"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    updates = await manager.checkForAvailableUpdates()
    assert [(entry.source, entry.type, entry.scope) for entry in updates] == [("npm:pi-fake", "npm", "user")]

    await manager.update("npm:pi-fake")
    installed_payload = json.loads(Path(installed_path, "package.json").read_text(encoding="utf-8"))
    assert installed_payload["version"] == "2.0.0"

    pinned_settings = SettingsManager.inMemory()
    pinned_settings.setNpmCommand(npm_command)
    pinned_settings.setPackages(["npm:pi-fake@1.0.0"])
    pinned_manager = DefaultPackageManager(
        {"cwd": str(cwd), "agentDir": str(agent_dir), "settingsManager": pinned_settings}
    )
    await pinned_manager.resolve()
    pinned_installed_path = pinned_manager.getInstalledPath("npm:pi-fake@1.0.0", "user")
    assert pinned_installed_path == installed_path
    pinned_payload = json.loads(Path(installed_path, "package.json").read_text(encoding="utf-8"))
    assert pinned_payload["version"] == "1.0.0"
    assert await pinned_manager.checkForAvailableUpdates() == []

    await pinned_manager.remove("npm:pi-fake@1.0.0")
    assert not Path(installed_path).exists()
