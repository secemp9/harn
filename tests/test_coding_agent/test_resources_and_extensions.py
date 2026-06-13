from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from harn_agent.types import AgentToolResult
from harn_coding_agent.config import CONFIG_DIR_NAME, ENV_AGENT_DIR, get_themes_dir
from harn_coding_agent.core.extensions.loader import create_extension_runtime, discover_and_load_extensions
from harn_coding_agent.core.extensions.runner import ExtensionRunner
from harn_coding_agent.core.extensions.wrapper import wrap_registered_tools
from harn_coding_agent.core.messages import BashExecutionMessage, bashExecutionToText, convertToLlm
from harn_coding_agent.core.prompt_templates import (
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)
from harn_coding_agent.core.resource_loader import DefaultResourceLoader, load_project_context_files
from harn_coding_agent.core.skills import Skill, format_skills_for_prompt, load_skills
from harn_coding_agent.core.source_info import create_synthetic_source_info
from harn_coding_agent.core.system_prompt import build_system_prompt


def _write_valid_theme(path: Path, *, name: str) -> None:
    payload = json.loads((Path(get_themes_dir()) / "dark.json").read_text(encoding="utf-8"))
    payload["name"] = name
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_core_messages_reexport_harness_behaviour() -> None:
    from harn_coding_agent.core import messages

    message = BashExecutionMessage(
        command="echo hi",
        output="hi",
        exitCode=0,
        cancelled=False,
        truncated=False,
        timestamp=1,
    )

    assert bashExecutionToText(message) == "Ran `echo hi`\n```\nhi\n```"
    converted = convertToLlm([message])
    assert converted[0].role == "user"
    assert converted[0].content[0].text == "Ran `echo hi`\n```\nhi\n```"
    assert messages.__all__ == [
        "BRANCH_SUMMARY_PREFIX",
        "BRANCH_SUMMARY_SUFFIX",
        "BashExecutionMessage",
        "BranchSummaryMessage",
        "COMPACTION_SUMMARY_PREFIX",
        "COMPACTION_SUMMARY_SUFFIX",
        "CompactionSummaryMessage",
        "CustomMessage",
        "bashExecutionToText",
        "convertToLlm",
        "createBranchSummaryMessage",
        "createCompactionSummaryMessage",
        "createCustomMessage",
    ]


def test_prompt_templates_load_defaults_and_expand(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    (agent_dir / "prompts").mkdir(parents=True)
    (cwd / CONFIG_DIR_NAME / "prompts").mkdir(parents=True)
    extra_dir = tmp_path / "extra-prompts"
    extra_dir.mkdir()
    typed_template = tmp_path / "typed.md"

    (agent_dir / "prompts" / "global.md").write_text(
        "---\ndescription: Global template\nargument-hint: <name>\n---\nHello $1 from $ARGUMENTS",
        encoding="utf-8",
    )
    (cwd / CONFIG_DIR_NAME / "prompts" / "project.md").write_text("Project prompt body", encoding="utf-8")
    (extra_dir / "extra.md").write_text("---\n---\nExtra template", encoding="utf-8")
    typed_template.write_text("---\ndescription: 7\nargument-hint: true\n---\nTyped body", encoding="utf-8")

    templates = load_prompt_templates(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "promptPaths": [str(extra_dir), str(typed_template)],
            "includeDefaults": True,
        }
    )

    names = [template.name for template in templates]
    assert names == ["global", "project", "extra", "typed"]
    assert templates[0].argumentHint == "<name>"
    assert templates[0].sourceInfo.scope == "user"
    assert templates[1].sourceInfo.scope == "project"
    assert templates[2].sourceInfo.scope == "temporary"
    typed = next(template for template in templates if template.name == "typed")
    assert typed.description == 7
    assert typed.argumentHint is True
    from harn_coding_agent.core import prompt_templates as prompt_templates_module

    assert prompt_templates_module.__all__ == [
        "LoadPromptTemplatesOptions",
        "PromptTemplate",
        "expandPromptTemplate",
        "loadPromptTemplates",
        "parseCommandArgs",
        "substituteArgs",
    ]
    assert not hasattr(prompt_templates_module, "CONFIG_DIR_NAME")
    assert parse_command_args('alpha "beta gamma" delta') == ["alpha", "beta gamma", "delta"]
    assert substitute_args("one=$1 rest=${@:2} all=$@", ["a", "b", "c"]) == "one=a rest=b c all=a b c"
    assert expand_prompt_template("/global world", templates) == "Hello world from world"


def test_skills_load_with_collisions_and_prompt_formatting(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    (agent_dir / "skills" / "global-skill").mkdir(parents=True)
    (cwd / CONFIG_DIR_NAME / "skills" / "project-skill").mkdir(parents=True)
    extra_dir = tmp_path / "extra-skills"
    extra_dir.mkdir()

    (agent_dir / "skills" / "global-skill" / "SKILL.md").write_text(
        "---\ndescription: Global description\n---\n# global",
        encoding="utf-8",
    )
    (cwd / CONFIG_DIR_NAME / "skills" / "project-skill" / "SKILL.md").write_text(
        "---\nname: project-skill\ndescription: Project description\n---\n# project",
        encoding="utf-8",
    )
    (extra_dir / ".ignore").write_text("ignored.md\n", encoding="utf-8")
    (extra_dir / "collision.md").write_text(
        "---\nname: global-skill\ndescription: Explicit duplicate\n---\n# dup",
        encoding="utf-8",
    )
    (extra_dir / "ignored.md").write_text("---\ndescription: Ignored\n---\n# ignored", encoding="utf-8")

    result = load_skills(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "skillPaths": [str(extra_dir)],
            "includeDefaults": True,
        }
    )

    assert [skill.name for skill in result.skills] == ["global-skill", "project-skill"]
    assert any(diagnostic.type == "collision" for diagnostic in result.diagnostics)
    prompt = format_skills_for_prompt(result.skills)
    assert "<name>global-skill</name>" in prompt
    assert "<location>" in prompt
    assert result.skills[0].sourceInfo.scope == "user"
    assert result.skills[1].sourceInfo.scope == "project"
    from harn_coding_agent.core import skills as skills_module

    assert skills_module.__all__ == [
        "SkillFrontmatter",
        "Skill",
        "LoadSkillsResult",
        "LoadSkillsFromDirOptions",
        "loadSkillsFromDir",
        "formatSkillsForPrompt",
        "LoadSkillsOptions",
        "loadSkills",
    ]


def test_skills_respect_explicit_empty_agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current_dir = tmp_path / "current"
    cwd = tmp_path / "project"
    current_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(current_dir)

    skill_dir = current_dir / "skills" / "local-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: Local description\n---\n# local", encoding="utf-8")

    result = load_skills(
        {
            "cwd": str(cwd),
            "agentDir": "",
            "skillPaths": [],
            "includeDefaults": True,
        }
    )

    assert [skill.name for skill in result.skills] == ["local-skill"]
    assert result.skills[0].sourceInfo.scope == "user"


@pytest.mark.asyncio
async def test_extensions_and_resource_loader_compose_session_start_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "workspace" / "nested"
    cwd.mkdir(parents=True)
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (agent_dir / "extensions").mkdir(parents=True)
    (agent_dir / "AGENTS.md").write_text("global context", encoding="utf-8")
    (cwd.parent / "AGENTS.md").write_text("parent context", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("cwd context", encoding="utf-8")
    (cwd / CONFIG_DIR_NAME).mkdir()
    (cwd / CONFIG_DIR_NAME / "SYSTEM.md").write_text("system body", encoding="utf-8")
    (cwd / CONFIG_DIR_NAME / "APPEND_SYSTEM.md").write_text("append body", encoding="utf-8")

    extension_dir = agent_dir / "extensions" / "demo"
    extension_dir.mkdir()
    skill_dir = agent_dir / "skills" / "extension-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: Extension skill\n---\n# ext", encoding="utf-8")
    prompt_dir = agent_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "summarize.md").write_text("---\ndescription: Prompt\n---\nSummarize $1", encoding="utf-8")

    (extension_dir / "index.py").write_text(
        (
            "from harn_ai.types import TextContent\n"
            "from harn_agent.types import AgentToolResult\n"
            "from harn_coding_agent.core.extensions.types import ToolDefinition\n"
            "async def default(api):\n"
            "    async def execute(tool_call_id, params, signal, on_update, ctx):\n"
            "        return AgentToolResult(\n"
            "            content=[TextContent(text=ctx.cwd)],\n"
            "            details={'seen': params['value']},\n"
            "        )\n"
            "    api.registerTool(\n"
            "        ToolDefinition(\n"
            "            name='demo',\n"
            "            label='Demo',\n"
            "            description='desc',\n"
            "            parameters={'type': 'object'},\n"
            "            execute=execute,\n"
            "        )\n"
            "    )\n"
            "    api.registerCommand('demo-cmd', {'description': 'cmd', 'handler': lambda args, ctx: None})\n"
        ),
        encoding="utf-8",
    )

    discovered = await discover_and_load_extensions([], str(cwd), str(agent_dir))
    assert discovered.errors == []
    assert len(discovered.extensions) == 1
    extension = discovered.extensions[0]
    assert list(extension.tools) == ["demo"]

    runner = ExtensionRunner(
        extensions=[],
        runtime=create_extension_runtime(),
        cwd="runner-cwd",
        sessionManager=None,
        modelRegistry=None,
    )
    wrapped = wrap_registered_tools(list(extension.tools.values()), runner)
    result = await wrapped[0].execute("call-1", {"value": "ok"}, None, None)
    assert isinstance(result, AgentToolResult)
    assert result.content[0].text == "runner-cwd"
    assert result.details == {"seen": "ok"}

    loader = DefaultResourceLoader(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "additionalExtensionPaths": [str(extension_dir)],
        }
    )
    await loader.reload()

    assert [skill.name for skill in loader.getSkills()["skills"]] == ["extension-skill"]
    assert [prompt.name for prompt in loader.getPrompts()["prompts"]] == ["summarize"]
    assert loader.getSkills()["skills"][0].sourceInfo.scope == "user"
    assert loader.getPrompts()["prompts"][0].sourceInfo.scope == "user"
    loaded_extension = loader.getExtensions().extensions[0]
    assert loaded_extension.sourceInfo.scope == "user"
    assert loaded_extension.tools["demo"].sourceInfo.scope == "user"
    assert loaded_extension.commands["demo-cmd"].sourceInfo.scope == "user"
    assert loader.getSystemPrompt() == "system body"
    assert loader.getAppendSystemPrompt() == ["append body"]
    assert [item["content"] for item in loader.getAgentsFiles()["agentsFiles"]] == [
        "global context",
        "parent context",
        "cwd context",
    ]


@pytest.mark.asyncio
async def test_extension_loader_requires_default_export_for_extension_modules(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    extension_dir = agent_dir / "extensions" / "bad"
    extension_dir.mkdir(parents=True)
    (extension_dir / "index.py").write_text(
        "async def extension_factory(api):\n    return None\n",
        encoding="utf-8",
    )

    discovered = await discover_and_load_extensions([], str(cwd), str(agent_dir))

    assert discovered.extensions == []
    assert discovered.errors == [
        {
            "path": str(extension_dir / "index.py"),
            "error": f"Extension does not export a valid factory function: {extension_dir / 'index.py'}",
        }
    ]


@pytest.mark.asyncio
async def test_discover_and_load_extensions_uses_configured_default_agent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "custom-agent"
    extension_dir = agent_dir / "extensions" / "demo"
    extension_dir.mkdir(parents=True)
    (extension_dir / "index.py").write_text(
        "async def default(api):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))

    discovered = await discover_and_load_extensions([], str(cwd))

    assert discovered.errors == []
    assert [extension.path for extension in discovered.extensions] == [str(extension_dir / "index.py")]


@pytest.mark.asyncio
async def test_discovered_extension_paths_preserve_leading_and_trailing_spaces(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    extension_dir = agent_dir / "extensions" / " demo "
    extension_dir.mkdir(parents=True)
    (extension_dir / "index.py").write_text(
        "async def default(api):\n    return None\n",
        encoding="utf-8",
    )

    discovered = await discover_and_load_extensions([], str(cwd), str(agent_dir))

    assert discovered.errors == []
    assert [extension.path for extension in discovered.extensions] == [str(extension_dir / "index.py")]


@pytest.mark.asyncio
async def test_discovered_extension_directory_supports_pyproject_harn_manifest(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    extension_dir = agent_dir / "extensions" / "demo"
    extension_path = extension_dir / "src" / "entry.py"
    extension_path.parent.mkdir(parents=True)
    extension_path.write_text("async def default(api):\n    return None\n", encoding="utf-8")
    (extension_dir / "pyproject.toml").write_text(
        (
            "[project]\n"
            'name = "demo-extension"\n'
            "\n"
            "[tool.harn]\n"
            'extensions = ["src/entry.py"]\n'
        ),
        encoding="utf-8",
    )

    discovered = await discover_and_load_extensions([], str(cwd), str(agent_dir))

    assert discovered.errors == []
    assert [extension.path for extension in discovered.extensions] == [str(extension_path)]


def test_build_system_prompt_uses_context_and_skills(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    skill = Skill(
        name="demo-skill",
        description="Demo skill",
        filePath=str(tmp_path / "skills" / "demo" / "SKILL.md"),
        baseDir=str(tmp_path / "skills" / "demo"),
        sourceInfo=create_synthetic_source_info(
            str(tmp_path / "skills" / "demo" / "SKILL.md"),
            {"source": "local", "scope": "project", "baseDir": str(tmp_path / "skills" / "demo")},
        ),
        disableModelInvocation=False,
    )
    prompt = build_system_prompt(
        {
            "customPrompt": "Base",
            "cwd": str(cwd),
            "selectedTools": ["read", "bash"],
            "appendSystemPrompt": "Appendix",
            "contextFiles": [{"path": "/tmp/AGENTS.md", "content": "Rules"}],
            "skills": [skill],
        }
    )

    assert "Base" in prompt
    assert "Appendix" in prompt
    assert '<project_instructions path="/tmp/AGENTS.md">' in prompt
    assert "<name>demo-skill</name>" in prompt
    assert f"Current date: {dt.date.today().isoformat()}" in prompt
    assert f"Current working directory: {str(cwd)}" in prompt


def test_build_system_prompt_default_surface_and_docs_paths() -> None:
    from harn_coding_agent.config import get_docs_path, get_examples_path, get_readme_path
    from harn_coding_agent.core import system_prompt as system_prompt_module

    assert system_prompt_module.__all__ == ["BuildSystemPromptOptions", "buildSystemPrompt"]

    prompt = build_system_prompt({"cwd": "/tmp/project"})

    assert prompt.startswith(
        "You are an expert coding assistant operating inside harn, a coding agent harness. "
        "You help users by reading files, executing commands, editing code, and writing new files."
    )
    assert f"- Main documentation: {get_readme_path()}" in prompt
    assert f"- Additional docs: {get_docs_path()}" in prompt
    assert f"- Examples: {get_examples_path()} (extensions, custom tools, SDK)" in prompt


def test_load_project_context_files_orders_global_then_ancestors(tmp_path: Path) -> None:
    cwd = tmp_path / "a" / "b" / "c"
    cwd.mkdir(parents=True)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "AGENTS.md").write_text("global", encoding="utf-8")
    (tmp_path / "a" / "AGENTS.md").write_text("a", encoding="utf-8")
    (tmp_path / "a" / "b" / "AGENTS.md").write_text("b", encoding="utf-8")

    files = load_project_context_files({"cwd": str(cwd), "agentDir": str(agent_dir)})
    assert [item["content"] for item in files] == ["global", "a", "b"]


@pytest.mark.asyncio
async def test_resource_loader_exports_and_nullish_prompt_overrides_match_ts(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (cwd / CONFIG_DIR_NAME).mkdir()
    (cwd / CONFIG_DIR_NAME / "SYSTEM.md").write_text("project system", encoding="utf-8")
    (cwd / CONFIG_DIR_NAME / "APPEND_SYSTEM.md").write_text("project append", encoding="utf-8")
    (agent_dir / "SYSTEM.md").write_text("agent system", encoding="utf-8")
    (agent_dir / "APPEND_SYSTEM.md").write_text("agent append", encoding="utf-8")

    loader = DefaultResourceLoader(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "systemPrompt": "",
            "appendSystemPrompt": [],
        }
    )
    await loader.reload()

    from harn_coding_agent.core import resource_loader as resource_loader_module

    assert resource_loader_module.__all__ == [
        "DefaultResourceLoader",
        "DefaultResourceLoaderOptions",
        "ResourceCollision",
        "ResourceDiagnostic",
        "ResourceExtensionPaths",
        "ResourceLoader",
        "loadProjectContextFiles",
    ]
    assert resource_loader_module.loadProjectContextFiles is load_project_context_files
    assert loader.getSystemPrompt() is None
    assert loader.getAppendSystemPrompt() == []


@pytest.mark.asyncio
async def test_resource_loader_supports_inline_extension_factories_dynamic_extension_and_dedupes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    inline_skill_dir = tmp_path / "inline-skills"
    inline_skill_dir.mkdir()
    (inline_skill_dir / "SKILL.md").write_text("---\ndescription: Inline skill\n---\n# inline", encoding="utf-8")

    inline_prompt_dir = tmp_path / "inline-prompts"
    inline_prompt_dir.mkdir()
    (inline_prompt_dir / "inline.md").write_text("---\ndescription: Inline prompt\n---\nPrompt", encoding="utf-8")

    inline_theme_dir = tmp_path / "inline-themes"
    inline_theme_dir.mkdir()
    _write_valid_theme(inline_theme_dir / "inline.json", name="Inline Theme")

    extra_skill_dir = cwd / CONFIG_DIR_NAME / "skills" / "extra-skill"
    extra_skill_dir.mkdir(parents=True)
    (extra_skill_dir / "SKILL.md").write_text("---\ndescription: Extra skill\n---\n# extra", encoding="utf-8")

    dynamic_skill_dir = tmp_path / "dynamic-skill"
    dynamic_skill_dir.mkdir()
    (dynamic_skill_dir / "SKILL.md").write_text("---\ndescription: Dynamic skill\n---\n# dynamic", encoding="utf-8")

    prompt_dir_one = tmp_path / "prompt-one"
    prompt_dir_two = tmp_path / "prompt-two"
    prompt_dir_one.mkdir()
    prompt_dir_two.mkdir()
    (prompt_dir_one / "shared.md").write_text("---\ndescription: one\n---\none", encoding="utf-8")
    (prompt_dir_two / "shared.md").write_text("---\ndescription: two\n---\ntwo", encoding="utf-8")

    theme_dir_one = tmp_path / "theme-one"
    theme_dir_two = tmp_path / "theme-two"
    theme_dir_one.mkdir()
    theme_dir_two.mkdir()
    _write_valid_theme(theme_dir_one / "shared.json", name="Shared Theme")
    _write_valid_theme(theme_dir_two / "shared.json", name="Shared Theme")

    async def inline_factory(_api: object) -> None:
        return None

    loader = DefaultResourceLoader(
        {
            "cwd": str(cwd),
            "agentDir": str(agent_dir),
            "extensionFactories": [inline_factory],
            "additionalPromptTemplatePaths": [str(prompt_dir_one), str(prompt_dir_two)],
            "additionalThemePaths": [str(theme_dir_one), str(theme_dir_two)],
        }
    )
    await loader.reload()

    loader.extendResources(
        {
            "skillPaths": [
                {
                    "path": str(inline_skill_dir),
                    "metadata": {
                        "source": "local",
                        "scope": "temporary",
                        "origin": "top-level",
                        "baseDir": str(inline_skill_dir),
                    },
                }
            ],
            "promptPaths": [
                {
                    "path": str(inline_prompt_dir),
                    "metadata": {
                        "source": "local",
                        "scope": "temporary",
                        "origin": "top-level",
                        "baseDir": str(inline_prompt_dir),
                    },
                }
            ],
            "themePaths": [
                {
                    "path": str(inline_theme_dir),
                    "metadata": {
                        "source": "local",
                        "scope": "temporary",
                        "origin": "top-level",
                        "baseDir": str(inline_theme_dir),
                    },
                }
            ],
        }
    )

    assert [extension.path for extension in loader.getExtensions().extensions] == ["<inline:1>"]
    assert [skill.name for skill in loader.getSkills()["skills"]] == ["extra-skill", "inline-skills"]
    assert [prompt.name for prompt in loader.getPrompts()["prompts"]] == ["shared", "inline"]
    assert [theme.name for theme in loader.getThemes()["themes"]] == ["Shared Theme", "Inline Theme"]
    assert loader.getSkills()["skills"][0].sourceInfo.scope == "project"
    assert loader.getSkills()["skills"][1].sourceInfo.scope == "temporary"
    assert loader.getPrompts()["prompts"][0].sourceInfo.scope == "temporary"
    assert loader.getThemes()["themes"][0].sourceInfo.scope == "temporary"
    assert any(diagnostic.type == "collision" for diagnostic in loader.getPrompts()["diagnostics"])
    assert any(diagnostic.type == "collision" for diagnostic in loader.getThemes()["diagnostics"])

    loader.extendResources(
        {
            "skillPaths": [
                {
                    "path": str(dynamic_skill_dir),
                    "metadata": {
                        "source": "local",
                        "scope": "project",
                        "origin": "top-level",
                        "baseDir": str(dynamic_skill_dir),
                    },
                }
            ]
        }
    )

    assert [skill.name for skill in loader.getSkills()["skills"]] == [
        "extra-skill",
        "inline-skills",
        "dynamic-skill",
    ]
    assert loader.getSkills()["skills"][2].sourceInfo.scope == "project"
