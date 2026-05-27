from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import harnify_agent.harness.types as harness_types
from harnify_agent.harness.messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    bash_execution_to_text,
    convert_to_llm,
)
from harnify_agent.harness.prompt_templates import (
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    parse_command_args,
)
from harnify_agent.harness.session.session import Session
from harnify_agent.harness.skills import format_skill_invocation, load_skills, load_sourced_skills
from harnify_agent.harness.system_prompt import format_skills_for_system_prompt
from harnify_agent.harness.types import (
    AgentHarnessError,
    FileError,
    FileInfo,
    PromptTemplate,
    Skill,
    err,
    get_or_throw,
    get_or_undefined,
    ok,
    to_error,
)


class PathExecutionEnv:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)

    async def createDir(self, path: str, options: dict[str, Any] | None = None):
        try:
            self._resolve(path).mkdir(parents=(options or {}).get("recursive", True), exist_ok=True)
            return ok(None)
        except OSError as error:
            return err(self._to_file_error(error, path))

    async def writeFile(self, path: str, content: str | bytes, abortSignal: Any | None = None):
        try:
            resolved = self._resolve(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                resolved.write_bytes(content)
            else:
                resolved.write_text(content, encoding="utf-8")
            return ok(None)
        except OSError as error:
            return err(self._to_file_error(error, path))

    async def readTextFile(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            return ok(resolved.read_text(encoding="utf-8"))
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def fileInfo(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            stat_result = os.lstat(resolved)
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))
        kind = "symlink" if resolved.is_symlink() else "directory" if resolved.is_dir() else "file"
        return ok(
            FileInfo(
                name=resolved.name,
                path=str(resolved),
                kind=kind,
                size=stat_result.st_size,
                mtimeMs=stat_result.st_mtime * 1000,
            )
        )

    async def listDir(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            infos: list[FileInfo] = []
            for entry in resolved.iterdir():
                stat_result = os.lstat(entry)
                kind = "symlink" if entry.is_symlink() else "directory" if entry.is_dir() else "file"
                infos.append(
                    FileInfo(
                        name=entry.name,
                        path=str(entry),
                        kind=kind,
                        size=stat_result.st_size,
                        mtimeMs=stat_result.st_mtime * 1000,
                    )
                )
            return ok(infos)
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def canonicalPath(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            return ok(str(resolved.resolve(strict=True)))
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw
        return Path(os.path.abspath(os.path.join(self.cwd, path)))

    def _to_file_error(self, error: OSError, path: str) -> FileError:
        if error.errno == 2:
            return FileError("not_found", str(error), path)
        if error.errno in {13, 1}:
            return FileError("permission_denied", str(error), path)
        return FileError("unknown", str(error), path)


def test_harness_result_helpers() -> None:
    success = ok({"value": 1})
    failure = err(ValueError("boom"))

    assert get_or_throw(success) == {"value": 1}
    assert get_or_undefined(success) == {"value": 1}
    assert get_or_undefined(failure) is None
    assert str(to_error("plain error")) == "plain error"
    assert str(to_error({"a": 1})) == '{"a": 1}'

    with pytest.raises(ValueError, match="boom"):
        get_or_throw(failure)

    assert FileError("unknown", "x").name == "FileError"
    assert AgentHarnessError("unknown", "x").name == "AgentHarnessError"
    assert harness_types.Session is Session
    assert harness_types.AgentHarness.__name__ == "AgentHarness"


def test_bash_execution_to_text_and_convert_to_llm() -> None:
    bash_message = BashExecutionMessage(
        command="pytest -q",
        output="2 passed",
        exitCode=1,
        cancelled=False,
        truncated=True,
        fullOutputPath="/tmp/full.log",
        timestamp=10,
    )
    custom_message = CustomMessage(
        customType="note",
        content="Remember this.",
        display=True,
        details={"a": 1},
        timestamp=11,
    )
    branch_message = BranchSummaryMessage(summary="branch summary", fromId="branch-1", timestamp=12)
    compaction_message = CompactionSummaryMessage(summary="compact summary", tokensBefore=42, timestamp=13)

    rendered = bash_execution_to_text(bash_message)
    assert rendered == (
        "Ran `pytest -q`\n```\n2 passed\n```\n\nCommand exited with code 1\n\n"
        "[Output truncated. Full output: /tmp/full.log]"
    )

    converted = convert_to_llm(
        [
            bash_message,
            custom_message,
            branch_message,
            compaction_message,
            {"role": "user", "content": "hello", "timestamp": 14},
        ]
    )

    assert [message.role for message in converted] == ["user", "user", "user", "user", "user"]
    assert converted[0].content[0].text == rendered
    assert converted[1].content[0].text == "Remember this."
    assert converted[2].content[0].text == BRANCH_SUMMARY_PREFIX + "branch summary" + BRANCH_SUMMARY_SUFFIX
    assert converted[3].content[0].text == COMPACTION_SUMMARY_PREFIX + "compact summary" + COMPACTION_SUMMARY_SUFFIX
    assert converted[4].content == "hello"


@pytest.mark.asyncio
async def test_load_prompt_templates_and_format_invocation(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.createDir("a/nested", {"recursive": True})
    await env.createDir("b", {"recursive": True})
    await env.writeFile("a/one.md", "---\ndescription: One template\n---\nHello $1")
    await env.writeFile("a/nested/ignored.md", "Ignored")
    await env.writeFile("b/two.md", "First line description\nBody")

    result = await load_prompt_templates(env, ["a", "b"])

    assert result.diagnostics == []
    assert result.promptTemplates == [
        PromptTemplate(name="one", description="One template", content="Hello $1"),
        PromptTemplate(name="two", description="First line description", content="First line description\nBody"),
    ]
    assert parse_command_args("""one "two three" 'four five'""") == ["one", "two three", "four five"]
    assert (
        format_prompt_template_invocation(
            PromptTemplate(name="review", content="$1 ${@:2} $ARGUMENTS"),
            ["hello world", "test"],
        )
        == "hello world test hello world test"
    )


@pytest.mark.asyncio
async def test_load_sourced_prompt_templates_and_prompt_diagnostics(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.writeFile("broken.md", "---\ndescription: [unterminated\n---\nBody")

    result = await load_sourced_prompt_templates(env, [{"path": "broken.md", "source": {"type": "user"}}])

    assert result.promptTemplates == []
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert isinstance(diagnostic, type(result.diagnostics[0]))
    assert diagnostic.code == "parse_failed"
    assert diagnostic.source == {"type": "user"}
    assert diagnostic.path == str(tmp_path / "broken.md")


@pytest.mark.asyncio
async def test_load_prompt_templates_handles_scalar_frontmatter_and_non_string_description(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.writeFile("scalar.md", "---\n[]\n---\nFirst line\nBody")
    await env.writeFile("typed.md", "---\ndescription: 123\n---\nActual first line\nBody")

    result = await load_prompt_templates(env, ["scalar.md", "typed.md"])

    assert result.diagnostics == []
    assert result.promptTemplates == [
        PromptTemplate(name="scalar", description="First line", content="First line\nBody"),
        PromptTemplate(name="typed", description="Actual first line", content="Actual first line\nBody"),
    ]


@pytest.mark.asyncio
async def test_load_prompt_templates_supports_symlinked_files(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.writeFile("target.md", "---\ndescription: Target\n---\nTarget body")
    (tmp_path / "link.md").symlink_to(tmp_path / "target.md")

    result = await load_prompt_templates(env, ["target.md", "link.md"])

    assert result.promptTemplates == [
        PromptTemplate(name="target", description="Target", content="Target body"),
        PromptTemplate(name="link", description="Target", content="Target body"),
    ]


@pytest.mark.asyncio
async def test_load_skills_and_sourced_diagnostics(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.createDir(".agents/skills/example", {"recursive": True})
    await env.writeFile(
        ".agents/skills/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n",
    )
    await env.createDir("user/broken", {"recursive": True})
    await env.writeFile("user/broken/SKILL.md", "---\nname: broken\n---\nMissing description.")

    loaded = await load_skills(env, ".agents/skills")
    sourced = await load_sourced_skills(env, [{"path": "user", "source": {"type": "user"}}])

    assert loaded.diagnostics == []
    assert loaded.skills == [
        Skill(
            name="example",
            description="Example skill",
            content="Use this skill.",
            filePath=str(tmp_path / ".agents/skills/example/SKILL.md"),
            disableModelInvocation=True,
        )
    ]
    assert sourced.skills == []
    assert len(sourced.diagnostics) == 1
    assert sourced.diagnostics[0].code == "invalid_metadata"
    assert sourced.diagnostics[0].message == "description is required"
    assert sourced.diagnostics[0].path == str(tmp_path / "user/broken/SKILL.md")
    assert sourced.diagnostics[0].source == {"type": "user"}


@pytest.mark.asyncio
async def test_load_skills_supports_symlinked_directories_and_root_markdown(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.createDir("actual/example", {"recursive": True})
    await env.writeFile(
        "actual/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )
    (tmp_path / "skills-link").symlink_to(tmp_path / "actual")

    await env.createDir("skills/nested", {"recursive": True})
    await env.writeFile("skills/root.md", "---\ndescription: Root skill\n---\nRoot content")
    await env.writeFile("skills/nested/ignored.md", "---\ndescription: Ignored\n---\nIgnored content")

    linked = await load_skills(env, "skills-link")
    root_loaded = await load_skills(env, "skills")

    assert [skill.name for skill in linked.skills] == ["example"]
    assert linked.skills[0].filePath == str(tmp_path / "skills-link/example/SKILL.md")
    assert [skill.name for skill in root_loaded.skills] == ["skills"]
    assert root_loaded.skills[0].content == "Root content"


@pytest.mark.asyncio
async def test_load_skills_applies_ignore_files(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.createDir("skills/nested/example", {"recursive": True})
    await env.writeFile("skills/.gitignore", "nested/\n")
    await env.writeFile(
        "skills/nested/example/SKILL.md",
        "---\nname: example\ndescription: Example skill\n---\nUse this skill.",
    )

    result = await load_skills(env, "skills")

    assert result.skills == []
    assert result.diagnostics == []


@pytest.mark.asyncio
async def test_load_skills_handles_scalar_frontmatter_and_ascii_only_names(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    await env.createDir("skills/example", {"recursive": True})
    await env.createDir("skills/typed", {"recursive": True})
    await env.createDir("skills/caf\xe9", {"recursive": True})
    await env.writeFile("skills/example/SKILL.md", "---\n[]\n---\nBody")
    await env.writeFile("skills/typed/SKILL.md", "---\nname: 123\ndescription: 456\n---\nBody")
    await env.writeFile(
        "skills/caf\xe9/SKILL.md",
        "---\nname: caf\xe9\ndescription: Accent name\n---\nBody",
    )

    result = await load_skills(env, "skills")

    assert result.skills == [
        Skill(
            name="caf\xe9",
            description="Accent name",
            content="Body",
            filePath=str(tmp_path / "skills/caf\xe9/SKILL.md"),
            disableModelInvocation=False,
        )
    ]
    assert sorted(
        (diagnostic.code, diagnostic.message, diagnostic.path) for diagnostic in result.diagnostics
    ) == sorted(
        [
            (
                "invalid_metadata",
                "description is required",
                str(tmp_path / "skills/example/SKILL.md"),
            ),
            (
                "invalid_metadata",
                "description is required",
                str(tmp_path / "skills/typed/SKILL.md"),
            ),
            (
                "invalid_metadata",
                "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)",
                str(tmp_path / "skills/caf\xe9/SKILL.md"),
            ),
        ]
    )


def test_format_skill_invocation_and_system_prompt() -> None:
    visible_skill = Skill(
        name="visible",
        description="Use <this> & that",
        content="visible content",
        filePath="/skills/visible/SKILL.md",
    )
    second_skill = Skill(
        name="second",
        description="Second skill",
        content="second content",
        filePath="/skills/second/SKILL.md",
    )
    disabled_skill = Skill(
        name="hidden",
        description="Hidden",
        content="hidden content",
        filePath="/skills/hidden/SKILL.md",
        disableModelInvocation=True,
    )

    assert format_skill_invocation(
        Skill(
            name="inspect",
            description="Inspect things",
            content="Use inspection tools.",
            filePath="/project/.harnify/skills/inspect/SKILL.md",
        ),
        "Check errors.",
    ) == (
        '<skill name="inspect" location="/project/.harnify/skills/inspect/SKILL.md">\n'
        "References are relative to /project/.harnify/skills/inspect.\n\n"
        "Use inspection tools.\n"
        "</skill>\n\n"
        "Check errors."
    )

    assert format_skills_for_system_prompt([visible_skill, disabled_skill, second_skill]) == (
        "The following skills provide specialized instructions for specific tasks.\n"
        "Read the full skill file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n"
        "\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>visible</name>\n"
        "    <description>Use &lt;this&gt; &amp; that</description>\n"
        "    <location>/skills/visible/SKILL.md</location>\n"
        "  </skill>\n"
        "  <skill>\n"
        "    <name>second</name>\n"
        "    <description>Second skill</description>\n"
        "    <location>/skills/second/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>"
    )
    assert format_skills_for_system_prompt([disabled_skill]) == ""
    assert "<name>a&amp;b</name>" in format_skills_for_system_prompt(
        [
            Skill(
                name="a&b",
                description="Quote \"double\" and 'single'",
                content="content",
                filePath='/skills/<bad>&"quote"/SKILL.md',
            )
        ]
    )
