"""System prompt construction for coding-agent sessions."""

from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
from typing import TypedDict

from harnify_coding_agent.core.skills import Skill, format_skills_for_prompt


class BuildSystemPromptOptions(TypedDict, total=False):
    customPrompt: str
    selectedTools: list[str]
    toolSnippets: dict[str, str]
    promptGuidelines: list[str]
    appendSystemPrompt: str
    cwd: str
    contextFiles: list[dict[str, str]]
    skills: list[Skill]


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    custom_prompt = options.get("customPrompt")
    selected_tools = options.get("selectedTools")
    tool_snippets = options.get("toolSnippets")
    prompt_guidelines = options.get("promptGuidelines")
    append_system_prompt = options.get("appendSystemPrompt")
    cwd = options["cwd"]
    context_files = options.get("contextFiles") or []
    skills = options.get("skills") or []

    prompt_cwd = cwd.replace("\\", "/")
    date = _datetime.date.today().isoformat()
    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""

    if custom_prompt:
        prompt = custom_prompt
        if append_section:
            prompt += append_section
        if context_files:
            prompt += _format_project_context(context_files)
        if (selected_tools is None or "read" in selected_tools) and skills:
            prompt += format_skills_for_prompt(skills)
        prompt += f"\nCurrent date: {date}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    readme_path, docs_path, examples_path = _default_docs_paths()
    tools = selected_tools or ["read", "bash", "edit", "write"]
    visible_tools = [name for name in tools if tool_snippets and tool_snippets.get(name)]
    tools_list = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"

    guidelines: list[str] = []
    seen_guidelines: set[str] = set()

    def add_guideline(guideline: str) -> None:
        if guideline in seen_guidelines:
            return
        seen_guidelines.add(guideline)
        guidelines.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        add_guideline("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    for guideline in prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines_text = "\n".join(f"- {guideline}" for guideline in guidelines)
    prompt = "\n".join(
        [
            "You are an expert coding assistant operating inside pi, a coding agent harness.",
            "You help users by reading files, executing commands, editing code, and writing new files.",
            "",
            "Available tools:",
            tools_list,
            "",
            "In addition to the tools above, you may have access to other custom tools depending on the project.",
            "",
            "Guidelines:",
            guidelines_text,
            "",
            (
                "Pi documentation (read only when the user asks about pi itself, its SDK, "
                "extensions, themes, skills, or TUI):"
            ),
            f"- Main documentation: {readme_path}",
            f"- Additional docs: {docs_path}",
            f"- Examples: {examples_path} (extensions, custom tools, SDK)",
            (
                "- When reading pi docs or examples, resolve docs/... under Additional docs and "
                "examples/... under Examples, not the current working directory"
            ),
            (
                "- When asked about: extensions (docs/extensions.md, examples/extensions/), "
                "themes (docs/themes.md), skills (docs/skills.md), prompt templates "
                "(docs/prompt-templates.md), TUI components (docs/tui.md), keybindings "
                "(docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers "
                "(docs/custom-provider.md), adding models (docs/models.md), pi packages "
                "(docs/packages.md)"
            ),
            (
                "- When working on pi topics, read the docs and examples, and follow .md "
                "cross-references before implementing"
            ),
            "- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)",
        ]
    )

    if append_section:
        prompt += append_section
    if context_files:
        prompt += _format_project_context(context_files)
    if has_read and skills:
        prompt += format_skills_for_prompt(skills)
    prompt += f"\nCurrent date: {date}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


def _format_project_context(context_files: list[dict[str, str]]) -> str:
    prompt = "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
    for item in context_files:
        prompt += f'<project_instructions path="{item["path"]}">\n{item["content"]}\n</project_instructions>\n\n'
    prompt += "</project_context>\n"
    return prompt


def _default_docs_paths() -> tuple[str, str, str]:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[5]
        / "important_repository_of_the_dependencies_source_code_to_read_better_than_online_docs"
        / "earendil-pi",
        module_path.parents[3],
    ]
    for candidate in candidates:
        readme = candidate / "README.md"
        docs = candidate / "docs"
        examples = candidate / "packages" / "coding-agent" / "examples"
        if readme.exists():
            return (
                os.fspath(readme),
                os.fspath(docs if docs.exists() else candidate / "docs"),
                os.fspath(examples if examples.exists() else candidate / "examples"),
            )
    return ("README.md", "docs", "examples")


buildSystemPrompt = build_system_prompt

__all__ = [
    "BuildSystemPromptOptions",
    "buildSystemPrompt",
    "build_system_prompt",
]
