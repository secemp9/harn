"""Shared slash-command metadata for coding-agent command surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from harnify_coding_agent.config import APP_NAME
from harnify_coding_agent.core.source_info import SourceInfo

type SlashCommandSource = Literal["extension", "prompt", "skill"]


class SlashCommandInfo(TypedDict):
    name: str
    source: SlashCommandSource
    sourceInfo: SourceInfo
    description: NotRequired[str]


@dataclass(slots=True, frozen=True)
class BuiltinSlashCommand:
    name: str
    description: str


BUILTIN_SLASH_COMMANDS: tuple[BuiltinSlashCommand, ...] = (
    BuiltinSlashCommand("settings", "Open settings menu"),
    BuiltinSlashCommand("model", "Select model (opens selector UI)"),
    BuiltinSlashCommand("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand("export", "Export session (HTML default, or specify path: .html/.jsonl)"),
    BuiltinSlashCommand("import", "Import and resume a session from a JSONL file"),
    BuiltinSlashCommand("share", "Share session as a secret GitHub gist"),
    BuiltinSlashCommand("copy", "Copy last agent message to clipboard"),
    BuiltinSlashCommand("name", "Set session display name"),
    BuiltinSlashCommand("session", "Show session info and stats"),
    BuiltinSlashCommand("changelog", "Show changelog entries"),
    BuiltinSlashCommand("hotkeys", "Show all keyboard shortcuts"),
    BuiltinSlashCommand("fork", "Create a new fork from a previous user message"),
    BuiltinSlashCommand("clone", "Duplicate the current session at the current position"),
    BuiltinSlashCommand("tree", "Navigate session tree (switch branches)"),
    BuiltinSlashCommand("login", "Configure provider authentication"),
    BuiltinSlashCommand("logout", "Remove provider authentication"),
    BuiltinSlashCommand("new", "Start a new session"),
    BuiltinSlashCommand("compact", "Manually compact the session context"),
    BuiltinSlashCommand("resume", "Resume a different session"),
    BuiltinSlashCommand("reload", "Reload keybindings, extensions, skills, prompts, and themes"),
    BuiltinSlashCommand("quit", f"Quit {APP_NAME}"),
)

_LOCAL_ALIAS_SLASH_COMMANDS: tuple[BuiltinSlashCommand, ...] = (
    BuiltinSlashCommand("models", "Configure the session model list"),
    BuiltinSlashCommand("theme", "Select a theme"),
)


def _make_slash_command_info(
    name: str,
    source: SlashCommandSource,
    source_info: SourceInfo,
    description: str | None = None,
) -> SlashCommandInfo:
    command_info: SlashCommandInfo = {
        "name": name,
        "source": source,
        "sourceInfo": source_info,
    }
    if description is not None:
        command_info["description"] = description
    return command_info


__all__ = [
    "SlashCommandSource",
    "SlashCommandInfo",
    "BuiltinSlashCommand",
    "BUILTIN_SLASH_COMMANDS",
]
