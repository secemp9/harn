"""Autocomplete helpers for slash commands and file paths."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from harnify_tui.fuzzy import fuzzy_filter

PATH_DELIMITERS = {" ", "\t", '"', "'", "="}
T = TypeVar("T")


def to_display_path(value: str) -> str:
    return value.replace("\\", "/")


def escape_regex(value: str) -> str:
    return re.escape(value)


def build_fd_path_query(query: str) -> str:
    normalized = to_display_path(query)
    if "/" not in normalized:
        return normalized

    has_trailing_separator = normalized.endswith("/")
    trimmed = normalized.strip("/")
    if not trimmed:
        return normalized

    segments = [escape_regex(segment) for segment in trimmed.split("/") if segment]
    if not segments:
        return normalized

    pattern = r"[\\/]"
    result = pattern.join(segments)
    if has_trailing_separator:
        result += pattern
    return result


def find_last_delimiter(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in PATH_DELIMITERS:
            return index
    return -1


def find_unclosed_quote_start(text: str) -> int | None:
    in_quotes = False
    quote_start = -1
    for index, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = index
    return quote_start if in_quotes else None


def is_token_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in PATH_DELIMITERS


def extract_quoted_prefix(text: str) -> str | None:
    quote_start = find_unclosed_quote_start(text)
    if quote_start is None:
        return None

    if quote_start > 0 and text[quote_start - 1] == "@":
        if not is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1 :]

    if not is_token_start(text, quote_start):
        return None

    return text[quote_start:]


@dataclass(slots=True)
class PathPrefix:
    rawPrefix: str
    isAtPrefix: bool
    isQuotedPrefix: bool


def parse_path_prefix(prefix: str) -> PathPrefix:
    if prefix.startswith('@"'):
        return PathPrefix(rawPrefix=prefix[2:], isAtPrefix=True, isQuotedPrefix=True)
    if prefix.startswith('"'):
        return PathPrefix(rawPrefix=prefix[1:], isAtPrefix=False, isQuotedPrefix=True)
    if prefix.startswith("@"):
        return PathPrefix(rawPrefix=prefix[1:], isAtPrefix=True, isQuotedPrefix=False)
    return PathPrefix(rawPrefix=prefix, isAtPrefix=False, isQuotedPrefix=False)


def build_completion_value(path: str, *, is_directory: bool, is_at_prefix: bool, is_quoted_prefix: bool) -> str:
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""
    if not needs_quotes:
        return f"{prefix}{path}"
    return f'{prefix}"{path}"'


def _is_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    aborted = getattr(signal, "aborted", None)
    if isinstance(aborted, bool):
        return aborted
    is_set = getattr(signal, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return False


async def walk_directory_with_fd(
    base_dir: str,
    fd_path: str,
    query: str,
    max_results: int,
    signal: Any,
) -> list[dict[str, object]]:
    if _is_aborted(signal):
        return []

    args = [
        fd_path,
        "--base-directory",
        base_dir,
        "--max-results",
        str(max_results),
        "--type",
        "f",
        "--type",
        "d",
        "--follow",
        "--hidden",
        "--exclude",
        ".git",
        "--exclude",
        ".git/*",
        "--exclude",
        ".git/**",
    ]
    if "/" in to_display_path(query):
        args.append("--full-path")
    if query:
        args.append(build_fd_path_query(query))

    def run_fd() -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, capture_output=True, text=True, check=False)

    result = await asyncio.to_thread(run_fd)
    if _is_aborted(signal) or result.returncode != 0 or not result.stdout:
        return []

    lines = [line for line in result.stdout.strip().splitlines() if line]
    entries: list[dict[str, object]] = []
    for line in lines:
        display_line = to_display_path(line)
        has_trailing_separator = display_line.endswith("/")
        normalized_path = display_line[:-1] if has_trailing_separator else display_line
        if normalized_path == ".git" or normalized_path.startswith(".git/") or "/.git/" in normalized_path:
            continue
        entries.append({"path": display_line, "isDirectory": has_trailing_separator})
    return entries


@dataclass(slots=True)
class AutocompleteItem:
    value: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class SlashCommand:
    name: str
    description: str | None = None
    argumentHint: str | None = None
    getArgumentCompletions: (
        Callable[[str], Awaitable[list[AutocompleteItem] | None] | list[AutocompleteItem] | None] | None
    ) = None


@dataclass(slots=True)
class AutocompleteSuggestions:
    items: list[AutocompleteItem]
    prefix: str


class AutocompleteProvider(Protocol):
    async def getSuggestions(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None: ...

    def applyCompletion(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> dict[str, object]: ...

    def shouldTriggerFileCompletion(self, lines: list[str], cursorLine: int, cursorCol: int) -> bool: ...


class CombinedAutocompleteProvider:
    def __init__(
        self,
        commands: list[SlashCommand | AutocompleteItem] | None = None,
        basePath: str = ".",
        fdPath: str | None = None,
    ) -> None:
        self.commands = commands or []
        self.basePath = basePath
        self.fdPath = fdPath if fdPath is not None else shutil.which("fd")

    async def getSuggestions(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        current_line = lines[cursorLine] if cursorLine < len(lines) else ""
        text_before_cursor = current_line[:cursorCol]
        force = bool(options.get("force", False))
        signal = options.get("signal")

        at_prefix = self.extractAtPrefix(text_before_cursor)
        if at_prefix:
            parsed = parse_path_prefix(at_prefix)
            suggestions = await self.getFuzzyFileSuggestions(
                parsed.rawPrefix,
                isQuotedPrefix=parsed.isQuotedPrefix,
                signal=signal,
            )
            if not suggestions:
                return None
            return AutocompleteSuggestions(items=suggestions, prefix=at_prefix)

        if not force and text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")
            if space_index == -1:
                prefix = text_before_cursor[1:]
                command_items: list[dict[str, str | None]] = []
                for command in self.commands:
                    if isinstance(command, SlashCommand):
                        name = command.name
                        hint = command.argumentHint
                        description = command.description or ""
                    else:
                        name = command.value
                        hint = None
                        description = command.description or ""
                    full_description = (
                        f"{hint} — {description}" if hint and description else hint or description or None
                    )
                    command_items.append({"name": name, "label": name, "description": full_description})

                filtered = fuzzy_filter(command_items, prefix, lambda item: str(item["name"]))
                if not filtered:
                    return None
                return AutocompleteSuggestions(
                    items=[
                        AutocompleteItem(
                            value=str(item["name"]),
                            label=str(item["label"]),
                            description=item["description"] if isinstance(item["description"], str) else None,
                        )
                        for item in filtered
                    ],
                    prefix=text_before_cursor,
                )

            command_name = text_before_cursor[1:space_index]
            argument_text = text_before_cursor[space_index + 1 :]
            command_match = next(
                (
                    command
                    for command in self.commands
                    if (command.name if isinstance(command, SlashCommand) else command.value) == command_name
                ),
                None,
            )
            if not isinstance(command_match, SlashCommand):
                return None
            if command_match.getArgumentCompletions is None:
                return None
            completions = command_match.getArgumentCompletions(argument_text)
            if asyncio.iscoroutine(completions):
                completions = await completions
            if not isinstance(completions, list) or len(completions) == 0:
                return None
            return AutocompleteSuggestions(items=completions, prefix=argument_text)

        path_match = self.extractPathPrefix(text_before_cursor, forceExtract=force)
        if path_match is None:
            return None

        suggestions = self.getFileSuggestions(path_match)
        if not suggestions:
            return None
        return AutocompleteSuggestions(items=suggestions, prefix=path_match)

    def applyCompletion(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> dict[str, object]:
        current_line = lines[cursorLine] if cursorLine < len(lines) else ""
        before_prefix = current_line[: cursorCol - len(prefix)]
        after_cursor = current_line[cursorCol:]
        is_quoted_prefix = prefix.startswith('"') or prefix.startswith('@"')
        has_leading_quote_after_cursor = after_cursor.startswith('"')
        has_trailing_quote_in_item = item.value.endswith('"')
        adjusted_after_cursor = (
            after_cursor[1:]
            if is_quoted_prefix and has_trailing_quote_in_item and has_leading_quote_after_cursor
            else after_cursor
        )

        is_slash_command = prefix.startswith("/") and before_prefix.strip() == "" and "/" not in prefix[1:]
        if is_slash_command:
            new_line = f"{before_prefix}/{item.value} {adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursorLine] = new_line
            return {"lines": new_lines, "cursorLine": cursorLine, "cursorCol": len(before_prefix) + len(item.value) + 2}

        if prefix.startswith("@"):
            is_directory = item.label.endswith("/")
            suffix = "" if is_directory else " "
            new_line = f"{before_prefix}{item.value}{suffix}{adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursorLine] = new_line
            has_trailing_quote = item.value.endswith('"')
            cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)
            return {
                "lines": new_lines,
                "cursorLine": cursorLine,
                "cursorCol": len(before_prefix) + cursor_offset + len(suffix),
            }

        text_before_cursor = current_line[:cursorCol]
        if "/" in text_before_cursor and " " in text_before_cursor:
            new_line = f"{before_prefix}{item.value}{adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursorLine] = new_line
            is_directory = item.label.endswith("/")
            has_trailing_quote = item.value.endswith('"')
            cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)
            return {"lines": new_lines, "cursorLine": cursorLine, "cursorCol": len(before_prefix) + cursor_offset}

        new_line = f"{before_prefix}{item.value}{adjusted_after_cursor}"
        new_lines = list(lines)
        new_lines[cursorLine] = new_line
        is_directory = item.label.endswith("/")
        has_trailing_quote = item.value.endswith('"')
        cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)
        return {"lines": new_lines, "cursorLine": cursorLine, "cursorCol": len(before_prefix) + cursor_offset}

    def extractAtPrefix(self, text: str) -> str | None:
        quoted_prefix = extract_quoted_prefix(text)
        if quoted_prefix and quoted_prefix.startswith('@"'):
            return quoted_prefix

        last_delimiter_index = find_last_delimiter(text)
        token_start = 0 if last_delimiter_index == -1 else last_delimiter_index + 1
        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]
        return None

    def extractPathPrefix(self, text: str, forceExtract: bool = False) -> str | None:
        quoted_prefix = extract_quoted_prefix(text)
        if quoted_prefix is not None:
            return quoted_prefix

        last_delimiter_index = find_last_delimiter(text)
        path_prefix = text if last_delimiter_index == -1 else text[last_delimiter_index + 1 :]

        if forceExtract:
            return path_prefix
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix
        if path_prefix == "" and text.endswith(" "):
            return path_prefix
        return None

    def expandHomePath(self, path: str) -> str:
        if path.startswith("~/"):
            expanded = os.path.join(str(Path.home()), path[2:])
            return f"{expanded}/" if path.endswith("/") and not expanded.endswith("/") else expanded
        if path == "~":
            return str(Path.home())
        return path

    def resolveScopedFuzzyQuery(self, raw_query: str) -> dict[str, str] | None:
        normalized_query = to_display_path(raw_query)
        slash_index = normalized_query.rfind("/")
        if slash_index == -1:
            return None

        display_base = normalized_query[: slash_index + 1]
        query = normalized_query[slash_index + 1 :]
        if display_base.startswith("~/"):
            base_dir = self.expandHomePath(display_base)
        elif display_base.startswith("/"):
            base_dir = display_base
        else:
            base_dir = os.path.join(self.basePath, display_base)

        try:
            if not os.path.isdir(base_dir):
                return None
        except OSError:
            return None

        return {"baseDir": base_dir, "query": query, "displayBase": display_base}

    def scopedPathForDisplay(self, display_base: str, relative_path: str) -> str:
        normalized_relative_path = to_display_path(relative_path)
        if display_base == "/":
            return f"/{normalized_relative_path}"
        return f"{to_display_path(display_base)}{normalized_relative_path}"

    def getFileSuggestions(self, prefix: str) -> list[AutocompleteItem]:
        try:
            parsed = parse_path_prefix(prefix)
            raw_prefix = parsed.rawPrefix
            expanded_prefix = self.expandHomePath(raw_prefix) if raw_prefix.startswith("~") else raw_prefix

            is_root_prefix = raw_prefix in {"", "./", "../", "~", "~/", "/"} or (parsed.isAtPrefix and raw_prefix == "")
            if is_root_prefix:
                search_dir = (
                    expanded_prefix
                    if raw_prefix.startswith("~") or expanded_prefix.startswith("/")
                    else os.path.join(self.basePath, expanded_prefix)
                )
                search_prefix = ""
            elif raw_prefix.endswith("/"):
                search_dir = (
                    expanded_prefix
                    if raw_prefix.startswith("~") or expanded_prefix.startswith("/")
                    else os.path.join(self.basePath, expanded_prefix)
                )
                search_prefix = ""
            else:
                directory = os.path.dirname(expanded_prefix)
                file_prefix = os.path.basename(expanded_prefix)
                search_dir = (
                    directory
                    if raw_prefix.startswith("~") or expanded_prefix.startswith("/")
                    else os.path.join(self.basePath, directory)
                )
                search_prefix = file_prefix

            entries = list(os.scandir(search_dir))
            suggestions: list[AutocompleteItem] = []
            for entry in entries:
                if not entry.name.lower().startswith(search_prefix.lower()):
                    continue

                is_directory = entry.is_dir(follow_symlinks=False)
                if not is_directory and entry.is_symlink():
                    try:
                        is_directory = os.path.isdir(os.path.join(search_dir, entry.name))
                    except OSError:
                        is_directory = False

                display_prefix = raw_prefix
                if display_prefix.endswith("/"):
                    relative_path = display_prefix + entry.name
                elif "/" in display_prefix or "\\" in display_prefix:
                    if display_prefix.startswith("~/"):
                        home_relative_dir = display_prefix[2:]
                        directory = os.path.dirname(home_relative_dir)
                        relative_path = (
                            f"~/{entry.name}"
                            if directory in {"", "."}
                            else f"~/{to_display_path(os.path.join(directory, entry.name))}"
                        )
                    elif display_prefix.startswith("/"):
                        directory = os.path.dirname(display_prefix)
                        relative_path = f"/{entry.name}" if directory == "/" else f"{directory}/{entry.name}"
                    else:
                        relative_path = os.path.join(os.path.dirname(display_prefix), entry.name)
                        if display_prefix.startswith("./") and not relative_path.startswith("./"):
                            relative_path = f"./{relative_path}"
                else:
                    relative_path = f"~/{entry.name}" if display_prefix.startswith("~") else entry.name

                relative_path = to_display_path(relative_path)
                path_value = f"{relative_path}/" if is_directory else relative_path
                value = build_completion_value(
                    path_value,
                    is_directory=is_directory,
                    is_at_prefix=parsed.isAtPrefix,
                    is_quoted_prefix=parsed.isQuotedPrefix,
                )
                suggestions.append(
                    AutocompleteItem(value=value, label=f"{entry.name}/" if is_directory else entry.name)
                )

            suggestions.sort(key=lambda item: (0 if item.value.endswith("/") else 1, item.label))
            return suggestions
        except OSError:
            return []

    def scoreEntry(self, filePath: str, query: str, isDirectory: bool) -> int:
        file_name = os.path.basename(filePath)
        lower_file_name = file_name.lower()
        lower_query = query.lower()

        score = 0
        if lower_file_name == lower_query:
            score = 100
        elif lower_file_name.startswith(lower_query):
            score = 80
        elif lower_query in lower_file_name:
            score = 50
        elif lower_query in filePath.lower():
            score = 30
        if isDirectory and score > 0:
            score += 10
        return score

    async def getFuzzyFileSuggestions(self, query: str, *, isQuotedPrefix: bool, signal: Any) -> list[AutocompleteItem]:
        if not self.fdPath or _is_aborted(signal):
            return []
        try:
            scoped_query = self.resolveScopedFuzzyQuery(query)
            fd_base_dir = scoped_query["baseDir"] if scoped_query is not None else self.basePath
            fd_query = scoped_query["query"] if scoped_query is not None else query
            entries = await walk_directory_with_fd(fd_base_dir, self.fdPath, fd_query, 100, signal)
            if _is_aborted(signal):
                return []

            scored_entries = []
            for entry in entries:
                score = self.scoreEntry(str(entry["path"]), fd_query, bool(entry["isDirectory"])) if fd_query else 1
                if score > 0:
                    scored_entries.append({**entry, "score": score})

            scored_entries.sort(key=lambda entry: int(entry["score"]), reverse=True)
            top_entries = scored_entries[:20]
            suggestions: list[AutocompleteItem] = []
            for entry in top_entries:
                entry_path = str(entry["path"])
                is_directory = bool(entry["isDirectory"])
                path_without_slash = entry_path[:-1] if is_directory else entry_path
                display_path = (
                    self.scopedPathForDisplay(scoped_query["displayBase"], path_without_slash)
                    if scoped_query is not None
                    else path_without_slash
                )
                entry_name = os.path.basename(path_without_slash)
                completion_path = f"{display_path}/" if is_directory else display_path
                value = build_completion_value(
                    completion_path,
                    is_directory=is_directory,
                    is_at_prefix=True,
                    is_quoted_prefix=isQuotedPrefix,
                )
                suggestions.append(
                    AutocompleteItem(
                        value=value,
                        label=f"{entry_name}/" if is_directory else entry_name,
                        description=display_path,
                    )
                )
            return suggestions
        except Exception:
            return []

    def shouldTriggerFileCompletion(self, lines: list[str], cursorLine: int, cursorCol: int) -> bool:
        current_line = lines[cursorLine] if cursorLine < len(lines) else ""
        text_before_cursor = current_line[:cursorCol]
        return not (text_before_cursor.strip().startswith("/") and " " not in text_before_cursor.strip())


__all__ = [
    "AutocompleteItem",
    "AutocompleteProvider",
    "AutocompleteSuggestions",
    "CombinedAutocompleteProvider",
    "PathPrefix",
    "SlashCommand",
    "build_completion_value",
    "build_fd_path_query",
    "escape_regex",
    "extract_quoted_prefix",
    "find_last_delimiter",
    "find_unclosed_quote_start",
    "is_token_start",
    "parse_path_prefix",
    "to_display_path",
    "walk_directory_with_fd",
]
