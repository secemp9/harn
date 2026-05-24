from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from harnify_agent.harness.session import uuid as session_uuid
from harnify_agent.harness.session.jsonl_repo import JsonlSessionRepo
from harnify_agent.harness.session.jsonl_storage import JsonlSessionStorage, load_jsonl_session_metadata
from harnify_agent.harness.session.memory_repo import InMemorySessionRepo
from harnify_agent.harness.session.memory_storage import InMemorySessionStorage
from harnify_agent.harness.session.repo_utils import create_timestamp
from harnify_agent.harness.session.session import Session, build_session_context
from harnify_agent.harness.types import FileError, FileInfo, SessionError, err, ok
from harnify_ai.types import TextContent


class PathExecutionEnv:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)

    async def absolutePath(self, path: str, abortSignal: Any | None = None):
        return ok(str(self._resolve(path)))

    async def joinPath(self, parts: list[str], abortSignal: Any | None = None):
        if not parts:
            return ok("")
        return ok(os.path.join(*parts))

    async def readTextFile(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            return ok(resolved.read_text(encoding="utf-8"))
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def readTextLines(self, path: str, options: dict[str, Any] | None = None):
        resolved = self._resolve(path)
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
            max_lines = (options or {}).get("maxLines")
            if max_lines is not None:
                lines = lines[:max_lines]
            return ok(lines)
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def writeFile(self, path: str, content: str | bytes, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                resolved.write_bytes(content)
            else:
                resolved.write_text(content, encoding="utf-8")
            return ok(None)
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def appendFile(self, path: str, content: str | bytes, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                with resolved.open("ab") as handle:
                    handle.write(content)
            else:
                with resolved.open("a", encoding="utf-8") as handle:
                    handle.write(content)
            return ok(None)
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

    async def exists(self, path: str, abortSignal: Any | None = None):
        resolved = self._resolve(path)
        return ok(resolved.exists())

    async def createDir(self, path: str, options: dict[str, Any] | None = None):
        resolved = self._resolve(path)
        try:
            resolved.mkdir(parents=(options or {}).get("recursive", True), exist_ok=True)
            return ok(None)
        except OSError as error:
            return err(self._to_file_error(error, str(resolved)))

    async def remove(self, path: str, options: dict[str, Any] | None = None):
        resolved = self._resolve(path)
        force = bool((options or {}).get("force", False))
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            return ok(None)
        except FileNotFoundError as error:
            if force:
                return ok(None)
            return err(self._to_file_error(error, str(resolved)))
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
        if error.errno in {1, 13}:
            return FileError("permission_denied", str(error), path)
        return FileError("unknown", str(error), path)


def _user_message(text: str, timestamp: int = 1) -> dict[str, Any]:
    return {"role": "user", "content": [TextContent(text=text)], "timestamp": timestamp}


def _assistant_message(text: str, timestamp: int = 2) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [TextContent(text=text)],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop",
        "timestamp": timestamp,
    }


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message["role"])
    return str(message.role)


def test_create_timestamp_matches_js_iso_millisecond_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", create_timestamp())


@pytest.mark.asyncio
async def test_in_memory_session_storage_behaviour() -> None:
    metadata = {"id": "session-1", "createdAt": "2026-01-01T00:00:00.000Z"}
    entry = {
        "type": "message",
        "id": "entry-1",
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": _user_message("one"),
    }
    initial_entries = [entry]
    storage = InMemorySessionStorage({"metadata": metadata, "entries": initial_entries})
    initial_entries.append({**entry, "id": "entry-2"})

    assert await storage.getMetadata() == metadata
    assert [stored["id"] for stored in await storage.getEntries()] == ["entry-1"]
    assert await storage.getLeafId() == "entry-1"

    await storage.setLeafId(None)
    assert await storage.getLeafId() is None
    assert (await storage.getEntries())[-1]["type"] == "leaf"

    await storage.appendEntry(
        {
            "type": "label",
            "id": "label-1",
            "parentId": "entry-1",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "targetId": "entry-1",
            "label": "checkpoint",
        }
    )
    assert await storage.getLabel("entry-1") == "checkpoint"
    assert [found["id"] for found in await storage.findEntries("message")] == ["entry-1"]

    with pytest.raises(SessionError, match="Entry missing not found"):
        await storage.setLeafId("missing")


@pytest.mark.asyncio
async def test_session_api_with_in_memory_and_jsonl_storage(tmp_path: Path) -> None:
    async def run_suite(storage: Any) -> None:
        session = Session(storage)
        await session.appendMessage(_user_message("one", 1))
        await session.appendMessage(_assistant_message("two", 2))
        context = await session.buildContext()
        assert [_message_role(message) for message in context.messages] == ["user", "assistant"]

        await session.appendModelChange("openai", "gpt-4.1")
        await session.appendThinkingLevelChange("high")
        changed = await session.buildContext()
        assert changed.thinkingLevel == "high"
        assert changed.model == {"provider": "openai", "modelId": "gpt-4.1"}

        user1 = await session.appendMessage(_user_message("three", 3))
        assistant1 = await session.appendMessage(_assistant_message("four", 4))
        await session.appendMessage(_user_message("five", 5))
        await session.moveTo(user1)
        await session.appendMessage(_assistant_message("branched", 6))
        branch = await session.getBranch()
        branch_ids = [_entry["id"] if isinstance(_entry, dict) else _entry.id for _entry in branch]
        assert user1 in branch_ids
        assert assistant1 not in branch_ids

        await session.moveTo(None)
        assert await session.getLeafId() is None
        assert (await session.buildContext()).messages == []

        await session.appendMessage(_user_message("one", 7))
        await session.appendMessage(_assistant_message("two", 8))
        user2 = await session.appendMessage(_user_message("three", 9))
        await session.appendMessage(_assistant_message("four", 10))
        await session.appendCompaction("summary", user2, 1234)
        await session.appendMessage(_user_message("five", 11))
        compacted = await session.buildContext()
        assert _message_role(compacted.messages[0]) == "compactionSummary"
        assert len(compacted.messages) == 4

        summary_id = await session.moveTo(user2, {"summary": "summary text"})
        summary_entry = await session.getEntry(summary_id)
        assert summary_entry["type"] == "branch_summary"
        assert (await session.buildContext()).messages[-1].role == "branchSummary"

        await session.appendCustomMessageEntry("custom", "hello", True, {"ok": True})
        custom_context = await session.buildContext()
        assert _message_role(custom_context.messages[-1]) == "custom"

        label_target = await session.appendMessage(_user_message("label", 12))
        await session.appendLabel(label_target, "checkpoint")
        await session.appendSessionName("name")
        assert await session.getLabel(label_target) == "checkpoint"
        assert await session.getSessionName() == "name"

    await run_suite(InMemorySessionStorage())

    env = PathExecutionEnv(tmp_path)
    jsonl_storage = await JsonlSessionStorage.create(
        env,
        str(tmp_path / "session.jsonl"),
        {"cwd": str(tmp_path), "sessionId": "session-1"},
    )
    await run_suite(jsonl_storage)


@pytest.mark.asyncio
async def test_jsonl_session_storage_round_trip_and_metadata(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    file_path = str(tmp_path / "session.jsonl")
    storage = await JsonlSessionStorage.create(
        env,
        file_path,
        {"cwd": str(tmp_path), "sessionId": "session-1", "parentSessionPath": "/tmp/parent.jsonl"},
    )

    assert Path(file_path).exists()
    assert len(Path(file_path).read_text(encoding="utf-8").strip().splitlines()) == 1
    metadata = await storage.getMetadata()
    assert metadata["id"] == "session-1"
    assert metadata["cwd"] == str(tmp_path)
    assert metadata["path"] == file_path
    assert metadata["parentSessionPath"] == "/tmp/parent.jsonl"

    root = {
        "type": "message",
        "id": "root",
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": _user_message("root", 1),
    }
    child = {
        "type": "message",
        "id": "child",
        "parentId": "root",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": _assistant_message("child", 2),
    }
    await storage.appendEntry(root)
    await storage.appendEntry(child)
    assert await load_jsonl_session_metadata(env, file_path) == metadata

    loaded = await JsonlSessionStorage.open(env, file_path)
    assert await loaded.getLeafId() == "child"
    assert [entry["id"] for entry in await loaded.getEntries()] == ["root", "child"]
    await loaded.setLeafId("root")
    reloaded = await JsonlSessionStorage.open(env, file_path)
    assert await reloaded.getLeafId() == "root"
    assert (await reloaded.getEntries())[-1]["type"] == "leaf"
    assert [entry["id"] for entry in await loaded.getPathToRoot("child")] == ["root", "child"]


@pytest.mark.asyncio
async def test_jsonl_session_storage_errors_and_metadata_reader(tmp_path: Path) -> None:
    env = PathExecutionEnv(tmp_path)
    missing_path = str(tmp_path / "missing.jsonl")
    with pytest.raises(SessionError) as missing_error:
        await JsonlSessionStorage.open(env, missing_path)
    assert missing_error.value.code == "not_found"

    malformed_header_path = tmp_path / "malformed-header.jsonl"
    malformed_header_path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(SessionError, match="first line is not a valid session header"):
        await JsonlSessionStorage.open(env, str(malformed_header_path))

    malformed_entry_path = tmp_path / "malformed-entry.jsonl"
    header = {
        "type": "session",
        "version": 3,
        "id": "session-1",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": str(tmp_path),
    }
    malformed_entry_path.write_text(f'{header!s}\n'.replace("'", '"') + "not json\n", encoding="utf-8")
    with pytest.raises(SessionError) as malformed_entry_error:
        await JsonlSessionStorage.open(env, str(malformed_entry_path))
    assert malformed_entry_error.value.code == "invalid_entry"

    minimally_valid_message_path = tmp_path / "minimally-valid-message.jsonl"
    minimally_valid_message_path.write_text(
        (
            f'{header!s}\n'.replace("'", '"')
            + '{"type":"message","id":"entry-1","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"user","content":123,"timestamp":1}}\n'
        ),
        encoding="utf-8",
    )
    minimally_valid_storage = await JsonlSessionStorage.open(env, str(minimally_valid_message_path))
    assert (await minimally_valid_storage.getEntries())[0]["message"]["content"] == 123

    class MetadataFs:
        async def readTextLines(self, path: str, options: dict[str, Any] | None = None):
            return ok(
                [
                    f'{{"type":"session","version":3,"id":"session-1","timestamp":"2026-01-01T00:00:00.000Z","cwd":"{tmp_path}"}}'
                ]
            )

        async def readTextFile(self, *args: Any, **kwargs: Any):
            raise AssertionError("readTextFile should not be called")

        async def writeFile(self, *args: Any, **kwargs: Any):
            return ok(None)

        async def appendFile(self, *args: Any, **kwargs: Any):
            return ok(None)

    metadata = await load_jsonl_session_metadata(MetadataFs(), str(tmp_path / "session.jsonl"))
    assert metadata["id"] == "session-1"
    assert metadata["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_session_repos_support_fork_delete_and_cwd_layout(tmp_path: Path) -> None:
    repo = InMemorySessionRepo()
    session = await repo.create({"id": "session-1"})
    empty_id_session = await repo.create({"id": ""})
    metadata = await session.getMetadata()
    assert (await empty_id_session.getMetadata())["id"] == ""
    user1 = await session.appendMessage(_user_message("one", 1))
    assistant1 = await session.appendMessage(_assistant_message("two", 2))
    user2 = await session.appendMessage(_user_message("three", 3))
    assert await repo.open(metadata) is session
    assert [info["id"] for info in await repo.list()] == ["session-1", ""]
    fork = await repo.fork(metadata, {"entryId": user2, "id": "session-2"})
    assert [entry["id"] for entry in await fork.getEntries()] == [user1, assistant1]
    full_fork = await repo.fork(metadata, {"id": "session-3"})
    assert [entry["id"] for entry in await full_fork.getEntries()] == [user1, assistant1, user2]
    await repo.delete(metadata)
    with pytest.raises(SessionError, match="Session not found: session-1"):
        await repo.open(metadata)

    env = PathExecutionEnv(tmp_path)
    jsonl_repo = JsonlSessionRepo({"fs": env, "sessionsRoot": str(tmp_path / "sessions")})
    source = await jsonl_repo.create({"cwd": "/tmp/source", "id": "source-session"})
    source_metadata = await source.getMetadata()
    j_user1 = await source.appendMessage(_user_message("one", 10))
    j_assistant1 = await source.appendMessage(_assistant_message("two", 11))
    j_user2 = await source.appendMessage(_user_message("three", 12))
    opened = await jsonl_repo.open(source_metadata)
    assert await opened.getMetadata() == source_metadata
    forked = await jsonl_repo.fork(source_metadata, {"cwd": "/tmp/target", "id": "fork-session", "entryId": j_user2})
    forked_metadata = await forked.getMetadata()
    assert forked_metadata["cwd"] == "/tmp/target"
    assert forked_metadata["parentSessionPath"] == source_metadata["path"]
    assert [entry["id"] for entry in await forked.getEntries()] == [j_user1, j_assistant1]
    empty_parent = await jsonl_repo.fork(
        source_metadata,
        {"cwd": "/tmp/target", "id": "fork-empty-parent", "parentSessionPath": ""},
    )
    assert (await empty_parent.getMetadata())["parentSessionPath"] == ""
    full_forked = await jsonl_repo.fork(source_metadata, {"cwd": "/tmp/target", "id": "full-fork-session"})
    assert [entry["id"] for entry in await full_forked.getEntries()] == [j_user1, j_assistant1, j_user2]
    assert "--tmp-source--" in source_metadata["path"]
    await jsonl_repo.delete(source_metadata)
    assert not Path(source_metadata["path"]).exists()


def test_build_session_context_and_uuidv7_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = build_session_context(
        [
            {
                "type": "message",
                "id": "u1",
                "parentId": None,
                "timestamp": "2026-01-01T00:00:00.000Z",
                "message": _user_message("one", 1),
            },
            {
                "type": "model_change",
                "id": "m1",
                "parentId": "u1",
                "timestamp": "2026-01-01T00:00:01.000Z",
                "provider": "openai",
                "modelId": "gpt-4.1",
            },
            {
                "type": "message",
                "id": "a1",
                "parentId": "m1",
                "timestamp": "2026-01-01T00:00:02.000Z",
                "message": _assistant_message("two", 2),
            },
            {
                "type": "thinking_level_change",
                "id": "t1",
                "parentId": "a1",
                "timestamp": "2026-01-01T00:00:03.000Z",
                "thinkingLevel": "high",
            },
        ]
    )
    assert loaded.thinkingLevel == "high"
    assert loaded.model == {"provider": "anthropic", "modelId": "claude-sonnet-4-5"}

    random_values = [
        bytes([0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFE, 0x01, 0x11, 0x22, 0x33, 0x44, 0x55]),
        bytes(16),
        bytes(16),
    ]
    monkeypatch.setattr(session_uuid, "_fill_random_bytes", lambda length: random_values.pop(0))
    monkeypatch.setattr(session_uuid.time, "time", lambda: 0x0123456789AB / 1000)
    monkeypatch.setattr(session_uuid, "_last_timestamp", -1)
    monkeypatch.setattr(session_uuid, "_sequence", 0)

    first = session_uuid.uuidv7()
    second = session_uuid.uuidv7()
    third = session_uuid.uuidv7()

    assert first == "01234567-89ab-7fff-bfff-f91122334455"
    assert second == "01234567-89ab-7fff-bfff-fc0000000000"
    assert third == "01234567-89ac-7000-8000-000000000000"
    assert first < second < third
