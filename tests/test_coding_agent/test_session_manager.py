from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from harn_coding_agent.core import session_cwd as session_cwd_module
from harn_coding_agent.core import session_manager as session_manager_module
from harn_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    getMissingSessionCwdIssue,
)
from harn_coding_agent.core.session_manager import (
    SessionInfo,
    SessionManager,
    buildSessionContext,
    findMostRecentSession,
    getDefaultSessionDir,
    loadEntriesFromFile,
    migrateSessionEntries,
)


def user_msg(text: str, timestamp: int = 1) -> dict[str, object]:
    return {"role": "user", "content": text, "timestamp": timestamp}


def assistant_msg(text: str, timestamp: int = 2) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-test",
        "usage": {
            "input": 1,
            "output": 1,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 2,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop",
        "timestamp": timestamp,
    }


def test_get_missing_session_cwd_issue_and_override(tmp_path: Path) -> None:
    fallback_cwd = tmp_path / "fallback"
    fallback_cwd.mkdir()
    missing_cwd = fallback_cwd / "does-not-exist"
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "session-id",
                "timestamp": "2025-01-01T00:00:00Z",
                "cwd": str(missing_cwd),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    session = SessionManager.open(str(session_file))
    issue = getMissingSessionCwdIssue(session, str(fallback_cwd))
    assert issue is not None
    assert issue.sessionFile == str(session_file)
    assert issue.sessionCwd == str(missing_cwd)
    assert issue.fallbackCwd == str(fallback_cwd)

    overridden = SessionManager.open(str(session_file), None, str(fallback_cwd))
    assert overridden.getCwd() == str(fallback_cwd)
    assert getMissingSessionCwdIssue(overridden, str(fallback_cwd)) is None


def test_missing_session_cwd_error_contains_paths(tmp_path: Path) -> None:
    fallback_cwd = tmp_path / "fallback"
    fallback_cwd.mkdir()
    session = SessionManager.inMemory(str(tmp_path))
    session.newSession()
    session.cwd = str(tmp_path / "missing")
    session.sessionFile = str(tmp_path / "session.jsonl")

    with pytest.raises(MissingSessionCwdError) as excinfo:
        from harn_coding_agent.core.session_cwd import assertSessionCwdExists

        assertSessionCwdExists(session, str(fallback_cwd))

    message = str(excinfo.value)
    assert "Stored session working directory does not exist" in message
    assert str(fallback_cwd) in message
    assert str(tmp_path / "missing") in message
    assert excinfo.value.name == "MissingSessionCwdError"


def test_session_cwd_exports_match_ts_surface() -> None:
    assert session_cwd_module.__all__ == [
        "MissingSessionCwdError",
        "SessionCwdIssue",
        "assertSessionCwdExists",
        "formatMissingSessionCwdError",
        "formatMissingSessionCwdPrompt",
        "getMissingSessionCwdIssue",
    ]


def test_migrate_session_entries_adds_ids_and_updates_hook_message_role() -> None:
    entries = [
        {"type": "session", "id": "sess-1", "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
        {"type": "message", "timestamp": "2025-01-01T00:00:01Z", "message": {"role": "user", "content": "hi"}},
        {
            "type": "message",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {"role": "hookMessage", "content": "legacy"},
        },
    ]

    migrateSessionEntries(entries)

    assert entries[0]["version"] == 3
    assert isinstance(entries[1]["id"], str)
    assert len(entries[1]["id"]) == 8
    assert entries[1]["parentId"] is None
    assert entries[2]["parentId"] == entries[1]["id"]
    assert entries[2]["message"]["role"] == "custom"


def test_build_session_context_handles_compaction_and_branches() -> None:
    entries = [
        {
            "type": "message",
            "id": "1",
            "parentId": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "message": user_msg("start"),
        },
        {
            "type": "message",
            "id": "2",
            "parentId": "1",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": assistant_msg("r1"),
        },
        {
            "type": "message",
            "id": "3",
            "parentId": "2",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": user_msg("q2", 3),
        },
        {
            "type": "message",
            "id": "4",
            "parentId": "3",
            "timestamp": "2025-01-01T00:00:03Z",
            "message": assistant_msg("r2", 4),
        },
        {
            "type": "compaction",
            "id": "5",
            "parentId": "4",
            "timestamp": "2025-01-01T00:00:04Z",
            "summary": "Compacted history",
            "firstKeptEntryId": "3",
            "tokensBefore": 1000,
        },
        {
            "type": "message",
            "id": "6",
            "parentId": "5",
            "timestamp": "2025-01-01T00:00:05Z",
            "message": user_msg("q3", 5),
        },
        {
            "type": "message",
            "id": "7",
            "parentId": "6",
            "timestamp": "2025-01-01T00:00:06Z",
            "message": assistant_msg("r3", 6),
        },
        {
            "type": "branch_summary",
            "id": "8",
            "parentId": "3",
            "timestamp": "2025-01-01T00:00:07Z",
            "summary": "Tried wrong approach",
            "fromId": "wrong-branch",
        },
        {
            "type": "message",
            "id": "9",
            "parentId": "8",
            "timestamp": "2025-01-01T00:00:08Z",
            "message": user_msg("better", 7),
        },
    ]

    main = buildSessionContext(entries, "7")
    assert len(main.messages) == 5
    assert getattr(main.messages[0], "role", None) == "compactionSummary"
    assert main.messages[1]["content"] == "q2"
    assert main.messages[4]["content"][0]["text"] == "r3"
    assert main.model == {"provider": "anthropic", "modelId": "claude-test"}

    branch = buildSessionContext(entries, "9")
    assert len(branch.messages) == 5
    assert branch.messages[0]["content"] == "start"
    assert getattr(branch.messages[3], "role", None) == "branchSummary"
    assert branch.messages[4]["content"] == "better"

    before_first = buildSessionContext(entries, None)
    assert before_first.messages == []
    assert before_first.model is None


def test_load_entries_from_file_skips_bad_lines_and_requires_header(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
        "not-json\n"
        '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z","message":{"role":"user","content":"hi","timestamp":1}}\n',
        encoding="utf-8",
    )
    assert len(loadEntriesFromFile(str(mixed))) == 2

    no_header = tmp_path / "no-header.jsonl"
    no_header.write_text('{"type":"message","id":"1"}\n', encoding="utf-8")
    assert loadEntriesFromFile(str(no_header)) == []


def test_find_most_recent_session_ignores_invalid_files(tmp_path: Path) -> None:
    older = tmp_path / "older.jsonl"
    older.write_text(
        '{"type":"session","id":"old","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n',
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"type":"not-session"}\n', encoding="utf-8")
    newer = tmp_path / "newer.jsonl"
    newer.write_text(
        '{"type":"session","id":"new","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n',
        encoding="utf-8",
    )

    import time
    os_order = [older, invalid, newer]
    for i, path in enumerate(os_order):
        t = time.time() + i
        os.utime(path, (t, t))

    assert findMostRecentSession(str(tmp_path)) == str(newer)


def test_session_info_modified_ignores_messages_without_content(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
        '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","timestamp":999999999999}}\n',
        encoding="utf-8",
    )

    info = session_manager_module._build_session_info_sync(str(session_file))
    assert info is not None
    assert info.modified == datetime(2025, 1, 1, tzinfo=UTC)


def test_open_recovers_corrupted_file_and_preserves_explicit_path(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.jsonl"
    corrupted.write_text("", encoding="utf-8")

    session = SessionManager.open(str(corrupted), str(tmp_path))

    assert session.getSessionFile() == str(corrupted)
    assert session.getHeader() is not None

    lines = [line for line in corrupted.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "session"

    reopened = SessionManager.open(str(corrupted), str(tmp_path))
    assert reopened.getSessionId() == session.getSessionId()


def test_session_manager_append_branch_and_tree_operations() -> None:
    session = SessionManager.inMemory()
    first = session.appendMessage(user_msg("first"))
    second = session.appendMessage(assistant_msg("second"))
    third = session.appendThinkingLevelChange("high")
    branch_leaf = session.appendMessage(user_msg("third", 3))

    assert session.getLeafId() == branch_leaf
    path = session.getBranch()
    assert [entry["id"] for entry in path] == [first, second, third, branch_leaf]

    session.branch(second)
    branched = session.appendMessage(user_msg("branch", 4))
    tree = session.getTree()
    assert len(tree) == 1
    node2 = tree[0].children[0]
    child_ids = sorted(child.entry["id"] for child in node2.children)
    assert child_ids == sorted([third, branched])


def test_labels_are_preserved_only_for_path_in_branched_session() -> None:
    session = SessionManager.inMemory()
    msg1 = session.appendMessage(user_msg("one"))
    msg2 = session.appendMessage(assistant_msg("two"))
    msg3 = session.appendMessage(user_msg("three", 3))

    session.appendLabelChange(msg1, "first")
    session.appendLabelChange(msg2, "second")
    session.appendLabelChange(msg3, "third")

    session.createBranchedSession(msg2)

    assert session.getLabel(msg1) == "first"
    assert session.getLabel(msg2) == "second"
    assert session.getLabel(msg3) is None
    assert len([entry for entry in session.getEntries() if entry["type"] == "label"]) == 2


def test_custom_entries_are_skipped_from_context_but_kept_in_tree() -> None:
    session = SessionManager.inMemory()
    msg1 = session.appendMessage(user_msg("hello"))
    custom = session.appendCustomEntry("my_data", {"foo": "bar"})
    msg2 = session.appendMessage(assistant_msg("hi"))

    path = session.getBranch()
    assert [entry["id"] for entry in path] == [msg1, custom, msg2]

    context = session.buildSessionContext()
    assert len(context.messages) == 2
    assert context.messages[0]["role"] == "user"
    assert context.messages[1]["role"] == "assistant"


def test_session_manager_exports_match_ts_surface() -> None:
    assert session_manager_module.__all__ == [
        "CURRENT_SESSION_VERSION",
        "SessionHeader",
        "NewSessionOptions",
        "SessionEntryBase",
        "SessionMessageEntry",
        "ThinkingLevelChangeEntry",
        "ModelChangeEntry",
        "CompactionEntry",
        "BranchSummaryEntry",
        "CustomEntry",
        "LabelEntry",
        "SessionInfoEntry",
        "CustomMessageEntry",
        "SessionEntry",
        "FileEntry",
        "SessionTreeNode",
        "SessionContext",
        "SessionInfo",
        "SessionListProgress",
        "ReadonlySessionManager",
        "migrateSessionEntries",
        "parseSessionEntries",
        "getLatestCompactionEntry",
        "buildSessionContext",
        "getDefaultSessionDir",
        "loadEntriesFromFile",
        "findMostRecentSession",
        "SessionManager",
    ]


def test_session_manager_accessors_return_live_references() -> None:
    session = SessionManager.inMemory()
    entry_id = session.appendMessage(user_msg("hello"))

    entries = session.getEntries()
    entries[0]["contentProbe"] = "entry"
    assert session.getEntry(entry_id)["contentProbe"] == "entry"

    branch = session.getBranch()
    branch[0]["branchProbe"] = "branch"
    assert session.getEntry(entry_id)["branchProbe"] == "branch"

    header = session.getHeader()
    assert header is not None
    header["headerProbe"] = "header"
    assert session.getHeader()["headerProbe"] == "header"


def test_session_manager_persisted_json_matches_ts_shape(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    session_dir = tmp_path / "sessions"
    cwd.mkdir()
    session = SessionManager.create(str(cwd), str(session_dir))

    first = session.appendMessage(user_msg("hello"))
    session.appendMessage(assistant_msg("hi"))
    session.appendCompaction("summary", first, 10)

    session_file = session.getSessionFile()
    assert session_file is not None
    lines = [line for line in Path(session_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    assert lines
    assert all(": " not in line and ", " not in line for line in lines)
    assert "\\u" not in lines[0]

    compaction_line = json.loads(lines[-1])
    assert "details" not in compaction_line
    assert "fromHook" not in compaction_line
    assert compaction_line["type"] == "compaction"


def test_session_manager_open_uses_explicit_empty_cwd_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current_cwd = tmp_path / "current"
    current_cwd.mkdir()
    monkeypatch.chdir(current_cwd)

    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/from-header"}\n',
        encoding="utf-8",
    )

    session = SessionManager.open(str(session_file), None, "")
    assert session.getCwd() == os.path.abspath("")


def test_get_default_session_dir_respects_explicit_empty_agent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_cwd = tmp_path / "current"
    project_cwd = tmp_path / "project"
    current_cwd.mkdir()
    project_cwd.mkdir()
    monkeypatch.chdir(current_cwd)

    session_dir = Path(getDefaultSessionDir(str(project_cwd), ""))
    assert session_dir.parent.parent == current_cwd


@pytest.mark.asyncio
async def test_session_manager_list_uses_bounded_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = []
    for index in range(12):
        path = tmp_path / f"{index}.jsonl"
        path.write_text("", encoding="utf-8")
        files.append(path)

    active = 0
    max_active = 0
    progress: list[tuple[int, int]] = []

    async def fake_build_session_info(file_path: str) -> SessionInfo | None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return SessionInfo(
            path=file_path,
            id=Path(file_path).stem,
            cwd=str(tmp_path),
            created=datetime(2025, 1, 1, tzinfo=UTC),
            modified=datetime(2025, 1, 1, tzinfo=UTC),
            messageCount=0,
            firstMessage="(no messages)",
            allMessagesText="",
        )

    monkeypatch.setattr(session_manager_module, "_build_session_info", fake_build_session_info)

    sessions = await SessionManager.list(str(tmp_path), str(tmp_path), lambda loaded, total: progress.append((loaded, total)))

    assert len(sessions) == len(files)
    assert max_active > 1
    assert max_active <= 10
    assert progress[-1] == (len(files), len(files))
