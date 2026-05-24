from __future__ import annotations

import json
from pathlib import Path

import pytest
from harnify_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    getMissingSessionCwdIssue,
)
from harnify_coding_agent.core.session_manager import (
    SessionManager,
    buildSessionContext,
    findMostRecentSession,
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
        from harnify_coding_agent.core.session_cwd import assertSessionCwdExists

        assertSessionCwdExists(session, str(fallback_cwd))

    message = str(excinfo.value)
    assert "Stored session working directory does not exist" in message
    assert str(fallback_cwd) in message
    assert str(tmp_path / "missing") in message


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

    os_order = [older, invalid, newer]
    for path in os_order:
        path.touch()

    assert findMostRecentSession(str(tmp_path)) == str(newer)


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
