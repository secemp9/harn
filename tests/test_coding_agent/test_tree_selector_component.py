from __future__ import annotations

import re
import time

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.session_manager import SessionTreeNode
from harnify_coding_agent.modes.interactive.components.tree_selector import TreeSelectorComponent
from harnify_coding_agent.modes.interactive.theme.theme import init_theme
from harnify_tui import setKeybindings

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")


def _build_tool_tree() -> list[SessionTreeNode]:
    return [
        SessionTreeNode(
            entry={
                "type": "message",
                "id": "1",
                "parentId": None,
                "timestamp": "2026-05-23T12:00:00Z",
                "message": {"role": "user", "content": "Show me the file contents"},
            },
            children=[
                SessionTreeNode(
                    entry={
                        "type": "message",
                        "id": "2",
                        "parentId": "1",
                        "timestamp": "2026-05-23T12:00:01Z",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call-1",
                                    "name": "read",
                                    "arguments": {
                                        "path": "/home/secemp9/demo.txt",
                                        "offset": 1,
                                        "limit": 3,
                                    },
                                }
                            ],
                            "stopReason": "toolUse",
                        },
                    },
                    children=[
                        SessionTreeNode(
                            entry={
                                "type": "message",
                                "id": "3",
                                "parentId": "2",
                                "timestamp": "2026-05-23T12:00:02Z",
                                "message": {
                                    "role": "toolResult",
                                    "toolCallId": "call-1",
                                    "toolName": "read",
                                    "content": "alpha",
                                },
                            },
                            children=[
                                SessionTreeNode(
                                    entry={
                                        "type": "message",
                                        "id": "4",
                                        "parentId": "3",
                                        "timestamp": "2026-05-23T12:00:03Z",
                                        "message": {
                                            "role": "assistant",
                                            "content": [{"type": "text", "text": "Here are the contents"}],
                                            "stopReason": "stop",
                                        },
                                    },
                                    children=[],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    ]


def _build_branch_tree() -> list[SessionTreeNode]:
    return [
        SessionTreeNode(
            entry={
                "type": "message",
                "id": "root",
                "parentId": None,
                "timestamp": "2026-05-23T10:00:00Z",
                "message": {"role": "user", "content": "Root"},
            },
            children=[
                SessionTreeNode(
                    entry={
                        "type": "message",
                        "id": "child-a",
                        "parentId": "root",
                        "timestamp": "2026-05-23T10:00:01Z",
                        "message": {"role": "user", "content": "Branch A"},
                    },
                    children=[],
                ),
                SessionTreeNode(
                    entry={
                        "type": "message",
                        "id": "child-b",
                        "parentId": "root",
                        "timestamp": "2026-05-23T10:00:02Z",
                        "message": {"role": "user", "content": "Branch B"},
                    },
                    children=[],
                ),
            ],
        )
    ]


def test_tree_selector_formats_tool_results_and_filters_search_cancel() -> None:
    cancelled: list[bool] = []
    selected: list[str] = []
    component = TreeSelectorComponent(
        _build_tool_tree(),
        "4",
        20,
        selected.append,
        lambda: cancelled.append(True),
    )

    output = _strip_ansi("\n".join(component.render(140)))
    assert "assistant: (no content)" not in output
    assert "[read: ~/demo.txt:1-3]" in output
    assert "assistant: Here are the contents" in output

    component.handleInput("\x14")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "[read: ~/demo.txt:1-3]" not in output
    assert "[no-tools]" in output

    component.handleInput("\x15")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "user: Show me the file contents" in output
    assert "assistant: Here are the contents" not in output
    assert "[user]" in output

    component.handleInput("\x04")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "user: Show me the file contents" in output
    assert "assistant: Here are the contents" in output
    assert "[user]" not in output

    component.handleInput("Here")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "assistant: Here are the contents" in output
    assert "user: Show me the file contents" not in output

    component.handleInput("\x1b")
    assert cancelled == []
    output = _strip_ansi("\n".join(component.render(140)))
    assert "user: Show me the file contents" in output

    component.handleInput("\r")
    assert selected == ["4"]

    component.handleInput("\x1b")
    assert cancelled == [True]


def test_tree_selector_folds_and_supports_label_editing() -> None:
    label_changes: list[tuple[str, str | None]] = []
    component = TreeSelectorComponent(
        _build_branch_tree(),
        None,
        20,
        lambda _entry_id: None,
        lambda: None,
        lambda entry_id, label: label_changes.append((entry_id, label)),
        initialSelectedId="root",
    )

    output = _strip_ansi("\n".join(component.render(140)))
    assert "Branch A" in output
    assert "Branch B" in output

    component.handleInput("\x1bB")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Branch A" not in output
    assert "Branch B" not in output

    component.handleInput("\x1bF")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Branch A" in output
    assert "Branch B" in output

    component.handleInput("\x1b[27;2;76~")
    assert component.labelInput is not None

    component.labelInput.input.setValue("Checkpoint")
    component.labelInput.input.cursor = len(component.labelInput.input.getValue())
    component.handleInput("\r")

    assert label_changes == [("root", "Checkpoint")]
    assert component.labelInput is None

    component.getTreeList().updateNodeLabel("root", "Checkpoint", "2026-01-02T12:34:00+00:00")
    component.handleInput("\x1b[27;2;84~")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "[Checkpoint]" in output
    assert "[+label time]" in output


def test_tree_selector_auto_cancels_empty_tree() -> None:
    cancelled: list[bool] = []
    TreeSelectorComponent([], None, 20, lambda _entry_id: None, lambda: cancelled.append(True))
    time.sleep(0.15)
    assert cancelled == [True]
