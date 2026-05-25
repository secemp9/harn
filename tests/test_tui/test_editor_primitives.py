from __future__ import annotations

from dataclasses import dataclass, field

from harnify_tui import editor_component as editor_component_module
from harnify_tui import kill_ring as kill_ring_module
from harnify_tui.kill_ring import KillRing
from harnify_tui.undo_stack import UndoStack
from harnify_tui.utils import getSegmenter


def test_segmenter_keeps_combining_marks_and_emoji_atomic() -> None:
    segmenter = getSegmenter()
    segments = segmenter.segment("a\u0301🙂🇨🇳")

    assert [(segment.segment, segment.index) for segment in segments] == [
        ("a\u0301", 0),
        ("🙂", 2),
        ("🇨🇳", 3),
    ]


def test_kill_ring_accumulates_backward_and_forward_kills() -> None:
    ring = KillRing()
    ring.push("world", {"prepend": True})
    ring.push("hello ", {"prepend": True, "accumulate": True})
    assert ring.peek() == "hello world"

    ring.push("foo", {"prepend": False})
    ring.push("bar", {"prepend": False, "accumulate": True})
    assert ring.peek() == "foobar"
    assert ring.length == 2


def test_kill_ring_rotate_cycles_latest_entry() -> None:
    ring = KillRing()
    ring.push("first", {"prepend": False})
    ring.push("second", {"prepend": False})
    ring.push("third", {"prepend": False})

    ring.rotate()
    assert ring.peek() == "second"
    ring.rotate()
    assert ring.peek() == "first"
    ring.rotate()
    assert ring.peek() == "third"


def test_kill_ring_module_exports_match_ts_surface() -> None:
    assert kill_ring_module.__all__ == ["KillRing"]
    assert not hasattr(kill_ring_module, "KillRingPushOptions")


def test_undo_stack_pushes_deep_clones_and_clears() -> None:
    stack: UndoStack[dict[str, object]] = UndoStack()
    state = {"lines": ["hello"], "cursor": 1}
    stack.push(state)
    state["lines"].append("world")
    state["cursor"] = 2

    snapshot = stack.pop()
    assert snapshot == {"lines": ["hello"], "cursor": 1}
    assert stack.pop() is None

    stack.push({"value": "x"})
    assert stack.length == 1
    stack.clear()
    assert stack.length == 0


@dataclass
class DummyEditor:
    text: str = ""
    onSubmit: object | None = None
    onChange: object | None = None
    addToHistory: object | None = None
    insertTextAtCursor: object | None = None
    getExpandedText: object | None = None
    setAutocompleteProvider: object | None = None
    borderColor: object | None = None
    setPaddingX: object | None = None
    setAutocompleteMaxVisible: object | None = None
    inputs: list[str] = field(default_factory=list)

    def render(self, _width: int) -> list[str]:
        return [self.text]

    def invalidate(self) -> None:
        return None

    def getText(self) -> str:
        return self.text

    def setText(self, text: str) -> None:
        self.text = text

    def handleInput(self, data: str) -> None:
        self.inputs.append(data)


def test_editor_component_module_exports_match_ts_surface() -> None:
    assert editor_component_module.__all__ == ["EditorComponent"]
    assert not hasattr(editor_component_module, "getExpandedText")
    assert not hasattr(editor_component_module, "get_expanded_text")
