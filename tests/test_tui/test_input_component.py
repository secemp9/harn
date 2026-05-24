from __future__ import annotations

from harnify_tui.components.input import Input
from harnify_tui.utils import visibleWidth


def _type_text(input_component: Input, text: str) -> None:
    for char in text:
        input_component.handleInput(char)


def _move_right(input_component: Input, count: int) -> None:
    for _ in range(count):
        input_component.handleInput("\x1b[C")


def test_input_submits_value_including_backslash_on_enter() -> None:
    input_component = Input()
    submitted: str | None = None

    def on_submit(value: str) -> None:
        nonlocal submitted
        submitted = value

    input_component.onSubmit = on_submit
    _type_text(input_component, "hello")
    input_component.handleInput("\\")
    input_component.handleInput("\r")

    assert submitted == "hello\\"


def test_input_inserts_backslash_as_regular_character() -> None:
    input_component = Input()

    input_component.handleInput("\\")
    input_component.handleInput("x")

    assert input_component.getValue() == "\\x"


def test_input_render_does_not_overflow_with_wide_cjk_and_fullwidth_text() -> None:
    width = 93
    cases = [
        (
            "가나다라마바사아자차카타파하 한글 텍스트가 터미널 너비를 초과하면 "
            "크래시가 발생합니다 이것은 재현용 테스트입니다"
        ),
        (
            "これはテスト文章です。日本語のテキストが正しく表示されるかどうかを"
            "確認するためのサンプルテキストです。あいうえお"
        ),
        (
            "这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，"
            "如果不正确就会导致用户界面崩溃的问题"
        ),
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ",
    ]

    for text in cases:
        for label, move in (
            ("start", lambda component: None),
            ("middle", lambda component: _move_right(component, 10)),
            ("end", lambda component: component.handleInput("\x05")),
        ):
            input_component = Input()
            input_component.setValue(text)
            input_component.focused = True
            move(input_component)

            line = input_component.render(width)[0]

            assert line, label
            assert visibleWidth(line) <= width, f"rendered line overflowed at {label}: {text}"


def test_input_render_keeps_cursor_visible_when_horizontally_scrolling_wide_text() -> None:
    input_component = Input()
    input_component.setValue("가나다라마바사아자차카타파하")
    input_component.focused = True
    input_component.handleInput("\x01")
    _move_right(input_component, 5)

    line = input_component.render(20)[0]

    assert line
    assert visibleWidth(line) <= 20


def test_input_ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it() -> None:
    input_component = Input()
    input_component.setValue("foo bar baz")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")

    assert input_component.getValue() == "foo bar "

    input_component.handleInput("\x01")
    input_component.handleInput("\x19")

    assert input_component.getValue() == "bazfoo bar "


def test_input_ctrl_u_saves_deleted_text_to_kill_ring() -> None:
    input_component = Input()
    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x15")

    assert input_component.getValue() == "world"

    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello world"


def test_input_ctrl_k_saves_deleted_text_to_kill_ring() -> None:
    input_component = Input()
    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    input_component.handleInput("\x0b")

    assert input_component.getValue() == ""

    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello world"


def test_input_ctrl_y_does_nothing_when_kill_ring_is_empty() -> None:
    input_component = Input()
    input_component.setValue("test")
    input_component.handleInput("\x05")
    input_component.handleInput("\x19")

    assert input_component.getValue() == "test"


def test_input_alt_y_cycles_through_kill_ring_after_ctrl_y() -> None:
    input_component = Input()

    for value in ("first", "second", "third"):
        input_component.setValue(value)
        input_component.handleInput("\x05")
        input_component.handleInput("\x17")

    assert input_component.getValue() == ""

    input_component.handleInput("\x19")
    assert input_component.getValue() == "third"

    input_component.handleInput("\x1by")
    assert input_component.getValue() == "second"

    input_component.handleInput("\x1by")
    assert input_component.getValue() == "first"

    input_component.handleInput("\x1by")
    assert input_component.getValue() == "third"


def test_input_alt_y_does_nothing_if_not_preceded_by_yank() -> None:
    input_component = Input()
    input_component.setValue("test")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.setValue("other")
    input_component.handleInput("\x05")
    input_component.handleInput("x")

    assert input_component.getValue() == "otherx"

    input_component.handleInput("\x1by")

    assert input_component.getValue() == "otherx"


def test_input_alt_y_does_nothing_if_kill_ring_has_one_entry() -> None:
    input_component = Input()
    input_component.setValue("only")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.handleInput("\x19")

    assert input_component.getValue() == "only"

    input_component.handleInput("\x1by")

    assert input_component.getValue() == "only"


def test_input_consecutive_ctrl_w_accumulates_into_one_kill_ring_entry() -> None:
    input_component = Input()
    input_component.setValue("one two three")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.handleInput("\x17")
    input_component.handleInput("\x17")

    assert input_component.getValue() == ""

    input_component.handleInput("\x19")

    assert input_component.getValue() == "one two three"


def test_input_non_delete_actions_break_kill_accumulation() -> None:
    input_component = Input()
    input_component.setValue("foo bar baz")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")

    assert input_component.getValue() == "foo bar "

    input_component.handleInput("x")
    assert input_component.getValue() == "foo bar x"

    input_component.handleInput("\x17")
    assert input_component.getValue() == "foo bar "

    input_component.handleInput("\x19")
    assert input_component.getValue() == "foo bar x"

    input_component.handleInput("\x1by")
    assert input_component.getValue() == "foo bar baz"


def test_input_non_yank_actions_break_alt_y_chain() -> None:
    input_component = Input()

    input_component.setValue("first")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.setValue("second")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.setValue("")

    input_component.handleInput("\x19")
    assert input_component.getValue() == "second"

    input_component.handleInput("x")
    assert input_component.getValue() == "secondx"

    input_component.handleInput("\x1by")
    assert input_component.getValue() == "secondx"


def test_input_kill_ring_rotation_persists_after_cycling() -> None:
    input_component = Input()

    for value in ("first", "second", "third"):
        input_component.setValue(value)
        input_component.handleInput("\x05")
        input_component.handleInput("\x17")
    input_component.setValue("")

    input_component.handleInput("\x19")
    input_component.handleInput("\x1by")
    assert input_component.getValue() == "second"

    input_component.handleInput("x")
    input_component.setValue("")
    input_component.handleInput("\x19")

    assert input_component.getValue() == "second"


def test_input_backward_deletions_prepend_and_forward_deletions_append_during_accumulation() -> None:
    input_component = Input()
    input_component.setValue("prefix|suffix")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x0b")

    assert input_component.getValue() == "prefix"

    input_component.handleInput("\x19")

    assert input_component.getValue() == "prefix|suffix"


def test_input_alt_d_deletes_word_forward_and_saves_to_kill_ring() -> None:
    input_component = Input()
    input_component.setValue("hello world test")
    input_component.handleInput("\x01")
    input_component.handleInput("\x1bd")

    assert input_component.getValue() == " world test"

    input_component.handleInput("\x1bd")
    assert input_component.getValue() == " test"

    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello world test"


def test_input_handles_yank_in_middle_of_text() -> None:
    input_component = Input()
    input_component.setValue("word")
    input_component.handleInput("\x05")
    input_component.handleInput("\x17")
    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello wordworld"


def test_input_handles_yank_pop_in_middle_of_text() -> None:
    input_component = Input()

    for value in ("FIRST", "SECOND"):
        input_component.setValue(value)
        input_component.handleInput("\x05")
        input_component.handleInput("\x17")

    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello SECONDworld"

    input_component.handleInput("\x1by")

    assert input_component.getValue() == "hello FIRSTworld"


def test_input_undo_does_nothing_when_stack_is_empty() -> None:
    input_component = Input()
    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == ""


def test_input_undo_coalesces_consecutive_word_characters_into_one_unit() -> None:
    input_component = Input()
    _type_text(input_component, "hello world")

    assert input_component.getValue() == "hello world"

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == "hello"

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == ""


def test_input_undoes_spaces_one_at_a_time() -> None:
    input_component = Input()
    _type_text(input_component, "hello")
    input_component.handleInput(" ")
    input_component.handleInput(" ")

    assert input_component.getValue() == "hello  "

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == "hello "

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == "hello"

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == ""


def test_input_undoes_backspace() -> None:
    input_component = Input()
    _type_text(input_component, "hello")
    input_component.handleInput("\x7f")

    assert input_component.getValue() == "hell"

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello"


def test_input_undoes_forward_delete() -> None:
    input_component = Input()
    _type_text(input_component, "hello")
    input_component.handleInput("\x01")
    input_component.handleInput("\x1b[C")
    input_component.handleInput("\x1b[3~")

    assert input_component.getValue() == "hllo"

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello"


def test_input_undoes_ctrl_w() -> None:
    input_component = Input()
    _type_text(input_component, "hello world")
    input_component.handleInput("\x17")

    assert input_component.getValue() == "hello "

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello world"


def test_input_undoes_ctrl_k() -> None:
    input_component = Input()
    _type_text(input_component, "hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x0b")

    assert input_component.getValue() == "hello "

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello world"


def test_input_undoes_ctrl_u() -> None:
    input_component = Input()
    _type_text(input_component, "hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 6)
    input_component.handleInput("\x15")

    assert input_component.getValue() == "world"

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello world"


def test_input_undoes_yank() -> None:
    input_component = Input()
    _type_text(input_component, "hello ")
    input_component.handleInput("\x17")
    input_component.handleInput("\x19")

    assert input_component.getValue() == "hello "

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == ""


def test_input_undoes_paste_atomically() -> None:
    input_component = Input()
    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    _move_right(input_component, 5)
    input_component.handleInput("\x1b[200~beep boop\x1b[201~")

    assert input_component.getValue() == "hellobeep boop world"

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello world"


def test_input_undoes_alt_d() -> None:
    input_component = Input()
    input_component.setValue("hello world")
    input_component.handleInput("\x01")
    input_component.handleInput("\x1bd")

    assert input_component.getValue() == " world"

    input_component.handleInput("\x1b[45;5u")

    assert input_component.getValue() == "hello world"


def test_input_cursor_movement_starts_new_undo_unit() -> None:
    input_component = Input()
    _type_text(input_component, "abc")
    input_component.handleInput("\x01")
    input_component.handleInput("\x05")
    _type_text(input_component, "de")

    assert input_component.getValue() == "abcde"

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == "abc"

    input_component.handleInput("\x1b[45;5u")
    assert input_component.getValue() == ""
