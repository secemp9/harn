from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harnify_coding_agent.core.tools import (
    compute_edits_diff,
    create_edit_tool,
    create_edit_tool_definition,
    create_write_tool,
)


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _content_text(result: object) -> str:
    content = getattr(result, "content", [])
    return "\n".join(block.text for block in content if getattr(block, "type", None) == "text")


def test_edit_prepare_arguments_keeps_legacy_fields_out_of_schema() -> None:
    definition = create_edit_tool_definition(str(Path.cwd()))
    properties = definition.parameters.model_json_schema()["properties"]
    assert "oldText" not in properties
    assert "newText" not in properties


def test_edit_prepare_arguments_folds_top_level_legacy_fields() -> None:
    definition = create_edit_tool_definition(str(Path.cwd()))
    prepared = definition.prepareArguments({"path": "file.txt", "oldText": "before", "newText": "after"})
    assert prepared == {"path": "file.txt", "edits": [{"oldText": "before", "newText": "after"}]}


def test_edit_prepare_arguments_appends_legacy_replacement_to_existing_edits() -> None:
    definition = create_edit_tool_definition(str(Path.cwd()))
    prepared = definition.prepareArguments(
        {
            "path": "file.txt",
            "edits": [{"oldText": "a", "newText": "b"}],
            "oldText": "c",
            "newText": "d",
        }
    )
    assert prepared == {
        "path": "file.txt",
        "edits": [{"oldText": "a", "newText": "b"}, {"oldText": "c", "newText": "d"}],
    }


def test_edit_prepare_arguments_passes_through_valid_input_unchanged() -> None:
    definition = create_edit_tool_definition(str(Path.cwd()))
    input_value = {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}
    assert definition.prepareArguments(input_value) is input_value


def test_edit_prepare_arguments_parses_json_edits_string() -> None:
    definition = create_edit_tool_definition(str(Path.cwd()))
    prepared = definition.prepareArguments({"path": "file.txt", "edits": '[{"oldText":"a","newText":"b"}]'})
    assert prepared == {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}


@pytest.mark.asyncio
async def test_edit_tool_replaces_text_and_returns_diff_and_patch(tmp_path: Path) -> None:
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("Hello, world!", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    result = await tool.execute(
        "edit-1",
        {"path": str(test_file), "edits": [{"oldText": "world", "newText": "testing"}]},
        None,
        None,
    )

    assert "Successfully replaced 1 block(s)" in _content_text(result)
    assert result.details is not None
    assert "testing" in result.details.diff
    assert "--- " in result.details.patch
    assert "+++ " in result.details.patch
    assert "@@" in result.details.patch
    assert _read(test_file) == "Hello, testing!"


@pytest.mark.asyncio
async def test_edit_tool_fails_when_text_is_not_found(tmp_path: Path) -> None:
    test_file = tmp_path / "missing-text.txt"
    test_file.write_text("Hello, world!", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Could not find the exact text"):
        await tool.execute(
            "edit-2",
            {"path": str(test_file), "edits": [{"oldText": "nonexistent", "newText": "testing"}]},
            None,
            None,
        )


@pytest.mark.asyncio
async def test_edit_tool_includes_enoent_for_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    tool = create_edit_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=rf"Could not edit file: {missing}\. Error code: ENOENT\."):
        await tool.execute(
            "edit-3",
            {"path": str(missing), "edits": [{"oldText": "hello", "newText": "world"}]},
            None,
            None,
        )


@pytest.mark.asyncio
async def test_edit_tool_rejects_duplicate_matches(tmp_path: Path) -> None:
    test_file = tmp_path / "dups.txt"
    test_file.write_text("foo foo foo", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Found 3 occurrences"):
        await tool.execute(
            "edit-4",
            {"path": str(test_file), "edits": [{"oldText": "foo", "newText": "bar"}]},
            None,
            None,
        )


@pytest.mark.asyncio
async def test_edit_tool_supports_multiple_disjoint_replacements_and_collapsed_diff(tmp_path: Path) -> None:
    test_file = tmp_path / "multi-gap.txt"
    lines = [f"line {index:03d}" for index in range(1, 601)]
    test_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    result = await tool.execute(
        "edit-5",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "line 100\n", "newText": "LINE 100\n"},
                {"oldText": "line 300\n", "newText": "LINE 300\n"},
                {"oldText": "line 500\n", "newText": "LINE 500\n"},
            ],
        },
        None,
        None,
    )

    diff = result.details.diff
    assert "LINE 100" in diff
    assert "LINE 300" in diff
    assert "LINE 500" in diff
    assert "..." in diff
    assert "line 250" not in diff
    assert len(diff.splitlines()) < 50


@pytest.mark.asyncio
async def test_edit_tool_matches_against_original_file_not_incrementally(tmp_path: Path) -> None:
    test_file = tmp_path / "original-mode.txt"
    test_file.write_text("foo\nbar\nbaz\n", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    await tool.execute(
        "edit-6",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "foo\n", "newText": "foo bar\n"},
                {"oldText": "bar\n", "newText": "BAR\n"},
            ],
        },
        None,
        None,
    )

    assert _read(test_file) == "foo bar\nBAR\nbaz\n"


@pytest.mark.asyncio
async def test_edit_tool_rejects_empty_edits_and_overlaps(tmp_path: Path) -> None:
    test_file = tmp_path / "overlap.txt"
    test_file.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = create_edit_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"edits must contain at least one replacement"):
        await tool.execute("edit-7", {"path": str(test_file), "edits": []}, None, None)

    with pytest.raises(RuntimeError, match=r"overlap"):
        await tool.execute(
            "edit-8",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                    {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                ],
            },
            None,
            None,
        )


@pytest.mark.asyncio
async def test_edit_tool_does_not_partially_apply_when_one_edit_fails(tmp_path: Path) -> None:
    test_file = tmp_path / "no-partial.txt"
    original = "alpha\nbeta\ngamma\n"
    test_file.write_text(original, encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Could not find"):
        await tool.execute(
            "edit-9",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "alpha\n", "newText": "ALPHA\n"},
                    {"oldText": "missing\n", "newText": "MISSING\n"},
                ],
            },
            None,
            None,
        )

    assert _read(test_file) == original


@pytest.mark.asyncio
async def test_edit_tool_includes_generic_access_errors(tmp_path: Path) -> None:
    class Ops:
        async def access(self, _path: str) -> None:
            raise RuntimeError("disk offline")

        async def readFile(self, _path: str) -> bytes:
            return b"hello\n"

        async def writeFile(self, _path: str, _content: str) -> None:
            return None

    tool = create_edit_tool(str(tmp_path), {"operations": Ops()})
    with pytest.raises(RuntimeError, match=r"Could not edit file: broken.txt\. Error: disk offline\."):
        await tool.execute(
            "edit-10",
            {"path": "broken.txt", "edits": [{"oldText": "hello", "newText": "world"}]},
            None,
            None,
        )


@pytest.mark.asyncio
async def test_compute_edits_diff_reports_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing-preview.txt"
    result = await compute_edits_diff(str(missing), [{"oldText": "hello", "newText": "world"}], str(tmp_path))

    assert hasattr(result, "error")
    assert result.error == f"Could not edit file: {missing}. Error code: ENOENT."


@pytest.mark.asyncio
async def test_edit_tool_supports_fuzzy_matching_modes(tmp_path: Path) -> None:
    test_file = tmp_path / "fuzzy.txt"
    test_file.write_text("console.log(\u2018hello\u2019);\nhello\u00a0world\nＡＢＣ１２３\n", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    await tool.execute(
        "edit-11",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "console.log('hello');\n", "newText": "console.log('world');\n"},
                {"oldText": "hello world\n", "newText": "hello universe\n"},
                {"oldText": "ABC123\n", "newText": "XYZ789\n"},
            ],
        },
        None,
        None,
    )

    assert _read(test_file) == "console.log('world');\nhello universe\nXYZ789\n"


@pytest.mark.asyncio
async def test_edit_tool_rejects_duplicates_after_fuzzy_normalization(tmp_path: Path) -> None:
    test_file = tmp_path / "fuzzy-dups.txt"
    test_file.write_text("hello world   \nhello world\n", encoding="utf-8")

    tool = create_edit_tool(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Found 2 occurrences"):
        await tool.execute(
            "edit-12",
            {"path": str(test_file), "edits": [{"oldText": "hello world", "newText": "replaced"}]},
            None,
            None,
        )


@pytest.mark.asyncio
async def test_edit_tool_preserves_crlf_and_bom(tmp_path: Path) -> None:
    test_file = tmp_path / "bom-crlf.txt"
    test_file.write_text("\ufefffirst\r\nsecond\r\nthird\r\nfourth\r\n", encoding="utf-8", newline="")

    tool = create_edit_tool(str(tmp_path))
    await tool.execute(
        "edit-13",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "second\n", "newText": "SECOND\n"},
                {"oldText": "fourth\n", "newText": "FOURTH\n"},
            ],
        },
        None,
        None,
    )

    assert _read(test_file) == "\ufefffirst\r\nSECOND\r\nthird\r\nFOURTH\r\n"


@pytest.mark.asyncio
async def test_edit_and_write_share_the_same_mutation_queue(tmp_path: Path) -> None:
    file_path = tmp_path / "mixed.txt"
    file_path.write_text("original\n", encoding="utf-8")

    async def delay(ms: int) -> None:
        await asyncio.sleep(ms / 1000)

    class EditOps:
        async def access(self, _path: str) -> None:
            return None

        async def readFile(self, path: str) -> bytes:
            data = Path(path).read_bytes()
            await delay(30)
            return data

        async def writeFile(self, path: str, content: str) -> None:
            await delay(30)
            Path(path).write_text(content, encoding="utf-8")

    class WriteOps:
        async def mkdir(self, _directory: str) -> None:
            return None

        async def writeFile(self, path: str, content: str) -> None:
            await delay(10)
            Path(path).write_text(content, encoding="utf-8")

    edit_tool = create_edit_tool(str(tmp_path), {"operations": EditOps()})
    write_tool = create_write_tool(str(tmp_path), {"operations": WriteOps()})

    edit_task = asyncio.create_task(
        edit_tool.execute(
            "edit-14",
            {"path": str(file_path), "edits": [{"oldText": "original", "newText": "edited"}]},
            None,
            None,
        )
    )
    await delay(5)
    write_task = asyncio.create_task(
        write_tool.execute("write-1", {"path": str(file_path), "content": "replacement\n"}, None, None)
    )

    await asyncio.gather(edit_task, write_task)
    assert _read(file_path) == "replacement\n"
