from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from harnify_coding_agent.core.tools import (
    GREP_MAX_LINE_LENGTH,
    create_find_tool,
    create_find_tool_definition,
    create_grep_tool,
    create_grep_tool_definition,
    create_ls_tool,
    get_text_output,
)
from harnify_coding_agent.core.tools import find as find_module
from harnify_coding_agent.core.tools import grep as grep_module


def _text(result: object) -> str:
    return get_text_output(result, show_images=True)


def _lines(result: object) -> list[str]:
    return [line.strip() for line in _text(result).splitlines() if line.strip() and not line.startswith("[")]


@pytest.fixture(autouse=True)
def _patch_find_ensure_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fd_script = tmp_path / "fake-fd"
    fd_script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            from __future__ import annotations

            import os
            import sys

            from harnify_coding_agent.core.tools.find import _glob_files


            def main(argv: list[str]) -> int:
                limit = 1000
                pattern = ""
                search_path = "."
                index = 0
                while index < len(argv):
                    arg = argv[index]
                    if arg == "--max-results":
                        limit = int(argv[index + 1])
                        index += 2
                        continue
                    if arg == "--":
                        pattern = argv[index + 1]
                        search_path = argv[index + 2]
                        break
                    index += 1

                try:
                    results = _glob_files(pattern, search_path, limit=limit)
                except Exception as error:
                    print(str(error), file=sys.stderr)
                    return 1

                for relative_path in sorted(results, key=str.lower):
                    print(os.path.join(search_path, relative_path.replace("/", os.sep)))
                return 0


            if __name__ == "__main__":
                raise SystemExit(main(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    fd_script.chmod(0o755)

    async def fake_ensure_tool(tool: str, silent: bool = False, **_kwargs: object) -> str:
        assert tool == "fd"
        assert silent is True
        return str(fd_script)

    monkeypatch.setattr("harnify_coding_agent.core.tools.find.ensure_tool", fake_ensure_tool)


def test_find_tool_definition_surface_matches_ts(tmp_path: Path) -> None:
    definition = create_find_tool_definition(str(tmp_path))

    assert definition.promptSnippet == "Find files by glob pattern (respects .gitignore)"
    assert definition.renderCall is not None
    assert definition.renderResult is not None
    assert definition.description.endswith("1000 results or 50KB (whichever is hit first).")


def test_find_module_exports_match_ts_surface() -> None:
    assert find_module.__all__ == [
        "FindOperations",
        "FindToolDetails",
        "FindToolInput",
        "FindToolOptions",
        "createFindTool",
        "createFindToolDefinition",
    ]


def test_grep_tool_definition_surface_matches_ts(tmp_path: Path) -> None:
    definition = create_grep_tool_definition(str(tmp_path))

    assert definition.promptSnippet == "Search file contents for patterns (respects .gitignore)"
    assert definition.renderCall is not None
    assert definition.renderResult is not None
    assert definition.description.endswith(f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars.")


def test_grep_module_exports_match_ts_surface() -> None:
    assert grep_module.__all__ == [
        "GrepOperations",
        "GrepToolDetails",
        "GrepToolInput",
        "GrepToolOptions",
        "createGrepTool",
        "createGrepToolDefinition",
    ]


@pytest.mark.asyncio
async def test_grep_tool_includes_filename_when_searching_single_file(tmp_path: Path) -> None:
    test_file = tmp_path / "example.txt"
    test_file.write_text("first line\nmatch line\nlast line", encoding="utf-8")

    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("grep-1", {"pattern": "match", "path": str(test_file)}, None, None)

    assert "example.txt:2: match line" in _text(result)


@pytest.mark.asyncio
async def test_grep_tool_respects_global_limit_and_includes_context_lines(tmp_path: Path) -> None:
    test_file = tmp_path / "context.txt"
    test_file.write_text("before\nmatch one\nafter\nmiddle\nmatch two\nafter two", encoding="utf-8")

    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute(
        "grep-2",
        {"pattern": "match", "path": str(test_file), "limit": 1, "context": 1},
        None,
        None,
    )

    output = _text(result)
    assert "context.txt-1- before" in output
    assert "context.txt:2: match one" in output
    assert "context.txt-3- after" in output
    assert "[1 matches limit reached. Use limit=2 for more, or refine pattern]" in output
    assert "match two" not in output
    assert result.details is not None
    assert result.details.matchLimitReached == 1


@pytest.mark.asyncio
async def test_grep_tool_treats_flag_like_patterns_as_search_text(tmp_path: Path) -> None:
    payload = tmp_path / "payload.sh"
    marker = tmp_path / "grep-injection-marker"
    test_file = tmp_path / "target.txt"
    payload.write_text(f"#!/bin/sh\necho executed > {marker}\ncat \"$1\"\n", encoding="utf-8")
    payload.chmod(0o755)
    test_file.write_text("target\n", encoding="utf-8")

    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("grep-3", {"pattern": f"--pre={payload}", "path": str(tmp_path)}, None, None)

    assert _text(result) == "No matches found"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_grep_tool_reports_line_truncation(tmp_path: Path) -> None:
    test_file = tmp_path / "long-line.txt"
    long_line = "match " + ("x" * (GREP_MAX_LINE_LENGTH + 50))
    test_file.write_text(long_line + "\n", encoding="utf-8")

    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("grep-4", {"pattern": "match", "path": str(test_file)}, None, None)

    output = _text(result)
    assert "... [truncated]" in output
    assert f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars" in output
    assert result.details is not None
    assert result.details.linesTruncated is True


@pytest.mark.asyncio
async def test_find_tool_includes_hidden_files_that_are_not_gitignored(tmp_path: Path) -> None:
    hidden_dir = tmp_path / ".secret"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-1", {"pattern": "**/*.txt", "path": str(tmp_path)}, None, None)

    assert "visible.txt" in _lines(result)
    assert ".secret/hidden.txt" in _lines(result)


@pytest.mark.asyncio
async def test_find_tool_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("kept", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-2", {"pattern": "**/*.txt", "path": str(tmp_path)}, None, None)

    lines = _lines(result)
    assert "kept.txt" in lines
    assert "ignored.txt" not in lines


@pytest.mark.asyncio
async def test_find_tool_surfaces_glob_parse_errors(tmp_path: Path) -> None:
    tool = create_find_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"error parsing glob"):
        await tool.execute("find-3", {"pattern": "[", "path": str(tmp_path)}, None, None)


@pytest.mark.asyncio
async def test_find_tool_treats_flag_like_patterns_as_search_text(tmp_path: Path) -> None:
    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-4", {"pattern": "--help", "path": str(tmp_path)}, None, None)

    assert _text(result) == "No files found matching pattern"


@pytest.mark.asyncio
async def test_find_tool_supports_path_based_glob_regressions(tmp_path: Path) -> None:
    (tmp_path / "some" / "parent" / "child").mkdir(parents=True)
    (tmp_path / "src" / "foo" / "bar").mkdir(parents=True)
    (tmp_path / "some" / "parent" / "child" / "file.ext").write_text("", encoding="utf-8")
    (tmp_path / "some" / "parent" / "child" / "test.spec.ts").write_text("", encoding="utf-8")
    (tmp_path / "src" / "foo" / "bar" / "example.spec.ts").write_text("", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))

    basename_result = await tool.execute("find-5", {"pattern": "*.spec.ts", "path": str(tmp_path)}, None, None)
    assert sorted(_lines(basename_result)) == ["some/parent/child/test.spec.ts", "src/foo/bar/example.spec.ts"]

    subtree_result = await tool.execute(
        "find-6",
        {"pattern": "some/parent/child/**", "path": str(tmp_path)},
        None,
        None,
    )
    subtree_lines = _lines(subtree_result)
    assert "some/parent/child/file.ext" in subtree_lines
    assert "some/parent/child/test.spec.ts" in subtree_lines

    wildcard_result = await tool.execute(
        "find-7",
        {"pattern": "**/parent/child/*", "path": str(tmp_path)},
        None,
        None,
    )
    wildcard_lines = _lines(wildcard_result)
    assert "some/parent/child/file.ext" in wildcard_lines
    assert "some/parent/child/test.spec.ts" in wildcard_lines

    src_result = await tool.execute(
        "find-8",
        {"pattern": "src/**/*.spec.ts", "path": str(tmp_path)},
        None,
        None,
    )
    assert _lines(src_result) == ["src/foo/bar/example.spec.ts"]


@pytest.mark.asyncio
async def test_find_tool_scopes_nested_gitignore_rules_to_their_subtrees(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "a" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "root.txt").write_text("", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-9", {"pattern": "**/*.txt", "path": str(tmp_path)}, None, None)

    assert _lines(result) == ["a/kept.txt", "b/ignored.txt", "b/kept.txt", "root.txt"]


@pytest.mark.asyncio
async def test_find_tool_scopes_deep_nested_gitignore_rules(tmp_path: Path) -> None:
    (tmp_path / "a" / "deep").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "a" / "deep" / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (tmp_path / "a" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "secret.txt").write_text("", encoding="utf-8")
    (tmp_path / "a" / "deep" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "root.txt").write_text("", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-10", {"pattern": "**/*.txt", "path": str(tmp_path)}, None, None)

    assert _lines(result) == ["a/deep/kept.txt", "a/kept.txt", "b/ignored.txt", "b/kept.txt", "root.txt"]


@pytest.mark.asyncio
async def test_find_tool_reports_result_limit_details(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file-{index}.txt").write_text("", encoding="utf-8")

    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("find-11", {"pattern": "*.txt", "path": str(tmp_path), "limit": 2}, None, None)

    output = _text(result)
    assert "[2 results limit reached. Use limit=4 for more, or refine pattern]" in output
    assert result.details is not None
    assert result.details.resultLimitReached == 2


@pytest.mark.asyncio
async def test_ls_tool_lists_dotfiles_and_directories(tmp_path: Path) -> None:
    (tmp_path / ".hidden-file").write_text("secret", encoding="utf-8")
    (tmp_path / ".hidden-dir").mkdir()

    tool = create_ls_tool(str(tmp_path))
    result = await tool.execute("ls-1", {"path": str(tmp_path)}, None, None)

    lines = _lines(result)
    assert ".hidden-file" in lines
    assert ".hidden-dir/" in lines


@pytest.mark.asyncio
async def test_ls_tool_reports_entry_limit_details(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"entry-{index}.txt").write_text("", encoding="utf-8")

    tool = create_ls_tool(str(tmp_path))
    result = await tool.execute("ls-2", {"path": str(tmp_path), "limit": 2}, None, None)

    output = _text(result)
    assert "[2 entries limit reached. Use limit=4 for more]" in output
    assert result.details is not None
    assert result.details.entryLimitReached == 2
