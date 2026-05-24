from __future__ import annotations

import asyncio
import os
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest
from harnify_agent.types import AgentToolResult
from harnify_ai.types import ImageContent, TextContent
from harnify_coding_agent.core.extensions.types import ToolDefinition
from harnify_coding_agent.core.tools import (
    OutputAccumulator,
    OutputAccumulatorOptions,
    all_tool_names,
    create_all_tool_definitions,
    create_all_tools,
    create_coding_tool_definitions,
    create_coding_tools,
    create_read_only_tool_definitions,
    create_read_only_tools,
    create_tool,
    create_tool_definition,
    expand_path,
    format_size,
    get_text_output,
    resolve_read_path,
    resolve_to_cwd,
    truncate_head,
    truncate_line,
    truncate_tail,
    with_file_mutation_queue,
    wrap_tool_definition,
)


def test_truncate_head_honors_line_and_byte_limits() -> None:
    result = truncate_head("alpha\nbeta\ngamma", options=SimpleNamespace(maxLines=2, maxBytes=100))

    assert result.content == "alpha\nbeta"
    assert result.truncated is True
    assert result.truncatedBy == "lines"
    assert result.outputLines == 2


def test_truncate_tail_can_return_partial_last_line() -> None:
    result = truncate_tail("abcdefghijklmnopqrstuvwxyz", options=SimpleNamespace(maxLines=10, maxBytes=8))

    assert result.content.endswith("stuvwxyz")
    assert result.truncated is True
    assert result.truncatedBy == "bytes"
    assert result.lastLinePartial is True


def test_truncate_line_and_format_size() -> None:
    assert truncate_line("abc", 5) == {"text": "abc", "wasTruncated": False}
    assert truncate_line("abcdef", 3) == {"text": "abc... [truncated]", "wasTruncated": True}
    assert format_size(999) == "999B"
    assert format_size(2048) == "2.0KB"


def test_expand_and_resolve_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    assert expand_path(f"@{nested.name}") == nested.name
    assert resolve_to_cwd("./nested", str(tmp_path)) == str(nested)


def test_resolve_read_path_handles_curly_quote_and_nfd_variants(tmp_path: Path) -> None:
    curly = tmp_path / "Capture d’écran.png"
    curly.write_text("x", encoding="utf-8")
    assert resolve_read_path("Capture d'ecran.png".replace("ecran", "écran"), str(tmp_path)) == str(curly)

    nfd_name = unicodedata.normalize("NFD", "résumé.txt")
    nfd_path = tmp_path / nfd_name
    nfd_path.write_text("resume", encoding="utf-8")
    assert resolve_read_path("résumé.txt", str(tmp_path)) == str(nfd_path)


def test_get_text_output_strips_ansi_and_renders_hidden_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harnify_coding_agent.core.tools.render_utils.get_capabilities",
        lambda: SimpleNamespace(images=False),
    )
    monkeypatch.setattr(
        "harnify_coding_agent.core.tools.render_utils.image_fallback",
        lambda mime, dims: f"[image {mime}]",
    )
    monkeypatch.setattr(
        "harnify_coding_agent.core.tools.render_utils.get_image_dimensions",
        lambda data, mime: None,
    )

    result = SimpleNamespace(
        content=[
            TextContent(text="\x1b[31mhello\r\nworld\x1b[0m"),
            ImageContent(data="abc", mimeType="image/png"),
        ]
    )

    assert get_text_output(result, show_images=False) == "hello\nworld\n[image image/png]"


@pytest.mark.asyncio
async def test_output_accumulator_persists_full_output_when_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "harnify_coding_agent.core.tools.output_accumulator.default_temp_file_path",
        lambda prefix: str(tmp_path / f"{prefix}.log"),
    )
    accumulator = OutputAccumulator(OutputAccumulatorOptions(maxLines=2, maxBytes=12, tempFilePrefix="shared"))
    accumulator.append(b"line1\nline2\nline3\n")
    accumulator.finish()

    snapshot = accumulator.snapshot(persistIfTruncated=True)
    await accumulator.close_temp_file()

    assert snapshot.truncation.truncated is True
    assert snapshot.fullOutputPath == str(tmp_path / "shared.log")
    assert Path(snapshot.fullOutputPath).read_text(encoding="utf-8") == "line1\nline2\nline3\n"


@pytest.mark.asyncio
async def test_file_mutation_queue_serializes_same_file() -> None:
    order: list[str] = []
    release_first = asyncio.Event()

    async def first() -> str:
        order.append("first-start")
        await release_first.wait()
        order.append("first-end")
        return "first"

    async def second() -> str:
        order.append("second-start")
        order.append("second-end")
        return "second"

    task1 = asyncio.create_task(with_file_mutation_queue("demo.txt", first))
    await asyncio.sleep(0)
    task2 = asyncio.create_task(with_file_mutation_queue("demo.txt", second))
    await asyncio.sleep(0)

    assert order == ["first-start"]

    release_first.set()
    assert await task1 == "first"
    assert await task2 == "second"
    assert order == ["first-start", "first-end", "second-start", "second-end"]


@pytest.mark.asyncio
async def test_wrap_tool_definition_passes_extension_context() -> None:
    seen: dict[str, object] = {}

    async def execute(
        tool_call_id: str,
        params: object,
        signal: object,
        on_update: object,
        ctx: object,
    ) -> AgentToolResult:
        seen.update(
            tool_call_id=tool_call_id,
            params=params,
            signal=signal,
            on_update=on_update,
            ctx=ctx,
        )
        return AgentToolResult(content=[TextContent(text="ok")], details={"done": True})

    definition = ToolDefinition(
        name="demo",
        label="Demo",
        description="desc",
        parameters={"type": "object"},
        execute=execute,
    )

    wrapped = wrap_tool_definition(definition, lambda: {"cwd": os.getcwd()})
    result = await wrapped.execute("call-1", {"value": 1}, None, None)

    assert wrapped.name == "demo"
    assert wrapped.label == "Demo"
    assert seen["ctx"] == {"cwd": os.getcwd()}
    assert result.details == {"done": True}


def test_tool_index_factories_preserve_upstream_tool_sets(tmp_path: Path) -> None:
    options = {
        "bash": {"commandPrefix": "printf ready && "},
        "read": {"showLineNumbers": False},
        "grep": {"maxResults": 5},
        "find": {"limit": 7},
        "ls": {"limit": 3},
    }

    assert all_tool_names == {"read", "bash", "edit", "write", "grep", "find", "ls"}

    coding_defs = create_coding_tool_definitions(str(tmp_path), options)
    readonly_defs = create_read_only_tool_definitions(str(tmp_path), options)
    all_defs = create_all_tool_definitions(str(tmp_path), options)
    coding_tools = create_coding_tools(str(tmp_path), options)
    readonly_tools = create_read_only_tools(str(tmp_path), options)
    all_tools = create_all_tools(str(tmp_path), options)

    assert [tool.name for tool in coding_defs] == ["read", "bash", "edit", "write"]
    assert [tool.name for tool in readonly_defs] == ["read", "grep", "find", "ls"]
    assert list(all_defs) == ["read", "bash", "edit", "write", "grep", "find", "ls"]
    assert [tool.name for tool in coding_tools] == ["read", "bash", "edit", "write"]
    assert [tool.name for tool in readonly_tools] == ["read", "grep", "find", "ls"]
    assert list(all_tools) == ["read", "bash", "edit", "write", "grep", "find", "ls"]
    assert create_tool_definition("bash", str(tmp_path), options).name == "bash"
    assert create_tool("grep", str(tmp_path), options).name == "grep"


def test_tool_index_rejects_unknown_tool_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown tool name: unknown"):
        create_tool_definition("unknown", str(tmp_path))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown tool name: unknown"):
        create_tool("unknown", str(tmp_path))  # type: ignore[arg-type]
