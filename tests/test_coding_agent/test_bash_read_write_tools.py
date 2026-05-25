from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from harnify_coding_agent.core.tools import (
    create_bash_tool,
    create_bash_tool_definition,
    create_local_bash_operations,
    create_read_tool,
    create_write_tool,
    get_text_output,
)
from harnify_coding_agent.core.tools import bash as bash_module

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


def _text(result: object) -> str:
    return get_text_output(result, show_images=True)


@pytest.mark.asyncio
async def test_read_tool_reads_text_file_without_truncation(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, world!\nLine 2\nLine 3", encoding="utf-8")

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-1", {"path": str(test_file)}, None, None)

    assert _text(result) == "Hello, world!\nLine 2\nLine 3"
    assert result.details is None


@pytest.mark.asyncio
async def test_read_tool_truncates_large_text_file_by_line_count(tmp_path: Path) -> None:
    test_file = tmp_path / "large.txt"
    test_file.write_text("\n".join(f"Line {index}" for index in range(1, 2501)), encoding="utf-8")

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-2", {"path": str(test_file)}, None, None)

    output = _text(result)
    assert "Line 1" in output
    assert "Line 2000" in output
    assert "Line 2001" not in output
    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in output
    assert result.details is not None
    assert result.details.truncation is not None
    assert result.details.truncation.truncatedBy == "lines"


@pytest.mark.asyncio
async def test_read_tool_honors_offset_and_limit(tmp_path: Path) -> None:
    test_file = tmp_path / "offset-limit.txt"
    test_file.write_text("\n".join(f"Line {index}" for index in range(1, 101)), encoding="utf-8")

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-3", {"path": str(test_file), "offset": 41, "limit": 20}, None, None)

    output = _text(result)
    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "[40 more lines in file. Use offset=61 to continue.]" in output


@pytest.mark.asyncio
async def test_read_tool_rejects_offset_beyond_end_of_file(tmp_path: Path) -> None:
    test_file = tmp_path / "short.txt"
    test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

    tool = create_read_tool(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Offset 100 is beyond end of file \(3 lines total\)"):
        await tool.execute("read-4", {"path": str(test_file), "offset": 100}, None, None)


@pytest.mark.asyncio
async def test_read_tool_detects_images_by_magic_instead_of_extension(tmp_path: Path) -> None:
    image_file = tmp_path / "image.txt"
    image_file.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-5", {"path": str(image_file)}, None, None)

    assert result.content[0].type == "text"
    assert "Read image file [image/png]" in _text(result)
    image_block = next(block for block in result.content if block.type == "image")
    assert image_block.mimeType == "image/png"
    assert image_block.data


@pytest.mark.asyncio
async def test_read_tool_treats_fake_image_files_as_text(tmp_path: Path) -> None:
    fake_image = tmp_path / "not-an-image.png"
    fake_image.write_text("definitely not a png", encoding="utf-8")

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-6", {"path": str(fake_image)}, None, None)

    assert "definitely not a png" in _text(result)
    assert all(block.type != "image" for block in result.content)


@pytest.mark.asyncio
async def test_read_tool_omits_image_when_auto_resize_cannot_make_it_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resize(_img, _options=None):
        return None

    monkeypatch.setattr("harnify_coding_agent.core.tools.read.resize_image", fake_resize)
    image_file = tmp_path / "image.png"
    image_file.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("read-7", {"path": str(image_file)}, None, None)

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "Image omitted" in _text(result)


@pytest.mark.asyncio
async def test_write_tool_creates_parent_directories_and_writes_content(tmp_path: Path) -> None:
    tool = create_write_tool(str(tmp_path))
    target = tmp_path / "nested" / "dir" / "test.txt"

    result = await tool.execute("write-1", {"path": "nested/dir/test.txt", "content": "Nested content"}, None, None)

    assert "Successfully wrote" in _text(result)
    assert target.read_text(encoding="utf-8") == "Nested content"


@pytest.mark.asyncio
async def test_bash_tool_executes_real_commands(tmp_path: Path) -> None:
    tool = create_bash_tool(str(tmp_path))

    result = await tool.execute("bash-1", {"command": "printf 'test output\\n'"}, None, None)

    assert _text(result).strip() == "test output"
    assert result.details is None


@pytest.mark.asyncio
async def test_bash_tool_handles_nonzero_exit_codes(tmp_path: Path) -> None:
    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"Command exited with code 3"):
        await tool.execute("bash-2", {"command": "printf 'boom\\n'; exit 3"}, None, None)


@pytest.mark.asyncio
async def test_bash_tool_respects_timeout(tmp_path: Path) -> None:
    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"Command timed out"):
        await tool.execute("bash-3", {"command": "sleep 2", "timeout": 0.1}, None, None)


@pytest.mark.asyncio
async def test_bash_tool_supports_command_prefix(tmp_path: Path) -> None:
    tool = create_bash_tool(str(tmp_path), {"commandPrefix": "export TEST_VAR=hello"})

    result = await tool.execute("bash-4", {"command": "printf '%s' \"$TEST_VAR\""}, None, None)

    assert _text(result) == "hello"


@pytest.mark.asyncio
async def test_bash_tool_coalesces_streaming_updates_for_chatty_output(tmp_path: Path) -> None:
    class Ops:
        async def exec(self, command: str, cwd: str, options) -> dict[str, int | None]:
            assert command == "chatty"
            assert cwd == str(tmp_path)
            for index in range(5000):
                options["onData"](f"line {index}\n".encode())
            await asyncio.sleep(0)
            return {"exitCode": 0}

    updates = []
    tool = create_bash_tool(str(tmp_path), {"operations": Ops()})

    result = await tool.execute("bash-5", {"command": "chatty"}, None, updates.append)

    assert len(updates) < 25
    assert "line 4999" in _text(result)


@pytest.mark.asyncio
async def test_bash_tool_decodes_split_utf8_output(tmp_path: Path) -> None:
    euro = "€\n".encode()

    class Ops:
        async def exec(self, _command: str, _cwd: str, options) -> dict[str, int | None]:
            options["onData"](euro[:1])
            options["onData"](euro[1:])
            return {"exitCode": 0}

    tool = create_bash_tool(str(tmp_path), {"operations": Ops()})
    result = await tool.execute("bash-6", {"command": "split-utf8"}, None, None)

    assert _text(result).strip() == "€"


@pytest.mark.asyncio
async def test_bash_tool_preserves_full_output_path_for_truncated_timeout_errors(tmp_path: Path) -> None:
    class Ops:
        async def exec(self, _command: str, _cwd: str, options) -> dict[str, int | None]:
            for index in range(1, 3001):
                options["onData"](f"{index}\n".encode())
            raise RuntimeError("timeout:5")

    tool = create_bash_tool(str(tmp_path), {"operations": Ops()})

    with pytest.raises(RuntimeError, match=r"Command timed out after 5 seconds") as excinfo:
        await tool.execute("bash-7", {"command": "chatty-fail"}, None, None)

    message = str(excinfo.value)
    assert "Full output: " in message
    full_output_path = Path(message.split("Full output: ", 1)[1].split("]", 1)[0])
    assert full_output_path.exists()
    full_output = full_output_path.read_text(encoding="utf-8")
    assert "1\n2\n3" in full_output
    assert "2998\n2999\n3000" in full_output


@pytest.mark.asyncio
async def test_create_local_bash_operations_streams_environment_output(tmp_path: Path) -> None:
    operations = create_local_bash_operations()
    chunks: list[bytes] = []

    result = await operations.exec(
        "printf '%s' \"$TEST_LOCAL_BASH_OPS\"",
        str(tmp_path),
        {
            "onData": chunks.append,
            "env": {"TEST_LOCAL_BASH_OPS": "from-local-ops"},
        },
    )

    assert result["exitCode"] == 0
    assert b"".join(chunks).decode("utf-8") == "from-local-ops"


@pytest.mark.asyncio
async def test_create_local_bash_operations_preserves_explicit_empty_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> dict[str, str]:
        raise AssertionError("get_shell_env should not be used when env={} is explicitly provided")

    monkeypatch.setattr("harnify_coding_agent.core.tools.bash.get_shell_env", fail_if_called)

    operations = create_local_bash_operations()
    chunks: list[bytes] = []
    result = await operations.exec(
        "printf 'ok'",
        str(tmp_path),
        {
            "onData": chunks.append,
            "env": {},
        },
    )

    assert result["exitCode"] == 0
    assert b"".join(chunks) == b"ok"


@pytest.mark.asyncio
async def test_create_local_bash_operations_maps_signal_exit_to_none(tmp_path: Path) -> None:
    operations = create_local_bash_operations()

    result = await operations.exec("kill -9 $$", str(tmp_path), {"onData": lambda _chunk: None})

    assert result["exitCode"] is None


def test_bash_tool_definition_surface_matches_ts(tmp_path: Path) -> None:
    definition = create_bash_tool_definition(str(tmp_path))

    assert definition.promptSnippet == "Execute bash commands (ls, grep, find, etc.)"
    assert definition.prepareArguments is None
    assert definition.renderCall is not None
    assert definition.renderResult is not None
    assert definition.description.endswith("Optionally provide a timeout in seconds.")
    assert "50KB" in definition.description


def test_bash_module_exports_match_ts_surface() -> None:
    assert bash_module.__all__ == [
        "BashOperations",
        "BashSpawnContext",
        "BashSpawnHook",
        "BashToolDetails",
        "BashToolInput",
        "BashToolOptions",
        "createBashTool",
        "createBashToolDefinition",
        "createLocalBashOperations",
    ]
