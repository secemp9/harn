from __future__ import annotations

import asyncio
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
from harnify_coding_agent.core.tools import create_grep_tool, get_text_output
from harnify_coding_agent.utils import tools_manager


def test_tools_manager_module_exports_match_ts_surface() -> None:
    assert tools_manager.__all__ == ["getToolPath", "ensureTool"]


def test_get_tool_path_prefers_local_binary_then_system_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_binary = tmp_path / "fd"
    local_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(tools_manager, "_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda _command: False)

    assert tools_manager.get_tool_path("fd", tools_dir=str(tmp_path)) == str(local_binary)

    local_binary.unlink()
    monkeypatch.setattr(tools_manager, "command_exists", lambda command: command == "fdfind")
    assert tools_manager.get_tool_path("fd", tools_dir=str(tmp_path)) == "fdfind"


@pytest.mark.asyncio
async def test_ensure_tool_short_circuits_for_offline_and_downloads_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    messages: list[str] = []

    monkeypatch.setattr(tools_manager, "get_tool_path", lambda tool, tools_dir=None: None)

    async def fake_download_tool(tool, tools_dir=None):
        return str(tmp_path / tool)

    monkeypatch.setattr(tools_manager, "download_tool", fake_download_tool)
    monkeypatch.setenv("HARNIFY_OFFLINE", "1")

    assert await tools_manager.ensure_tool("rg", tools_dir=str(tmp_path), printer=messages.append) is None
    assert messages == ["ripgrep not found. Offline mode enabled, skipping download."]

    messages.clear()
    monkeypatch.delenv("HARNIFY_OFFLINE", raising=False)
    path = await tools_manager.ensure_tool("rg", tools_dir=str(tmp_path), printer=messages.append)
    assert path == str(tmp_path / "rg")
    assert messages == ["ripgrep not found. Downloading...", f"ripgrep installed to {tmp_path / 'rg'}"]


@pytest.mark.asyncio
async def test_download_tool_extracts_nested_binary_from_tar_gz(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "fixture.tar.gz"
    version = "13.0.0"
    nested_binary_name = "rg"

    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"#!/bin/sh\necho rg\n"
        info = tarfile.TarInfo(name=f"ripgrep-{version}-x86_64-unknown-linux-musl/{nested_binary_name}")
        info.mode = 0o755
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))

    monkeypatch.setattr(tools_manager, "_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "_arch", lambda: "x64")
    monkeypatch.setattr(tools_manager, "get_latest_version", lambda repo: asyncio.sleep(0, result=version))

    async def fake_download(_url: str, dest: str) -> None:
        Path(dest).write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(tools_manager, "download_file", fake_download)

    bin_dir = tmp_path / "bin"
    binary_path = await tools_manager.download_tool("rg", tools_dir=str(bin_dir))
    binary = Path(binary_path)
    assert binary.exists()
    assert binary.read_text(encoding="utf-8") == "#!/bin/sh\necho rg\n"
    assert binary.stat().st_mode & 0o777 == 0o755
    assert sorted(path.name for path in bin_dir.iterdir()) == ["rg"]


@pytest.mark.asyncio
async def test_grep_tool_uses_ensure_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_file = tmp_path / "example.txt"
    test_file.write_text("first line\nmatch line\nlast line", encoding="utf-8")

    calls: list[tuple[str, bool]] = []
    real_rg = tools_manager.get_tool_path("rg")
    assert real_rg is not None

    async def fake_ensure_tool(tool: str, silent: bool = False, **_kwargs) -> str:
        calls.append((tool, silent))
        return str(real_rg)

    monkeypatch.setattr("harnify_coding_agent.core.tools.grep.ensure_tool", fake_ensure_tool)

    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("grep-ensure", {"pattern": "match", "path": str(test_file)}, None, None)

    assert "example.txt:2: match line" in get_text_output(result, show_images=True)
    assert calls == [("rg", True)]
