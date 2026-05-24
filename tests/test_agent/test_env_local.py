from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harnify_agent import AbortController
from harnify_agent.harness.env.local import NodeExecutionEnv
from harnify_agent.harness.types import FileError, getOrThrow
from harnify_agent.harness.utils.shell_output import execute_shell_with_capture, sanitize_binary_output


@pytest.mark.asyncio
async def test_node_execution_env_file_operations(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})

    assert getOrThrow(await env.absolutePath("nested/child")) == str(tmp_path / "nested/child")
    assert getOrThrow(await env.joinPath([str(tmp_path), "nested", "child"])) == str(tmp_path / "nested/child")
    getOrThrow(await env.createDir("nested/child"))
    getOrThrow(await env.writeFile("nested/child/file.txt", "hel"))
    getOrThrow(await env.appendFile("nested/child/file.txt", "lo"))
    assert getOrThrow(await env.readTextFile("nested/child/file.txt")) == "hello"
    assert getOrThrow(await env.readTextLines("nested/child/file.txt", {"maxLines": 1})) == ["hello"]
    assert getOrThrow(await env.readBinaryFile("nested/child/file.txt")) == b"hello"

    entries = getOrThrow(await env.listDir("nested/child"))
    assert len(entries) == 1
    assert entries[0].name == "file.txt"
    assert entries[0].path == str(tmp_path / "nested/child/file.txt")
    assert entries[0].kind == "file"
    assert entries[0].size == 5
    assert isinstance(entries[0].mtimeMs, float)

    assert getOrThrow(await env.exists("nested/child/file.txt")) is True
    getOrThrow(await env.remove("nested/child/file.txt"))
    assert getOrThrow(await env.exists("nested/child/file.txt")) is False


@pytest.mark.asyncio
async def test_node_execution_env_symlinks_and_metadata(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})
    getOrThrow(await env.createDir("dir", {"recursive": True}))
    getOrThrow(await env.writeFile("dir/file.txt", "hello"))
    (tmp_path / "file-link").symlink_to(tmp_path / "dir/file.txt")
    (tmp_path / "dir-link").symlink_to(tmp_path / "dir")

    assert getOrThrow(await env.fileInfo("dir")).kind == "directory"
    assert getOrThrow(await env.fileInfo("dir/file.txt")).kind == "file"
    assert getOrThrow(await env.fileInfo("file-link")).kind == "symlink"
    assert getOrThrow(await env.fileInfo("dir-link")).kind == "symlink"
    assert getOrThrow(await env.canonicalPath("file-link")) == str((tmp_path / "dir/file.txt").resolve())

    entries = getOrThrow(await env.listDir("."))
    assert sorted((entry.name, entry.kind) for entry in entries) == [
        ("dir", "directory"),
        ("dir-link", "symlink"),
        ("file-link", "symlink"),
    ]


@pytest.mark.asyncio
async def test_node_execution_env_missing_and_non_directory_errors(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})
    info = await env.fileInfo("missing.txt")
    assert info.ok is False
    assert isinstance(info.error, FileError)
    assert info.error.code == "not_found"
    assert info.error.path == str(tmp_path / "missing.txt")
    assert getOrThrow(await env.exists("missing.txt")) is False

    getOrThrow(await env.writeFile("file.txt", "hello"))
    result = await env.listDir("file.txt")
    assert result.ok is False
    assert result.error.code == "not_directory"


@pytest.mark.asyncio
async def test_node_execution_env_append_temp_and_recursive_remove_behaviour(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})
    getOrThrow(await env.appendFile("new/nested/file.txt", "a"))
    getOrThrow(await env.appendFile("new/nested/file.txt", "b"))
    assert getOrThrow(await env.readTextFile("new/nested/file.txt")) == "ab"

    temp_dir = getOrThrow(await env.createTempDir("node-env-test-"))
    assert Path(temp_dir).exists()
    temp_file = getOrThrow(await env.createTempFile({"prefix": "prefix-", "suffix": ".txt"}))
    assert Path(temp_file).exists()
    assert temp_file.endswith(".txt")

    create_result = await env.createDir("missing/child", {"recursive": False})
    assert create_result.ok is False
    assert create_result.error.code == "not_found"

    getOrThrow(await env.writeFile("dir/child/file.txt", "hello"))
    remove_directory = await env.remove("dir", {"recursive": False})
    assert remove_directory.ok is False
    getOrThrow(await env.remove("dir", {"recursive": True}))
    assert getOrThrow(await env.exists("dir")) is False

    remove_missing = await env.remove("missing", {"force": False})
    assert remove_missing.ok is False
    getOrThrow(await env.remove("missing", {"force": True}))


@pytest.mark.asyncio
async def test_node_execution_env_aborted_file_operations_and_cleanup(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})
    getOrThrow(await env.writeFile("file.txt", "hello"))
    controller = AbortController()
    controller.abort()
    signal_obj = controller.signal

    results = await asyncio.gather(
        env.readTextFile("file.txt", signal_obj),
        env.readTextLines("file.txt", {"abortSignal": signal_obj}),
        env.readBinaryFile("file.txt", signal_obj),
        env.writeFile("other.txt", "hello", signal_obj),
        env.listDir(".", signal_obj),
    )
    for result in results:
        assert result.ok is False
        assert result.error.code == "aborted"

    assert await env.cleanup() is None


@pytest.mark.asyncio
async def test_node_execution_env_exec_behaviour(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})

    result = getOrThrow(
        await env.exec(
            'printf "%s:%s" "$PWD" "$NODE_ENV_TEST"',
            {"env": {"NODE_ENV_TEST": "ok"}},
        )
    )
    assert result == {"stdout": f"{tmp_path.resolve()}:ok", "stderr": "", "exitCode": 0}

    collected_stdout = []
    collected_stderr = []
    streamed_callbacks = getOrThrow(
        await env.exec(
            "printf out; printf err >&2",
            {
                "onStdout": lambda chunk: collected_stdout.append(chunk),
                "onStderr": lambda chunk: collected_stderr.append(chunk),
            },
        )
    )
    assert streamed_callbacks == {"stdout": "out", "stderr": "err", "exitCode": 0}
    assert "".join(collected_stdout) == "out"
    assert "".join(collected_stderr) == "err"

    assert getOrThrow(await env.exec("exit 7")) == {"stdout": "", "stderr": "", "exitCode": 7}

    timeout_result = await env.exec("sleep 5", {"timeout": 0.01})
    assert timeout_result.ok is False
    assert timeout_result.error.code == "timeout"

    callback_result = await env.exec(
        "printf out",
        {"onStdout": lambda _chunk: (_ for _ in ()).throw(RuntimeError("callback failed"))},
    )
    assert callback_result.ok is False
    assert callback_result.error.code == "callback_error"
    assert callback_result.error.message == "callback failed"

    missing_shell_env = NodeExecutionEnv({"cwd": str(tmp_path), "shellPath": str(tmp_path / "missing-shell")})
    missing_shell = await missing_shell_env.exec("printf ok")
    assert missing_shell.ok is False
    assert missing_shell.error.code == "shell_unavailable"

    shell_path = tmp_path / "not-executable-shell"
    shell_path.write_text("not executable", encoding="utf-8")
    spawn_error_env = NodeExecutionEnv({"cwd": str(tmp_path), "shellPath": str(shell_path)})
    spawn_error = await spawn_error_env.exec("printf ok")
    assert spawn_error.ok is False
    assert spawn_error.error.code == "spawn_error"

    controller = AbortController()
    promise = asyncio.create_task(env.exec("sleep 5", {"abortSignal": controller.signal}))
    controller.abort()
    aborted = await promise
    assert aborted.ok is False
    assert aborted.error.code == "aborted"


@pytest.mark.asyncio
async def test_execute_shell_with_capture_and_binary_sanitization(tmp_path: Path) -> None:
    env = NodeExecutionEnv({"cwd": str(tmp_path)})
    result = getOrThrow(await execute_shell_with_capture(env, "yes line | head -n 15000"))
    assert result.truncated is True
    assert result.fullOutputPath is not None
    full_output = getOrThrow(await env.readTextFile(result.fullOutputPath))
    assert len(full_output.split("\n")) > 10000
    assert len(result.output) < len(full_output)

    assert sanitize_binary_output("ok\x00\x01\t\n\rmore\ufff9bad") == "ok\t\n\rmorebad"

    callback_failure = await execute_shell_with_capture(
        env,
        "printf out",
        {"onChunk": lambda _chunk: (_ for _ in ()).throw(RuntimeError("chunk failed"))},
    )
    assert callback_failure.ok is False
    assert callback_failure.error.code == "unknown"
    assert callback_failure.error.message == "chunk failed"
