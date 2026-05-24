from __future__ import annotations

import io
from pathlib import Path

import pytest
from harnify_ai.types import Model
import harnify_coding_agent.cli.file_processor as file_processor_module
from harnify_coding_agent.cli.args import parse_args, print_help
from harnify_coding_agent.cli.file_processor import process_file_arguments
from harnify_coding_agent.cli.initial_message import build_initial_message
from harnify_coding_agent.cli.list_models import list_models
from harnify_coding_agent.config import (
    APP_NAME,
    VERSION,
    get_bundled_interactive_asset_path,
    get_export_template_dir,
    get_themes_dir,
)
from harnify_coding_agent.core.extensions.types import ExtensionFlag
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.main import (
    build_session_options,
    create_session_manager,
    main,
    resolve_app_mode,
    resolve_cli_paths,
    resolve_session_path,
    validate_fork_flags,
)
from PIL import Image


def _model(provider: str, model_id: str, *, reasoning: bool = True, input_modalities: list[str] | None = None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        baseUrl=f"https://{provider}.example.com",
        reasoning=reasoning,
        input=input_modalities or ["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
        contextWindow=200_000,
        maxTokens=16_000,
    )


def test_parse_args_collects_messages_files_and_unknown_flags() -> None:
    parsed = parse_args(
        [
            "--model",
            "openai/gpt-4o:high",
            "--thinking",
            "minimal",
            "--plan",
            "strict",
            "@notes.txt",
            "hello",
            "--list-models",
            "sonnet",
        ]
    )

    assert parsed.model == "openai/gpt-4o:high"
    assert parsed.thinking == "minimal"
    assert parsed.fileArgs == ["notes.txt"]
    assert parsed.messages == ["hello"]
    assert parsed.unknownFlags == {"plan": "strict"}
    assert parsed.listModels == "sonnet"


def test_parse_args_print_consumes_inline_message_and_records_invalid_thinking() -> None:
    parsed = parse_args(["-p", "summarize this", "--thinking", "wild"])

    assert parsed.print is True
    assert parsed.messages == ["summarize this"]
    assert parsed.diagnostics[0].type == "warning"
    assert "Invalid thinking level" in parsed.diagnostics[0].message


def test_print_help_includes_extension_flags() -> None:
    buffer = io.StringIO()
    print_help(
        [
            ExtensionFlag(
                name="plan",
                extensionPath="/tmp/plan.py",
                type="boolean",
                description="Enable planning",
            )
        ],
        stream=buffer,
    )

    output = buffer.getvalue()
    assert "Extension CLI Flags:" in output
    assert "--plan" in output
    assert "Enable planning" in output
    assert "AZURE_OPENAI_API_KEY" in output
    assert "# Interactive mode" in output
    assert "CLOUDFLARE_GATEWAY_ID" in output


def test_args_module_exports_match_ts_surface() -> None:
    from harnify_coding_agent.cli import args as args_module

    assert args_module.__all__ == [
        "Args",
        "Mode",
        "isValidThinkingLevel",
        "parseArgs",
        "printHelp",
    ]


def test_build_initial_message_combines_inputs_and_consumes_first_message() -> None:
    parsed = parse_args(["first", "second"])

    result = build_initial_message(parsed=parsed, fileText="<file>body</file>", stdinContent="stdin:")

    assert result.initialMessage == "stdin:<file>body</file>first"
    assert parsed.messages == ["second"]


@pytest.mark.asyncio
async def test_process_file_arguments_handles_text_and_images(tmp_path: Path) -> None:
    text_path = tmp_path / "prompt.txt"
    text_path.write_text("hello world", encoding="utf-8")

    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image_path)

    processed = await file_processor_module._process_file_arguments([str(text_path), str(image_path)], cwd=str(tmp_path))

    assert '<file name="' in processed.text
    assert "hello world" in processed.text
    assert len(processed.images) == 1
    assert processed.images[0].mimeType == "image/png"


@pytest.mark.asyncio
async def test_process_file_arguments_exits_for_missing_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        await file_processor_module._process_file_arguments(["missing.txt"], cwd=str(tmp_path))

    assert error.value.code == 1
    assert "File not found" in capsys.readouterr().err


def test_file_processor_module_exports_match_ts_surface() -> None:
    assert file_processor_module.__all__ == [
        "ProcessFileOptions",
        "ProcessedFiles",
        "processFileArguments",
    ]


@pytest.mark.asyncio
async def test_list_models_formats_rows_and_searches() -> None:
    class Registry:
        def getError(self):
            return None

        def getAvailable(self):
            return [
                _model("anthropic", "claude-sonnet-4-5", input_modalities=["text", "image"]),
                _model("openai", "gpt-4o-mini", reasoning=False),
            ]

    out = io.StringIO()
    await list_models(Registry(), "sonnet", stream=out)

    rendered = out.getvalue()
    assert "provider" in rendered
    assert "claude-sonnet-4-5" in rendered
    assert "200K" in rendered
    assert "gpt-4o-mini" not in rendered


def test_config_helpers_point_at_bundled_assets() -> None:
    assert Path(get_themes_dir(), "dark.json").exists()
    assert Path(get_export_template_dir(), "template.html").exists()
    assert get_bundled_interactive_asset_path("logo.txt").endswith("logo.txt")


@pytest.mark.asyncio
async def test_main_supports_version_and_export_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert await main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == VERSION

    missing = tmp_path / "missing.jsonl"
    assert await main(["--export", str(missing)]) == 1
    assert "Error:" in capsys.readouterr().err


def test_resolve_app_mode_matches_cli_rules() -> None:
    assert resolve_app_mode(parse_args(["--mode", "rpc"]), True) == "rpc"
    assert resolve_app_mode(parse_args(["--mode", "json"]), True) == "json"
    assert resolve_app_mode(parse_args(["-p"]), True) == "print"
    assert resolve_app_mode(parse_args([]), False) == "print"
    assert resolve_app_mode(parse_args([]), True) == "interactive"


@pytest.mark.asyncio
async def test_resolve_session_path_distinguishes_local_global_and_paths(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    local_project = tmp_path / "local"
    local_project.mkdir()
    local_manager = SessionManager.create(str(local_project))
    _persist_session(local_manager)

    other_project = tmp_path / "other"
    other_project.mkdir()
    other_manager = SessionManager.create(str(other_project))
    _persist_session(other_manager)

    local_prefix = local_manager.getSessionId()
    global_prefix = other_manager.getSessionId()

    local_result = await resolve_session_path(local_prefix, str(local_project))
    assert local_result.type == "local"
    assert local_result.path == local_manager.getSessionFile()

    global_result = await resolve_session_path(global_prefix, str(local_project))
    assert global_result.type == "global"
    assert global_result.path == other_manager.getSessionFile()
    assert global_result.cwd == str(other_project)

    direct_result = await resolve_session_path("../other/direct.jsonl", str(local_project))
    assert direct_result.type == "path"
    assert direct_result.path == str((other_project / "direct.jsonl").resolve())


@pytest.mark.asyncio
async def test_create_session_manager_handles_continue_and_fork(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    current_project = tmp_path / "current"
    current_project.mkdir()
    settings = SettingsManager.inMemory()

    existing = SessionManager.create(str(current_project))
    _persist_session(existing)
    continued = await create_session_manager(parse_args(["--continue"]), str(current_project), None, settings)
    assert continued.getSessionFile() == existing.getSessionFile()

    source_project = tmp_path / "source"
    source_project.mkdir()
    source_manager = SessionManager.create(str(source_project))
    _persist_session(source_manager)
    source_prefix = source_manager.getSessionId()

    forked = await create_session_manager(parse_args(["--fork", source_prefix]), str(current_project), None, settings)
    assert forked.getCwd() == str(current_project)
    assert forked.getSessionFile() != source_manager.getSessionFile()
    assert forked.getHeader()["parentSession"] == source_manager.getSessionFile()


@pytest.mark.asyncio
async def test_create_session_manager_prompts_to_fork_global_sessions(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    current_project = tmp_path / "current"
    current_project.mkdir()
    source_project = tmp_path / "source"
    source_project.mkdir()
    source_manager = SessionManager.create(str(source_project))
    _persist_session(source_manager)
    settings = SettingsManager.inMemory()
    out = io.StringIO()

    manager = await create_session_manager(
        parse_args(["--session", source_manager.getSessionId()]),
        str(current_project),
        None,
        settings,
        prompt_confirm_fn=lambda _message: _async_bool(True),
        output_stream=out,
    )

    assert "Session found in different project" in out.getvalue()
    assert manager.getCwd() == str(current_project)
    assert manager.getHeader()["parentSession"] == source_manager.getSessionFile()


@pytest.mark.asyncio
async def test_create_session_manager_resume_uses_injected_selector(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager.create(str(project))
    _persist_session(manager)
    settings = SettingsManager.inMemory()

    selected = await create_session_manager(
        parse_args(["--resume"]),
        str(project),
        None,
        settings,
        select_session_fn=lambda _current, _all: _async_path(manager.getSessionFile()),
    )
    assert selected.getSessionFile() == manager.getSessionFile()


@pytest.mark.asyncio
async def test_create_session_manager_resume_initializes_theme_and_stops_watcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))

    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager.create(str(project))
    _persist_session(manager)
    settings = SettingsManager.inMemory({"theme": "light"})
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr("harnify_coding_agent.main.init_theme", lambda name, enable=False: calls.append(("init", (name, enable))))
    monkeypatch.setattr("harnify_coding_agent.main.stop_theme_watcher", lambda: calls.append(("stop", None)))

    selected = await create_session_manager(
        parse_args(["--resume"]),
        str(project),
        None,
        settings,
        select_session_fn=lambda _current, _all: _async_path(manager.getSessionFile()),
    )

    assert selected.getSessionFile() == manager.getSessionFile()
    assert calls == [("init", ("light", True)), ("stop", None)]


def test_validate_fork_flags_rejects_conflicts() -> None:
    with pytest.raises(ValueError):
        validate_fork_flags(parse_args(["--fork", "abc123", "--session", "def456"]))


def test_resolve_cli_paths_resolves_local_inputs_only(tmp_path: Path) -> None:
    resolved = resolve_cli_paths(str(tmp_path), ["./ext.py", "npm:package", "https://example.com/theme.json"])

    assert resolved is not None
    assert resolved[0] == str((tmp_path / "ext.py").resolve())
    assert resolved[1] == "npm:package"
    assert resolved[2] == "https://example.com/theme.json"


def test_build_session_options_prefers_cli_model_and_scoped_defaults() -> None:
    models = [
        _model("anthropic", "claude-sonnet-4-5"),
        _model("openai", "gpt-4o-mini", reasoning=False),
    ]

    class Registry:
        def getAll(self):
            return models

        def find(self, provider: str, model_id: str):
            return next((model for model in models if model.provider == provider and model.id == model_id), None)

    settings = SettingsManager.inMemory(
        {"defaultProvider": "openai", "defaultModel": "gpt-4o-mini", "enabledModels": ["openai/gpt-4o-mini:low"]}
    )
    scoped_models = [type("Scoped", (), {"model": models[1], "thinkingLevel": "low"})()]

    explicit = build_session_options(
        parse_args(["--model", "anthropic/claude-sonnet-4-5:high", "--tools", "read,bash"]),
        [],
        False,
        Registry(),
        settings,
    )
    assert explicit.options["model"].provider == "anthropic"
    assert explicit.options["thinkingLevel"] == "high"
    assert explicit.cliThinkingFromModel is True
    assert explicit.options["tools"] == ["read", "bash"]

    defaulted = build_session_options(parse_args([]), scoped_models, False, Registry(), settings)
    assert defaulted.options["model"].provider == "openai"
    assert defaulted.options["thinkingLevel"] == "low"


@pytest.mark.asyncio
async def test_main_rejects_file_args_in_rpc_mode(capsys: pytest.CaptureFixture[str]) -> None:
    code = await main(["--mode", "rpc", "@prompt.md"])
    assert code == 1
    assert "@file arguments are not supported in RPC mode" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_list_models_uses_runtime_bootstrap(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "auth.json").write_text('{"openai":{"type":"api_key","key":"sk-test"}}', encoding="utf-8")
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.chdir(tmp_path)

    code = await main(["--list-models", "gpt-4o"])

    assert code == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "gpt-4o" in output


@pytest.mark.asyncio
async def test_main_help_includes_runtime_extension_flags(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.setenv(f"{APP_NAME.upper()}_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.chdir(tmp_path)

    async def extension_factory(api: object) -> None:
        api.registerFlag(  # type: ignore[attr-defined]
            "demo-flag",
            {
                "type": "string",
                "description": "Demo extension flag",
            },
        )

    code = await main(["--help"], {"extensionFactories": [extension_factory]})

    assert code == 0
    output = capsys.readouterr().out
    assert "Extension CLI Flags:" in output
    assert "--demo-flag <value>" in output
    assert "Demo extension flag" in output


async def _async_bool(value: bool) -> bool:
    return value


async def _async_path(value: str | None) -> str | None:
    return value


def _persist_session(manager: SessionManager) -> None:
    manager.appendMessage({"role": "assistant", "content": "persist"})
