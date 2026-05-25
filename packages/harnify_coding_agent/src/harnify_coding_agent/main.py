"""Minimal Phase 9 CLI entry orchestration."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from harnify_ai.models import models_are_equal
from harnify_ai.types import ImageContent
from harnify_tui import ProcessTerminal, TUI, setKeybindings

from harnify_coding_agent.cli import session_picker
from harnify_coding_agent.cli.args import Args, parse_args, print_help
from harnify_coding_agent.cli.file_processor import ProcessFileOptions, process_file_arguments
from harnify_coding_agent.cli.initial_message import build_initial_message
from harnify_coding_agent.cli.list_models import list_models
from harnify_coding_agent.config import ENV_SESSION_DIR, VERSION, expand_tilde_path, get_agent_dir, get_package_dir
from harnify_coding_agent.core.agent_session_runtime import (
    CreateAgentSessionRuntimeResult,
    create_agent_session_runtime,
)
from harnify_coding_agent.core.agent_session_services import (
    AgentSessionRuntimeDiagnostic,
    create_agent_session_from_services,
    create_agent_session_services,
)
from harnify_coding_agent.core.auth_guidance import formatNoModelsAvailableMessage
from harnify_coding_agent.core.auth_storage import AuthStorage
from harnify_coding_agent.core.export_html import export_from_file
from harnify_coding_agent.core.http_dispatcher import configureHttpDispatcher
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.model_resolver import ScopedModel, resolveCliModel, resolveModelScope
from harnify_coding_agent.core.output_guard import isStdoutTakenOver, restoreStdout, takeOverStdout
from harnify_coding_agent.core.session_cwd import (
    MissingSessionCwdError,
    SessionCwdIssue,
    format_missing_session_cwd_prompt,
    get_missing_session_cwd_issue,
)
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.settings_manager import SettingsManager
from harnify_coding_agent.core.timings import print_timings, reset_timings, time as time_mark
from harnify_coding_agent.migrations import run_migrations, show_deprecation_warnings
from harnify_coding_agent.modes import run_print_mode
from harnify_coding_agent.modes.interactive.components.extension_selector import ExtensionSelectorComponent
from harnify_coding_agent.modes.interactive import InteractiveMode
from harnify_coding_agent.modes.interactive.theme.theme import init_theme, stop_theme_watcher
from harnify_coding_agent.modes.rpc import run_rpc_mode
from harnify_coding_agent.package_manager_cli import handle_config_command, handle_package_command
from harnify_coding_agent.utils.paths import is_local_path, normalize_path, resolve_path
from harnify_coding_agent.utils.windows_self_update import cleanup_windows_self_update_quarantine

AppMode = Literal["interactive", "print", "json", "rpc"]
PrintOutputMode = Literal["text", "json"]


@dataclass(slots=True)
class RuntimeDiagnostic:
    type: Literal["warning", "error", "info"]
    message: str


@dataclass(slots=True)
class ResolvedSession:
    type: Literal["path", "local", "global", "not_found"]
    path: str | None = None
    cwd: str | None = None
    arg: str | None = None


@dataclass(slots=True)
class BuildSessionOptionsResult:
    options: dict[str, Any] = field(default_factory=dict)
    cliThinkingFromModel: bool = False
    diagnostics: list[RuntimeDiagnostic] = field(default_factory=list)


class SettingsErrorLike(Protocol):
    scope: str
    error: Exception


class MainOptions(TypedDict, total=False):
    extensionFactories: list[Any]


SelectSessionFn = Callable[
    [Callable[..., Awaitable[list[Any]]], Callable[..., Awaitable[list[Any]]]],
    Awaitable[str | None],
]
ConfirmFn = Callable[[str], Awaitable[bool]]


async def read_piped_stdin() -> str | None:
    if sys.stdin.isatty():
        return None
    content = await asyncio.to_thread(sys.stdin.read)
    return content.strip() or None


def is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return value == "1" or lowered in {"true", "yes"}


def resolve_app_mode(parsed: Args, stdin_is_tty: bool) -> AppMode:
    if parsed.mode == "rpc":
        return "rpc"
    if parsed.mode == "json":
        return "json"
    if parsed.print or not stdin_is_tty:
        return "print"
    return "interactive"


def to_print_output_mode(app_mode: AppMode) -> PrintOutputMode:
    return "json" if app_mode == "json" else "text"


def collect_settings_diagnostics(settings_manager: SettingsManager, context: str) -> list[RuntimeDiagnostic]:
    return [
        RuntimeDiagnostic(type="warning", message=f"({context}, {error.scope} settings) {error.error}")
        for error in settings_manager.drainErrors()
    ]


def report_diagnostics(
    diagnostics: list[RuntimeDiagnostic],
    *,
    stream: Any | None = None,
) -> None:
    output = stream or sys.stderr
    for diagnostic in diagnostics:
        prefix = "Error: " if diagnostic.type == "error" else "Warning: " if diagnostic.type == "warning" else ""
        output.write(f"{prefix}{diagnostic.message}\n")


async def prepare_initial_message(
    parsed: Args,
    auto_resize_images: bool,
    stdin_content: str | None = None,
) -> tuple[str | None, list[ImageContent] | None]:
    if not parsed.fileArgs:
        result = build_initial_message(parsed=parsed, stdinContent=stdin_content)
        return result.initialMessage, result.initialImages

    processed = await process_file_arguments(
        parsed.fileArgs,
        options=ProcessFileOptions(autoResizeImages=auto_resize_images),
    )
    result = build_initial_message(
        parsed=parsed,
        fileText=processed.text,
        fileImages=processed.images,
        stdinContent=stdin_content,
    )
    return result.initialMessage, result.initialImages


async def resolve_session_path(session_arg: str, cwd: str, session_dir: str | None = None) -> ResolvedSession:
    if "/" in session_arg or "\\" in session_arg or session_arg.endswith(".jsonl"):
        return ResolvedSession(type="path", path=resolve_path(session_arg, cwd))

    local_sessions = await SessionManager.list(cwd, session_dir)
    local_matches = [session for session in local_sessions if session.id.startswith(session_arg)]
    if local_matches:
        return ResolvedSession(type="local", path=local_matches[0].path)

    global_sessions = await SessionManager.listAll()
    global_matches = [session for session in global_sessions if session.id.startswith(session_arg)]
    if global_matches:
        match = global_matches[0]
        return ResolvedSession(type="global", path=match.path, cwd=match.cwd)

    return ResolvedSession(type="not_found", arg=session_arg)


async def prompt_confirm(message: str, *, input_stream: Any | None = None, output_stream: Any | None = None) -> bool:
    input_handle = input_stream or sys.stdin
    output_handle = output_stream or sys.stdout
    output_handle.write(f"{message} [y/N] ")
    flush = getattr(output_handle, "flush", None)
    if callable(flush):
        flush()
    answer = await asyncio.to_thread(input_handle.readline)
    return answer.strip().lower() in {"y", "yes"}


async def prompt_for_missing_session_cwd(
    issue: SessionCwdIssue,
    settings_manager: SettingsManager,
    *,
    terminal_factory: type[ProcessTerminal] = ProcessTerminal,
    ui_factory: type[TUI] = TUI,
    component_factory: type[ExtensionSelectorComponent] = ExtensionSelectorComponent,
    keybindings_factory: Callable[[], KeybindingsManager] = KeybindingsManager.create,
    set_keybindings_fn: Callable[[KeybindingsManager], None] = setKeybindings,
) -> str | None:
    init_theme(settings_manager.getTheme())
    keybindings = keybindings_factory()
    set_keybindings_fn(keybindings)
    try:
        ui = ui_factory(
            terminal_factory(),
            bool(getattr(settings_manager, "getShowHardwareCursor", lambda: False)()),
        )
    except TypeError:
        ui = ui_factory(terminal_factory())
    if hasattr(ui, "setClearOnShrink"):
        ui.setClearOnShrink(bool(getattr(settings_manager, "getClearOnShrink", lambda: False)()))

    done: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
    closed = False

    def finish(result: str | None) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        ui.stop()
        done.set_result(result)

    selector = component_factory(
        format_missing_session_cwd_prompt(issue),
        ["Continue", "Cancel"],
        lambda option: finish(issue.fallbackCwd if option == "Continue" else None),
        lambda: finish(None),
        {"tui": ui},
    )
    ui.addChild(selector)
    ui.setFocus(selector)
    try:
        ui.start()
        return await done
    finally:
        if not closed:
            closed = True
            ui.stop()


def validate_fork_flags(parsed: Args) -> None:
    if not parsed.fork:
        return
    conflicting_flags = [
        "--session" if parsed.session else None,
        "--continue" if parsed.continue_ else None,
        "--resume" if parsed.resume else None,
        "--no-session" if parsed.noSession else None,
    ]
    conflicts = [flag for flag in conflicting_flags if flag is not None]
    if conflicts:
        raise ValueError(f"--fork cannot be combined with {', '.join(conflicts)}")


def resolve_cli_paths(cwd: str, paths: list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    return [resolve_path(value, cwd) if is_local_path(value) else value for value in paths]


def build_session_options(
    parsed: Args,
    scoped_models: list[ScopedModel],
    has_existing_session: bool,
    model_registry: ModelRegistry,
    settings_manager: SettingsManager,
) -> BuildSessionOptionsResult:
    options: dict[str, Any] = {}
    diagnostics: list[RuntimeDiagnostic] = []
    cli_thinking_from_model = False

    if parsed.model:
        resolved = resolveCliModel(
            {
                "cliProvider": parsed.provider,
                "cliModel": parsed.model,
                "modelRegistry": model_registry,
            }
        )
        if resolved.warning:
            diagnostics.append(RuntimeDiagnostic(type="warning", message=resolved.warning))
        if resolved.error:
            diagnostics.append(RuntimeDiagnostic(type="error", message=resolved.error))
        if resolved.model is not None:
            options["model"] = resolved.model
            if parsed.thinking is None and resolved.thinkingLevel:
                options["thinkingLevel"] = resolved.thinkingLevel
                cli_thinking_from_model = True

    if "model" not in options and scoped_models and not has_existing_session:
        saved_provider = settings_manager.getDefaultProvider()
        saved_model_id = settings_manager.getDefaultModel()
        saved_model = model_registry.find(saved_provider, saved_model_id) if saved_provider and saved_model_id else None
        saved_in_scope = (
            next(
                (scoped for scoped in scoped_models if saved_model and models_are_equal(scoped.model, saved_model)),
                None,
            )
            if saved_model
            else None
        )
        selected = saved_in_scope or scoped_models[0]
        options["model"] = selected.model
        if parsed.thinking is None and selected.thinkingLevel:
            options["thinkingLevel"] = selected.thinkingLevel

    if parsed.thinking is not None:
        options["thinkingLevel"] = parsed.thinking

    if scoped_models:
        options["scopedModels"] = [
            {"model": scoped_model.model, "thinkingLevel": scoped_model.thinkingLevel}
            for scoped_model in scoped_models
        ]

    if parsed.noTools:
        options["noTools"] = "all"
    elif parsed.noBuiltinTools:
        options["noTools"] = "builtin"
    if parsed.tools:
        options["tools"] = list(parsed.tools)

    return BuildSessionOptionsResult(
        options=options,
        cliThinkingFromModel=cli_thinking_from_model,
        diagnostics=diagnostics,
    )


def _to_agent_runtime_diagnostics(diagnostics: list[RuntimeDiagnostic]) -> list[AgentSessionRuntimeDiagnostic]:
    return [AgentSessionRuntimeDiagnostic(type=item.type, message=item.message) for item in diagnostics]


def create_runtime_factory(
    parsed: Args,
    auth_storage: AuthStorage,
    *,
    resolved_extension_paths: list[str] | None = None,
    resolved_skill_paths: list[str] | None = None,
    resolved_prompt_template_paths: list[str] | None = None,
    resolved_theme_paths: list[str] | None = None,
    extension_factories: list[Any] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[CreateAgentSessionRuntimeResult]]:
    async def _factory(runtime_options: dict[str, Any]) -> CreateAgentSessionRuntimeResult:
        resource_loader_options: dict[str, Any] = {
            "noExtensions": parsed.noExtensions,
            "noSkills": parsed.noSkills,
            "noPromptTemplates": parsed.noPromptTemplates,
            "noThemes": parsed.noThemes,
            "noContextFiles": parsed.noContextFiles,
            "systemPrompt": parsed.systemPrompt,
            "appendSystemPrompt": parsed.appendSystemPrompt,
        }
        if resolved_extension_paths is not None:
            resource_loader_options["additionalExtensionPaths"] = resolved_extension_paths
        if resolved_skill_paths is not None:
            resource_loader_options["additionalSkillPaths"] = resolved_skill_paths
        if resolved_prompt_template_paths is not None:
            resource_loader_options["additionalPromptTemplatePaths"] = resolved_prompt_template_paths
        if resolved_theme_paths is not None:
            resource_loader_options["additionalThemePaths"] = resolved_theme_paths
        if extension_factories is not None:
            resource_loader_options["extensionFactories"] = extension_factories

        services = await create_agent_session_services(
            {
                "cwd": runtime_options["cwd"],
                "agentDir": runtime_options["agentDir"],
                "authStorage": auth_storage,
                "extensionFlagValues": parsed.unknownFlags,
                "resourceLoaderOptions": resource_loader_options,
            }
        )
        settings_manager = services.settingsManager
        model_registry = services.modelRegistry
        resource_loader = services.resourceLoader

        diagnostics: list[AgentSessionRuntimeDiagnostic] = [
            *services.diagnostics,
            *_to_agent_runtime_diagnostics(collect_settings_diagnostics(settings_manager, "runtime creation")),
        ]
        for item in resource_loader.getExtensions().errors:
            path = item.get("path", "") if isinstance(item, dict) else getattr(item, "path", "")
            error = item.get("error", "") if isinstance(item, dict) else getattr(item, "error", "")
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(type="error", message=f'Failed to load extension "{path}": {error}')
            )

        model_patterns = parsed.models or settings_manager.getEnabledModels()
        scoped_models = (
            await resolveModelScope(model_patterns, model_registry)
            if model_patterns and len(model_patterns) > 0
            else []
        )
        session_options = build_session_options(
            parsed,
            scoped_models,
            len(runtime_options["sessionManager"].buildSessionContext().messages) > 0,
            model_registry,
            settings_manager,
        )
        diagnostics.extend(_to_agent_runtime_diagnostics(session_options.diagnostics))

        if parsed.apiKey:
            selected_model = session_options.options.get("model")
            if selected_model is None:
                diagnostics.append(
                    AgentSessionRuntimeDiagnostic(
                        type="error",
                        message=(
                            "--api-key requires a model to be specified via --model, "
                            "--provider/--model, or --models"
                        ),
                    )
                )
            else:
                auth_storage.setRuntimeApiKey(selected_model.provider, parsed.apiKey)

        created = await create_agent_session_from_services(
            {
                "services": services,
                "sessionManager": runtime_options["sessionManager"],
                "sessionStartEvent": runtime_options.get("sessionStartEvent"),
                "model": session_options.options.get("model"),
                "thinkingLevel": session_options.options.get("thinkingLevel"),
                "scopedModels": session_options.options.get("scopedModels"),
                "tools": session_options.options.get("tools"),
                "noTools": session_options.options.get("noTools"),
                "customTools": session_options.options.get("customTools"),
            }
        )
        session = created["session"] if isinstance(created, dict) else created.session
        if session.model and (parsed.thinking is not None or session_options.cliThinkingFromModel):
            session.setThinkingLevel(session.thinkingLevel)

        return CreateAgentSessionRuntimeResult(
            session=session,
            services=services,
            diagnostics=diagnostics,
            extensionsResult=created.get("extensionsResult") if isinstance(created, dict) else created.extensionsResult,
            modelFallbackMessage=(
                created.get("modelFallbackMessage") if isinstance(created, dict) else created.modelFallbackMessage
            ),
        )

    return _factory


async def create_session_manager(
    parsed: Args,
    cwd: str,
    session_dir: str | None,
    settings_manager: SettingsManager,
    *,
    prompt_confirm_fn: ConfirmFn = prompt_confirm,
    select_session_fn: SelectSessionFn | None = None,
    output_stream: Any | None = None,
    error_stream: Any | None = None,
) -> SessionManager:
    out = output_stream or sys.stdout
    err = error_stream or sys.stderr
    selector = select_session_fn or session_picker.select_session

    if parsed.noSession:
        return SessionManager.inMemory()

    if parsed.fork:
        resolved = await resolve_session_path(parsed.fork, cwd, session_dir)
        if resolved.type in {"path", "local", "global"} and resolved.path:
            return SessionManager.forkFrom(resolved.path, cwd, session_dir)
        err.write(f"No session found matching '{resolved.arg}'\n")
        raise SystemExit(1)

    if parsed.session:
        resolved = await resolve_session_path(parsed.session, cwd, session_dir)
        if resolved.type in {"path", "local"} and resolved.path:
            return SessionManager.open(resolved.path, session_dir)
        if resolved.type == "global" and resolved.path:
            out.write(f"Session found in different project: {resolved.cwd}\n")
            should_fork = await prompt_confirm_fn("Fork this session into current directory?")
            if not should_fork:
                out.write("Aborted.\n")
                raise SystemExit(0)
            return SessionManager.forkFrom(resolved.path, cwd, session_dir)
        err.write(f"No session found matching '{resolved.arg}'\n")
        raise SystemExit(1)

    if parsed.resume:
        init_theme(settings_manager.getTheme(), True)
        try:
            selected_path = await selector(
                lambda onProgress=None: SessionManager.list(cwd, session_dir, onProgress),
                SessionManager.listAll,
            )
            if not selected_path:
                out.write("No session selected\n")
                raise SystemExit(0)
            return SessionManager.open(selected_path, session_dir)
        finally:
            stop_theme_watcher()

    if parsed.continue_:
        return SessionManager.continueRecent(cwd, session_dir)

    return SessionManager.create(cwd, session_dir)


async def main(args: list[str], options: MainOptions | None = None) -> int:
    reset_timings()
    if "--offline" in args or is_truthy_env_flag(os.environ.get("PI_OFFLINE")):
        os.environ["PI_OFFLINE"] = "1"
        os.environ["PI_SKIP_VERSION_CHECK"] = "1"

    if sys.platform == "win32":
        cleanup_windows_self_update_quarantine(get_package_dir())

    package_command_result = await handle_package_command(args)
    if package_command_result is not None:
        return package_command_result
    config_command_result = await handle_config_command(args)
    if config_command_result is not None:
        return config_command_result

    parsed = parse_args(args)
    for diagnostic in parsed.diagnostics:
        prefix = "Error" if diagnostic.type == "error" else "Warning"
        print(f"{prefix}: {diagnostic.message}", file=sys.stderr)
    if any(diagnostic.type == "error" for diagnostic in parsed.diagnostics):
        return 1
    time_mark("parseArgs")

    app_mode = resolve_app_mode(parsed, sys.stdin.isatty())
    took_over_stdout = app_mode != "interactive"
    if took_over_stdout:
        takeOverStdout()

    def finish(code: int) -> int:
        if took_over_stdout and isStdoutTakenOver():
            restoreStdout()
        return code

    if parsed.version:
        print(VERSION)
        return finish(0)

    if parsed.export:
        output_path = parsed.messages[0] if parsed.messages else None
        try:
            result = await export_from_file(parsed.export, output_path)
        except Exception as error:
            print(f"Error: {error}", file=sys.stderr)
            return finish(1)
        print(f"Exported to: {result}")
        return finish(0)

    if parsed.mode == "rpc" and parsed.fileArgs:
        print("Error: @file arguments are not supported in RPC mode", file=sys.stderr)
        return finish(1)

    try:
        validate_fork_flags(parsed)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return finish(1)

    cwd = os.getcwd()
    migration_result = run_migrations(cwd) or {}
    time_mark("runMigrations")
    agent_dir = get_agent_dir()
    startup_settings_manager = SettingsManager.create(cwd, agent_dir)
    report_diagnostics(collect_settings_diagnostics(startup_settings_manager, "startup session lookup"))
    session_dir = (
        normalize_path(parsed.sessionDir)
        if parsed.sessionDir
        else expand_tilde_path(os.environ[ENV_SESSION_DIR])
        if os.environ.get(ENV_SESSION_DIR)
        else startup_settings_manager.getSessionDir()
    )
    try:
        session_manager = await create_session_manager(parsed, cwd, session_dir, startup_settings_manager)
    except SystemExit as exit_signal:
        code = exit_signal.code
        return finish(int(code) if isinstance(code, int) else 1)
    missing_session_cwd_issue = None
    if hasattr(session_manager, "getSessionFile") and hasattr(session_manager, "getCwd"):
        missing_session_cwd_issue = get_missing_session_cwd_issue(session_manager, cwd)
    if missing_session_cwd_issue is not None:
        if resolve_app_mode(parsed, sys.stdin.isatty()) == "interactive":
            selected_cwd = await prompt_for_missing_session_cwd(missing_session_cwd_issue, startup_settings_manager)
            if selected_cwd is None:
                return finish(0)
            if not missing_session_cwd_issue.sessionFile:
                print(f"Error: {MissingSessionCwdError(missing_session_cwd_issue)}", file=sys.stderr)
                return finish(1)
            session_manager = SessionManager.open(
                missing_session_cwd_issue.sessionFile,
                session_dir,
                selected_cwd,
            )
        else:
            print(f"Error: {MissingSessionCwdError(missing_session_cwd_issue)}", file=sys.stderr)
            return finish(1)
    time_mark("createSessionManager")

    resolved_extension_paths = resolve_cli_paths(cwd, parsed.extensions)
    resolved_skill_paths = resolve_cli_paths(cwd, parsed.skills)
    resolved_prompt_template_paths = resolve_cli_paths(cwd, parsed.promptTemplates)
    resolved_theme_paths = resolve_cli_paths(cwd, parsed.themes)
    auth_storage = AuthStorage.create()
    try:
        runtime = await create_agent_session_runtime(
            create_runtime_factory(
                parsed,
                auth_storage,
                resolved_extension_paths=resolved_extension_paths,
                resolved_skill_paths=resolved_skill_paths,
                resolved_prompt_template_paths=resolved_prompt_template_paths,
                resolved_theme_paths=resolved_theme_paths,
                extension_factories=options.get("extensionFactories") if options else None,
            ),
            {
                "cwd": session_manager.getCwd(),
                "agentDir": agent_dir,
                "sessionManager": session_manager,
            },
        )
        services = runtime.services
        session = runtime.session
        settings_manager = services.settingsManager
        model_registry = services.modelRegistry
        configureHttpDispatcher(settings_manager.getHttpIdleTimeoutMs())
        time_mark("createAgentSessionRuntime")

        if parsed.help:
            extension_flags = [
                flag
                for extension in services.resourceLoader.getExtensions().extensions
                for flag in extension.flags.values()
            ]
            print_help(extension_flags)
            return 0

        if parsed.listModels is not None:
            await list_models(
                model_registry,
                parsed.listModels if isinstance(parsed.listModels, str) else None,
            )
            return 0

        stdin_content = None
        if app_mode != "rpc":
            stdin_content = await read_piped_stdin()
            if stdin_content is not None and app_mode == "interactive":
                app_mode = "print"
        time_mark("readPipedStdin")

        initial_message, initial_images = await prepare_initial_message(
            parsed,
            settings_manager.getImageAutoResize(),
            stdin_content,
        )
        time_mark("prepareInitialMessage")
        init_theme(settings_manager.getTheme(), app_mode == "interactive")
        time_mark("initTheme")

        deprecation_warnings = migration_result.get("deprecationWarnings") or []
        if app_mode == "interactive" and deprecation_warnings:
            await show_deprecation_warnings(deprecation_warnings)

        time_mark("resolveModelScope")
        report_diagnostics(list(runtime.diagnostics))
        if any(item.type == "error" for item in runtime.diagnostics):
            return 1
        time_mark("createAgentSession")

        if app_mode != "interactive" and session.model is None:
            print(formatNoModelsAvailableMessage(), file=sys.stderr)
            return 1

        startup_benchmark = is_truthy_env_flag(os.environ.get("PI_STARTUP_BENCHMARK"))
        if startup_benchmark and app_mode != "interactive":
            print("Error: PI_STARTUP_BENCHMARK only supports interactive mode", file=sys.stderr)
            return 1

        if app_mode == "rpc":
            print_timings()
            return await run_rpc_mode(runtime)

        if app_mode == "interactive":
            interactive_mode = InteractiveMode(
                runtime,
                {
                    "migratedProviders": migration_result.get("migratedAuthProviders"),
                    "modelFallbackMessage": runtime.modelFallbackMessage,
                    "initialMessage": initial_message,
                    "initialImages": initial_images,
                    "initialMessages": list(parsed.messages),
                    "verbose": parsed.verbose,
                },
            )
            if startup_benchmark:
                await interactive_mode.init()
                time_mark("interactiveMode.init")
                print_timings()
                interactive_mode.requestShutdown()
                interactive_mode.dispose()
                flush_stdout = getattr(sys.stdout, "flush", None)
                if callable(flush_stdout):
                    flush_stdout()
                flush_stderr = getattr(sys.stderr, "flush", None)
                if callable(flush_stderr):
                    flush_stderr()
                return 0

            print_timings()
            return await interactive_mode.run()

        print_timings()
        exit_code = await run_print_mode(
            runtime,
            {
                "mode": to_print_output_mode(app_mode),
                "messages": list(parsed.messages),
                "initialMessage": initial_message,
                "initialImages": initial_images,
            },
        )
        stop_theme_watcher()
        return exit_code
    finally:
        if took_over_stdout and isStdoutTakenOver():
            restoreStdout()

__all__ = ["MainOptions", "main"]
