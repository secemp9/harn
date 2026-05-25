from __future__ import annotations

import os
import sys
from pathlib import Path

import harnify_coding_agent as root_package
from harnify_coding_agent import cli as cli_package
from harnify_coding_agent import config
from harnify_coding_agent.core import auth_guidance


def test_config_metadata_defaults_match_package_configuration() -> None:
    assert config.PACKAGE_NAME == "harnify-coding-agent"
    assert config.APP_NAME == "pi"
    assert config.APP_TITLE == "π"
    assert config.VERSION == "0.1.0"
    assert config.isBunBinary is False
    assert config.isBunRuntime is False
    assert Path(config.get_package_json_path()).name == "pyproject.toml"
    assert config.__all__ == [
        "APP_NAME",
        "APP_TITLE",
        "CONFIG_DIR_NAME",
        "ENV_AGENT_DIR",
        "ENV_SESSION_DIR",
        "InstallMethod",
        "PACKAGE_NAME",
        "SelfUpdateCommand",
        "VERSION",
        "detectInstallMethod",
        "expandTildePath",
        "getAgentDir",
        "getAuthPath",
        "getBinDir",
        "getBundledInteractiveAssetPath",
        "getChangelogPath",
        "getCustomThemesDir",
        "getDebugLogPath",
        "getDocsPath",
        "getExamplesPath",
        "getExportTemplateDir",
        "getInteractiveAssetsDir",
        "getModelsPath",
        "getPackageDir",
        "getPackageJsonPath",
        "getPromptsDir",
        "getReadmePath",
        "getSessionsDir",
        "getSettingsPath",
        "getShareViewerUrl",
        "getSelfUpdateCommand",
        "getSelfUpdateUnavailableInstruction",
        "getThemesDir",
        "getToolsDir",
        "getUpdateInstruction",
        "isBunBinary",
        "isBunRuntime",
    ]


def test_detect_install_method_handles_python_layouts() -> None:
    assert config._detect_install_method(
        "/home/test/.local/share/pipx/venvs/harnify-coding-agent/lib/python3.12/site-packages/harnify_coding_agent",
        "/home/test/.local/share/pipx/venvs/harnify-coding-agent/bin/python",
    ) == "pipx"
    assert config._detect_install_method(
        "/home/test/.local/share/uv/tools/harnify-coding-agent/lib/python3.12/site-packages/harnify_coding_agent",
        "/home/test/.local/share/uv/tools/harnify-coding-agent/bin/python",
    ) == "uv-tool"
    assert config._detect_install_method(
        "/usr/lib/python3/dist-packages/harnify_coding_agent",
        "/usr/bin/python3",
    ) == "pip"


def test_detect_install_method_reports_source_checkout_for_repo_tree() -> None:
    assert config.detect_install_method() == "source"


def test_self_update_commands_match_python_install_methods(monkeypatch) -> None:
    monkeypatch.setattr(config, "detect_install_method", lambda: "pipx")
    pipx_command = config.get_self_update_command(config.PACKAGE_NAME)
    assert pipx_command is not None
    assert pipx_command.display == "pipx upgrade harnify-coding-agent"

    monkeypatch.setattr(config, "detect_install_method", lambda: "uv-tool")
    uv_command = config.get_self_update_command(config.PACKAGE_NAME)
    assert uv_command is not None
    assert uv_command.display == "uv tool upgrade harnify-coding-agent"

    monkeypatch.setattr(config, "detect_install_method", lambda: "pip")
    pip_command = config.get_self_update_command(config.PACKAGE_NAME, python_command=["python", "-m", "pip"])
    assert pip_command is not None
    assert pip_command.display == "python -m pip install --upgrade harnify-coding-agent"
    assert config.get_update_instruction(config.PACKAGE_NAME) == (
        f"Run: {sys.executable} -m pip install --upgrade harnify-coding-agent"
    )


def test_self_update_command_requires_writable_install_path(monkeypatch) -> None:
    monkeypatch.setattr(config, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(config, "_is_self_update_path_writable", lambda package_dir=None: False)

    assert config.get_self_update_command(config.PACKAGE_NAME) is None
    assert config.get_update_instruction(config.PACKAGE_NAME) == (
        f"Run: {sys.executable} -m pip install --upgrade harnify-coding-agent"
    )
    assert config.get_self_update_unavailable_instruction(config.PACKAGE_NAME) == (
        "This installation is managed by a pip install, but the install path is not writable. "
        f"Update it yourself with: {sys.executable} -m pip install --upgrade harnify-coding-agent"
    )


def test_self_update_fallback_mentions_source_checkout(monkeypatch) -> None:
    monkeypatch.setattr(config, "detect_install_method", lambda: "source")
    assert config.get_self_update_command(config.PACKAGE_NAME) is None
    instruction = config.get_self_update_unavailable_instruction(config.PACKAGE_NAME)
    assert "source checkout" in instruction
    assert "uv sync" in instruction


def test_cli_package_entrypoint_wraps_async_main(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_invoke(argv: list[str]) -> int:
        seen["argv"] = argv
        seen["env"] = os.environ.get("PI_CODING_AGENT")
        return 17

    monkeypatch.setattr(cli_package, "_invoke_main", fake_invoke)
    monkeypatch.setattr(cli_package, "_set_process_title", lambda title: seen.setdefault("title", title))
    monkeypatch.setattr(cli_package, "_suppress_runtime_warnings", lambda: seen.setdefault("warnings", True))
    monkeypatch.setattr(cli_package, "configureHttpDispatcher", lambda: seen.setdefault("dispatcher", True))
    monkeypatch.delenv("PI_CODING_AGENT", raising=False)

    assert cli_package.main(["--demo"]) == 17
    assert seen["argv"] == ["--demo"]
    assert seen["env"] == "true"
    assert seen["title"] == "pi"
    assert seen["warnings"] is True
    assert seen["dispatcher"] is True
    assert cli_package.__all__ == ["main"]


def test_root_package_exports_match_ts_surface() -> None:
    expected = [
        "getAgentDir",
        "VERSION",
        "AgentSession",
        "AgentSessionConfig",
        "AgentSessionEvent",
        "AgentSessionEventListener",
        "ModelCycleResult",
        "ParsedSkillBlock",
        "PromptOptions",
        "parseSkillBlock",
        "SessionStats",
        "ApiKeyCredential",
        "AuthCredential",
        "AuthStatus",
        "AuthStorage",
        "AuthStorageBackend",
        "FileAuthStorageBackend",
        "InMemoryAuthStorageBackend",
        "OAuthCredential",
        "BranchPreparation",
        "BranchSummaryResult",
        "CollectEntriesResult",
        "CompactionResult",
        "CutPointResult",
        "calculateContextTokens",
        "collectEntriesForBranchSummary",
        "compact",
        "DEFAULT_COMPACTION_SETTINGS",
        "estimateTokens",
        "FileOperations",
        "findCutPoint",
        "findTurnStartIndex",
        "GenerateBranchSummaryOptions",
        "generateBranchSummary",
        "generateSummary",
        "getLastAssistantUsage",
        "prepareBranchEntries",
        "serializeConversation",
        "shouldCompact",
        "createEventBus",
        "EventBus",
        "EventBusController",
        "AgentEndEvent",
        "AgentStartEvent",
        "AgentToolResult",
        "AgentToolUpdateCallback",
        "AppKeybinding",
        "AutocompleteProviderFactory",
        "BashToolCallEvent",
        "BeforeAgentStartEvent",
        "BeforeAgentStartEventResult",
        "BeforeProviderRequestEvent",
        "BeforeProviderRequestEventResult",
        "BuildSystemPromptOptions",
        "CompactOptions",
        "ContextEvent",
        "ContextUsage",
        "CustomToolCallEvent",
        "EditToolCallEvent",
        "ExecOptions",
        "ExecResult",
        "Extension",
        "ExtensionActions",
        "ExtensionAPI",
        "ExtensionCommandContext",
        "ExtensionCommandContextActions",
        "ExtensionContext",
        "ExtensionContextActions",
        "ExtensionError",
        "ExtensionEvent",
        "ExtensionFactory",
        "ExtensionFlag",
        "ExtensionHandler",
        "ExtensionRuntime",
        "ExtensionShortcut",
        "ExtensionUIContext",
        "ExtensionUIDialogOptions",
        "ExtensionWidgetOptions",
        "FindToolCallEvent",
        "GrepToolCallEvent",
        "InputEvent",
        "InputEventResult",
        "InputSource",
        "KeybindingsManager",
        "LoadExtensionsResult",
        "LsToolCallEvent",
        "MessageRenderer",
        "MessageRenderOptions",
        "ProviderConfig",
        "ProviderModelConfig",
        "ReadToolCallEvent",
        "RegisteredCommand",
        "RegisteredTool",
        "ResolvedCommand",
        "SessionBeforeCompactEvent",
        "SessionBeforeForkEvent",
        "SessionBeforeSwitchEvent",
        "SessionBeforeTreeEvent",
        "SessionCompactEvent",
        "SessionShutdownEvent",
        "SessionStartEvent",
        "SessionTreeEvent",
        "SlashCommandInfo",
        "SlashCommandSource",
        "SourceInfo",
        "TerminalInputHandler",
        "ToolCallEvent",
        "ToolCallEventResult",
        "ToolDefinition",
        "ToolExecutionMode",
        "ToolInfo",
        "ToolRenderResultOptions",
        "ToolResultEvent",
        "TurnEndEvent",
        "TurnStartEvent",
        "UserBashEvent",
        "UserBashEventResult",
        "WidgetPlacement",
        "WorkingIndicatorOptions",
        "WriteToolCallEvent",
        "createExtensionRuntime",
        "defineTool",
        "discoverAndLoadExtensions",
        "ExtensionRunner",
        "isBashToolResult",
        "isEditToolResult",
        "isFindToolResult",
        "isGrepToolResult",
        "isLsToolResult",
        "isReadToolResult",
        "isToolCallEventType",
        "isWriteToolResult",
        "wrapRegisteredTool",
        "wrapRegisteredTools",
        "ReadonlyFooterDataProvider",
        "convertToLlm",
        "ModelRegistry",
        "PackageManager",
        "PathMetadata",
        "ProgressCallback",
        "ProgressEvent",
        "ResolvedPaths",
        "ResolvedResource",
        "DefaultPackageManager",
        "ResourceCollision",
        "ResourceDiagnostic",
        "ResourceLoader",
        "DefaultResourceLoader",
        "loadProjectContextFiles",
        "AgentSessionRuntime",
        "AgentSessionRuntimeDiagnostic",
        "AgentSessionServices",
        "CreateAgentSessionFromServicesOptions",
        "CreateAgentSessionOptions",
        "CreateAgentSessionResult",
        "CreateAgentSessionRuntimeFactory",
        "CreateAgentSessionRuntimeResult",
        "CreateAgentSessionServicesOptions",
        "createAgentSession",
        "createAgentSessionFromServices",
        "createAgentSessionRuntime",
        "createAgentSessionServices",
        "createBashTool",
        "createCodingTools",
        "createEditTool",
        "createFindTool",
        "createGrepTool",
        "createLsTool",
        "createReadOnlyTools",
        "createReadTool",
        "createWriteTool",
        "PromptTemplate",
        "BranchSummaryEntry",
        "buildSessionContext",
        "CompactionEntry",
        "CURRENT_SESSION_VERSION",
        "CustomEntry",
        "CustomMessageEntry",
        "FileEntry",
        "getLatestCompactionEntry",
        "ModelChangeEntry",
        "migrateSessionEntries",
        "NewSessionOptions",
        "parseSessionEntries",
        "SessionContext",
        "SessionEntry",
        "SessionEntryBase",
        "SessionHeader",
        "SessionInfo",
        "SessionInfoEntry",
        "SessionManager",
        "SessionMessageEntry",
        "ThinkingLevelChangeEntry",
        "CompactionSettings",
        "ImageSettings",
        "PackageSource",
        "RetrySettings",
        "SettingsManager",
        "formatSkillsForPrompt",
        "LoadSkillsFromDirOptions",
        "LoadSkillsResult",
        "loadSkills",
        "loadSkillsFromDir",
        "Skill",
        "SkillFrontmatter",
        "createSyntheticSourceInfo",
        "BashOperations",
        "BashSpawnContext",
        "BashSpawnHook",
        "BashToolDetails",
        "BashToolInput",
        "BashToolOptions",
        "createBashToolDefinition",
        "createEditToolDefinition",
        "createFindToolDefinition",
        "createGrepToolDefinition",
        "createLocalBashOperations",
        "createLsToolDefinition",
        "createReadToolDefinition",
        "createWriteToolDefinition",
        "DEFAULT_MAX_BYTES",
        "DEFAULT_MAX_LINES",
        "EditOperations",
        "EditToolDetails",
        "EditToolInput",
        "EditToolOptions",
        "FindOperations",
        "FindToolDetails",
        "FindToolInput",
        "FindToolOptions",
        "formatSize",
        "GrepOperations",
        "GrepToolDetails",
        "GrepToolInput",
        "GrepToolOptions",
        "LsOperations",
        "LsToolDetails",
        "LsToolInput",
        "LsToolOptions",
        "ReadOperations",
        "ReadToolDetails",
        "ReadToolInput",
        "ReadToolOptions",
        "ToolsOptions",
        "TruncationOptions",
        "TruncationResult",
        "truncateHead",
        "truncateLine",
        "truncateTail",
        "WriteOperations",
        "WriteToolInput",
        "WriteToolOptions",
        "withFileMutationQueue",
        "MainOptions",
        "main",
        "InteractiveMode",
        "InteractiveModeOptions",
        "ModelInfo",
        "PrintModeOptions",
        "RpcClient",
        "RpcClientOptions",
        "RpcCommand",
        "RpcEventListener",
        "RpcResponse",
        "RpcSessionState",
        "runPrintMode",
        "runRpcMode",
        "ArminComponent",
        "AssistantMessageComponent",
        "BashExecutionComponent",
        "BorderedLoader",
        "BranchSummaryMessageComponent",
        "CompactionSummaryMessageComponent",
        "CustomEditor",
        "CustomMessageComponent",
        "DynamicBorder",
        "ExtensionEditorComponent",
        "ExtensionInputComponent",
        "ExtensionSelectorComponent",
        "FooterComponent",
        "keyHint",
        "keyText",
        "LoginDialogComponent",
        "ModelSelectorComponent",
        "OAuthSelectorComponent",
        "RenderDiffOptions",
        "rawKeyHint",
        "renderDiff",
        "SessionSelectorComponent",
        "SettingsCallbacks",
        "SettingsConfig",
        "SettingsSelectorComponent",
        "ShowImagesSelectorComponent",
        "SkillInvocationMessageComponent",
        "ThemeSelectorComponent",
        "ThinkingSelectorComponent",
        "ToolExecutionComponent",
        "ToolExecutionOptions",
        "TreeSelectorComponent",
        "truncateToVisualLines",
        "UserMessageComponent",
        "UserMessageSelectorComponent",
        "VisualTruncateResult",
        "getLanguageFromPath",
        "getMarkdownTheme",
        "getSelectListTheme",
        "getSettingsListTheme",
        "highlightCode",
        "initTheme",
        "Theme",
        "ThemeColor",
        "copyToClipboard",
        "parseFrontmatter",
        "stripFrontmatter",
        "formatDimensionNote",
        "ResizedImage",
        "resizeImage",
        "getShellConfig",
    ]
    assert root_package.__all__ == expected
    for name in expected:
        assert getattr(root_package, name) is not None


def test_http_dispatcher_configuration_validates_and_records_timeout() -> None:
    from harnify_coding_agent.core import http_dispatcher

    http_dispatcher.configureHttpDispatcher(60_000)
    assert http_dispatcher._get_configured_http_idle_timeout_ms() == 60_000
    assert http_dispatcher.__all__ == [
        "DEFAULT_HTTP_IDLE_TIMEOUT_MS",
        "HTTP_IDLE_TIMEOUT_CHOICES",
        "configureHttpDispatcher",
        "formatHttpIdleTimeoutMs",
        "parseHttpIdleTimeoutMs",
    ]

    try:
        http_dispatcher.configureHttpDispatcher(-1)
    except ValueError as error:
        assert str(error) == "Invalid HTTP idle timeout: -1"
    else:  # pragma: no cover
        raise AssertionError("configureHttpDispatcher should reject invalid values")


def test_auth_guidance_uses_config_docs_path_and_public_exports(monkeypatch) -> None:
    monkeypatch.setattr(auth_guidance, "get_docs_path", lambda: "/tmp/docs")

    help_text = auth_guidance.get_provider_login_help()

    assert "/tmp/docs/providers.md" in help_text
    assert "/tmp/docs/models.md" in help_text
    assert "the selected model" in auth_guidance.format_no_api_key_found_message("unknown")
    assert auth_guidance.__all__ == [
        "formatNoApiKeyFoundMessage",
        "formatNoModelSelectedMessage",
        "formatNoModelsAvailableMessage",
        "format_no_api_key_found_message",
        "format_no_model_selected_message",
        "format_no_models_available_message",
        "getProviderLoginHelp",
        "get_provider_login_help",
    ]
