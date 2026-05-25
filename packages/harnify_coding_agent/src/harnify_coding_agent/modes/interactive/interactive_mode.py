"""Interactive-mode runtime shell and foundational helpers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from harnify_ai.types import ImageContent
from harnify_ai.models import getProviders
from harnify_tui import (
    TUI,
    AutocompleteProvider,
    CombinedAutocompleteProvider,
    Container,
    DefaultTextStyle,
    Loader,
    LoaderIndicatorOptions,
    Markdown,
    ProcessTerminal,
    SlashCommand,
    Spacer,
    Text,
    TruncatedText,
    getCapabilities,
    hyperlink,
    matchesKey,
    setKeybindings,
    visibleWidth,
)

from harnify_coding_agent.config import (
    APP_NAME,
    APP_TITLE,
    PACKAGE_NAME,
    VERSION,
    get_agent_dir,
    get_auth_path,
    get_changelog_path,
    get_debug_log_path,
    get_docs_path,
    get_share_viewer_url,
    get_update_instruction,
)
from harnify_coding_agent.core.agent_session import parse_skill_block
from harnify_coding_agent.core.agent_session_runtime import SessionImportFileNotFoundError
from harnify_coding_agent.core.footer_data_provider import FooterDataProvider
from harnify_coding_agent.core.keybindings import KEYBINDINGS, KeybindingsManager
from harnify_coding_agent.core.messages import createCompactionSummaryMessage
from harnify_coding_agent.core.model_resolver import (
    defaultModelPerProvider,
    findExactModelReferenceMatch,
    resolveModelScope,
)
from harnify_coding_agent.core.package_manager import DefaultPackageManager
from harnify_coding_agent.core.provider_display_names import BUILT_IN_PROVIDER_DISPLAY_NAMES
from harnify_coding_agent.core.session_cwd import MissingSessionCwdError, format_missing_session_cwd_prompt
from harnify_coding_agent.core.session_manager import SessionManager
from harnify_coding_agent.core.slash_commands import BUILTIN_SLASH_COMMANDS, _LOCAL_ALIAS_SLASH_COMMANDS
from harnify_coding_agent.core.telemetry import is_install_telemetry_enabled
from harnify_coding_agent.modes.interactive.components.assistant_message import AssistantMessageComponent
from harnify_coding_agent.modes.interactive.components.armin import ArminComponent
from harnify_coding_agent.modes.interactive.components.bash_execution import BashExecutionComponent
from harnify_coding_agent.modes.interactive.components.bordered_loader import BorderedLoader
from harnify_coding_agent.modes.interactive.components.branch_summary_message import (
    BranchSummaryMessageComponent,
)
from harnify_coding_agent.modes.interactive.components.compaction_summary_message import (
    CompactionSummaryMessageComponent,
)
from harnify_coding_agent.modes.interactive.components.countdown_timer import CountdownTimer
from harnify_coding_agent.modes.interactive.components.custom_editor import CustomEditor
from harnify_coding_agent.modes.interactive.components.custom_message import CustomMessageComponent
from harnify_coding_agent.modes.interactive.components.daxnuts import DaxnutsComponent
from harnify_coding_agent.modes.interactive.components.dynamic_border import DynamicBorder
from harnify_coding_agent.modes.interactive.components.earendil_announcement import EarendilAnnouncementComponent
from harnify_coding_agent.modes.interactive.components.extension_editor import ExtensionEditorComponent
from harnify_coding_agent.modes.interactive.components.extension_input import ExtensionInputComponent
from harnify_coding_agent.modes.interactive.components.extension_selector import (
    ExtensionSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.footer import FooterComponent
from harnify_coding_agent.modes.interactive.components.keybinding_hints import (
    key_display_text,
    key_hint,
    key_text,
    raw_key_hint,
)
from harnify_coding_agent.modes.interactive.components.login_dialog import LoginDialogComponent
from harnify_coding_agent.modes.interactive.components.model_selector import (
    ModelSelectorComponent,
    ScopedModelItem,
)
from harnify_coding_agent.modes.interactive.components.oauth_selector import (
    AuthSelectorProvider,
    OAuthSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.session_selector import SessionSelectorComponent
from harnify_coding_agent.modes.interactive.components.settings_selector import (
    SettingsCallbacks,
    SettingsConfig,
    SettingsSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.skill_invocation_message import (
    SkillInvocationMessageComponent,
)
from harnify_coding_agent.modes.interactive.components.theme_selector import ThemeSelectorComponent
from harnify_coding_agent.modes.interactive.components.tool_execution import ToolExecutionComponent
from harnify_coding_agent.modes.interactive.components.tree_selector import TreeSelectorComponent
from harnify_coding_agent.modes.interactive.components.user_message import UserMessageComponent
from harnify_coding_agent.modes.interactive.components.user_message_selector import (
    UserMessageItem,
    UserMessageSelectorComponent,
)
from harnify_coding_agent.utils.changelog import get_new_entries, parse_changelog
from harnify_coding_agent.utils.clipboard import copy_to_clipboard
from harnify_coding_agent.utils.clipboard_image import (
    extension_for_image_mime_type,
    read_clipboard_image,
)
from harnify_coding_agent.utils.shell import kill_tracked_detached_children
from harnify_coding_agent.utils.version_check import LatestPiRelease, check_for_new_pi_version

interactive_theme = import_module("harnify_coding_agent.modes.interactive.theme.theme")

ANTHROPIC_SUBSCRIPTION_AUTH_WARNING = (
    "Anthropic subscription auth is active. Third-party harness usage draws from extra usage and is billed per "
    "token, not your Claude plan limits. Manage extra usage at https://claude.ai/settings/usage."
)

_BUILT_IN_MODEL_PROVIDERS = frozenset(getProviders())
_BEDROCK_PROVIDER_ID = "amazon-bedrock"
_DEAD_TERMINAL_ERROR_CODES = frozenset({"EIO", "EPIPE", "ENOTCONN"})


@dataclass(slots=True)
class InteractiveModeOptions:
    migratedProviders: list[str] | None = None
    modelFallbackMessage: str | None = None
    initialMessage: str | None = None
    initialImages: list[ImageContent] | None = None
    initialMessages: list[str] | None = None
    verbose: bool = False


class ExpandableText(Text):
    def __init__(
        self,
        getCollapsedText: Callable[[], str],
        getExpandedText: Callable[[], str],
        expanded: bool = False,
        paddingX: int = 0,
        paddingY: int = 0,
    ) -> None:
        self._getCollapsedText = getCollapsedText
        self._getExpandedText = getExpandedText
        self.expanded = expanded
        super().__init__(getExpandedText() if expanded else getCollapsedText(), paddingX, paddingY)

    def setExpanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.setText(self._getExpandedText() if expanded else self._getCollapsedText())


def is_anthropic_subscription_auth_key(api_key: str | None) -> bool:
    return isinstance(api_key, str) and api_key.startswith("sk-ant-oat")


def isApiKeyLoginProvider(
    providerId: str,
    oauthProviderIds: frozenset[str] | set[str],
    builtInProviderIds: frozenset[str] | set[str] = _BUILT_IN_MODEL_PROVIDERS,
) -> bool:
    if BUILT_IN_PROVIDER_DISPLAY_NAMES.get(providerId):
        return True
    if providerId in builtInProviderIds:
        return False
    return providerId not in oauthProviderIds


def _is_unknown_model(model: Any) -> bool:
    return (
        model is not None
        and _value(model, "provider") == "unknown"
        and _value(model, "id") == "unknown"
        and _value(model, "api") == "unknown"
    )


def _is_dead_terminal_error(error: Any) -> bool:
    return getattr(error, "code", None) in _DEAD_TERMINAL_ERROR_CODES


async def _noop_async(*_args: Any, **_kwargs: Any) -> Any:
    return None


async def _cancelled_async_result(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
    return {"cancelled": True}


def _callable_attr(obj: Any, name: str) -> Callable[..., Any] | None:
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_signal_aborted(signal: Any) -> bool:
    return bool(getattr(signal, "aborted", False))


def _register_abort_handler(signal: Any, callback: Callable[[], None]) -> Callable[[], None]:
    if signal is None:
        return lambda: None

    if _is_signal_aborted(signal):
        callback()
        return lambda: None

    add_listener = getattr(signal, "addEventListener", None)
    remove_listener = getattr(signal, "removeEventListener", None)
    if callable(add_listener):
        add_listener("abort", callback, {"once": True})

        def _remove_event_listener() -> None:
            if callable(remove_listener):
                remove_listener("abort", callback)

        return _remove_event_listener

    wait = getattr(signal, "wait", None)
    if callable(wait):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return lambda: None

        async def _watch_abort() -> None:
            await wait()
            callback()

        task = loop.create_task(_watch_abort())

        def _cancel_task() -> None:
            task.cancel()

        return _cancel_task

    return lambda: None


def _message_role(message: Any) -> str | None:
    role = _value(message, "role")
    return role if isinstance(role, str) else None


def _tool_definition(session: Any, name: str) -> Any | None:
    getter = _callable_attr(session, "getToolDefinition")
    return getter(name) if getter is not None else None


class _ExtensionUIContext:
    def __init__(self, mode: InteractiveMode) -> None:
        self._mode = mode

    @property
    def theme(self) -> interactive_theme.Theme:
        return interactive_theme.theme

    def notify(self, message: str, type: str | None = None) -> None:
        self._mode.showExtensionNotify(message, type)

    async def select(
        self,
        title: str,
        options: list[str],
        opts: dict[str, Any] | None = None,
    ) -> str | None:
        return await self._mode.showExtensionSelector(title, options, opts)

    async def confirm(
        self,
        title: str,
        message: str,
        opts: dict[str, Any] | None = None,
    ) -> bool:
        return await self._mode.showExtensionConfirm(title, message, opts)

    async def input(
        self,
        title: str,
        placeholder: str | None = None,
        opts: dict[str, Any] | None = None,
    ) -> str | None:
        return await self._mode.showExtensionInput(title, placeholder, opts)

    def onTerminalInput(self, handler: Any) -> Any:
        return self._mode.addExtensionTerminalInputListener(handler)

    def setStatus(self, key: str, text: str | None = None) -> None:
        self._mode.setExtensionStatus(key, text)

    def setWorkingMessage(self, message: str | None) -> None:
        self._mode.workingMessage = message
        if self._mode.loadingAnimation is not None:
            self._mode.loadingAnimation.setMessage(self._mode.getWorkingLoaderMessage())

    def setWorkingVisible(self, visible: bool) -> None:
        self._mode.setWorkingVisible(visible)

    def setWorkingIndicator(self, options: LoaderIndicatorOptions | None = None) -> None:
        self._mode.setWorkingIndicator(options)

    def setHiddenThinkingLabel(self, label: str | None = None) -> None:
        self._mode.setHiddenThinkingLabel(label)

    def setWidget(self, key: str, content: Any, options: dict[str, Any] | None = None) -> None:
        self._mode.setExtensionWidget(key, content, options)

    def setFooter(self, factory: Any) -> None:
        self._mode.setExtensionFooter(factory)

    def setHeader(self, factory: Any) -> None:
        self._mode.setExtensionHeader(factory)

    def setTitle(self, title: str) -> None:
        set_title = _callable_attr(getattr(self._mode.ui, "terminal", None), "setTitle")
        if set_title is not None:
            set_title(title)

    async def custom(self, factory: Any, options: dict[str, Any] | None = None) -> Any:
        return await self._mode.showExtensionCustom(factory, options)

    def pasteToEditor(self, text: str) -> None:
        editor = self._mode.editor
        handle_input = _callable_attr(editor, "handleInput")
        if handle_input is not None:
            handle_input(f"\x1b[200~{text}\x1b[201~")
            return
        self.setEditorText(text)

    def setEditorText(self, text: str) -> None:
        set_text = _callable_attr(self._mode.editor, "setText")
        if set_text is not None:
            set_text(text)

    def getEditorText(self) -> str:
        expanded = _callable_attr(self._mode.editor, "getExpandedText")
        if expanded is not None:
            return str(expanded())
        get_text = _callable_attr(self._mode.editor, "getText")
        if get_text is not None:
            return str(get_text())
        return ""

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        return await self._mode.showExtensionEditor(title, prefill)

    def addAutocompleteProvider(self, factory: Callable[[AutocompleteProvider], AutocompleteProvider]) -> None:
        self._mode.autocompleteProviderWrappers.append(factory)
        self._mode.setupAutocompleteProvider()

    def setEditorComponent(self, factory: Any) -> None:
        self._mode.setCustomEditorComponent(factory)

    def getEditorComponent(self) -> Any:
        return self._mode.editorComponentFactory

    def getAllThemes(self) -> list[Any]:
        return interactive_theme.get_available_themes_with_paths()

    def getTheme(self, name: str | None = None) -> interactive_theme.Theme:
        return interactive_theme.get_theme_by_name(name)

    def setTheme(self, theme_or_name: str | interactive_theme.Theme) -> dict[str, Any]:
        if isinstance(theme_or_name, interactive_theme.Theme):
            interactive_theme.set_theme_instance(theme_or_name)
            self._mode._request_render()
            return {"success": True}

        result = interactive_theme.set_theme(theme_or_name, True)
        if result.get("success"):
            current_theme = None
            get_theme = _callable_attr(self._mode.settingsManager, "getTheme")
            if get_theme is not None:
                current_theme = get_theme()
            if current_theme != theme_or_name:
                set_theme = _callable_attr(self._mode.settingsManager, "setTheme")
                if set_theme is not None:
                    set_theme(theme_or_name)
            self._mode._request_render()
        return result

    def getToolsExpanded(self) -> bool:
        return bool(self._mode.toolOutputExpanded)

    def setToolsExpanded(self, expanded: bool) -> None:
        self._mode.setToolsExpanded(expanded)


class InteractiveMode:
    MAX_WIDGET_LINES = 10

    def __init__(
        self,
        runtimeHost: Any | None = None,
        options: InteractiveModeOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        runtime_host = runtimeHost if runtimeHost is not None else kwargs.pop("runtimeHost", None)
        runtime_session = getattr(runtime_host, "session", None)

        if "session" not in kwargs and runtime_session is not None:
            kwargs["session"] = runtime_session
        if "sessionManager" not in kwargs and getattr(runtime_session, "sessionManager", None) is not None:
            kwargs["sessionManager"] = runtime_session.sessionManager
        if "settingsManager" not in kwargs and getattr(runtime_session, "settingsManager", None) is not None:
            kwargs["settingsManager"] = runtime_session.settingsManager

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.runtimeHost = runtime_host
        self.options = (
            options
            if isinstance(options, InteractiveModeOptions)
            else InteractiveModeOptions(**dict(options or {}))
        )

        self.session = getattr(
            self,
            "session",
            SimpleNamespace(
                promptTemplates=[],
                scopedModels=[],
                autoCompactionEnabled=False,
                isStreaming=False,
                isCompacting=False,
                extensionRunner=SimpleNamespace(
                    get_registered_commands=lambda: [],
                    get_message_renderer=lambda _custom_type: None,
                ),
                resourceLoader=SimpleNamespace(getSkills=lambda: {"skills": []}, getThemes=lambda: {"themes": []}),
                modelRegistry=SimpleNamespace(
                    authStorage=SimpleNamespace(get=lambda *_args, **_kwargs: None),
                    getApiKeyForProvider=_noop_async,
                    getAvailable=lambda: [],
                    isUsingOAuth=lambda _model: False,
                ),
                state=SimpleNamespace(messages=[], model=None, thinkingLevel="off"),
                subscribe=lambda _listener: (lambda: None),
                bindExtensions=_noop_async,
                prompt=_noop_async,
                setModel=_noop_async,
                cycleModel=_noop_async,
                cycleThinkingLevel=lambda: None,
                setThinkingLevel=lambda _level: None,
                getAvailableThinkingLevels=lambda: ["off", "minimal", "low", "medium", "high"],
                executeBash=_noop_async,
                navigateTree=_cancelled_async_result,
                abortBranchSummary=lambda: None,
            ),
        )
        self.sessionManager = getattr(
            self,
            "sessionManager",
            SimpleNamespace(
                getCwd=lambda: os.getcwd(),
                getLeafId=lambda: None,
                getSessionFile=lambda: None,
                getSessionDir=lambda: None,
                getSessionName=lambda: None,
                buildSessionContext=lambda: SimpleNamespace(messages=[]),
                getEntries=lambda: [],
                getTree=lambda: [],
                appendLabelChange=lambda _entry_id, _label: None,
            ),
        )
        self.settingsManager = getattr(
            self,
            "settingsManager",
            SimpleNamespace(
                getTheme=lambda: None,
                setTheme=lambda _theme: None,
                getWarnings=lambda: {},
                getEnableSkillCommands=lambda: True,
                getShowTerminalProgress=lambda: False,
                getQuietStartup=lambda: False,
                getCollapseChangelog=lambda: True,
                getLastChangelogVersion=lambda: None,
                getCodeBlockIndent=lambda: "  ",
                getHideThinkingBlock=lambda: False,
                getEnableInstallTelemetry=lambda: False,
                getEditorPaddingX=lambda: 0,
                getAutocompleteMaxVisible=lambda: 5,
                getClearOnShrink=lambda: False,
                getShowHardwareCursor=lambda: False,
                setLastChangelogVersion=lambda _version: None,
                setHideThinkingBlock=lambda _hide: None,
            ),
        )
        self.runtimeHost = getattr(
            self,
            "runtimeHost",
            SimpleNamespace(
                importFromJsonl=_noop_async,
                fork=_noop_async,
                switchSession=_noop_async,
                newSession=_noop_async,
                dispose=_noop_async,
                setBeforeSessionInvalidate=lambda *_args, **_kwargs: None,
                setRebindSession=lambda *_args, **_kwargs: None,
            ),
        )
        if getattr(self, "ui", None) is None:
            if runtime_session is not None:
                self.ui = TUI(ProcessTerminal(), _safe_call_bool(self.settingsManager, "getShowHardwareCursor"))
                self.ui.setClearOnShrink(_safe_call_bool(self.settingsManager, "getClearOnShrink"))
            else:
                self.ui = SimpleNamespace(
                    requestRender=lambda *_args, **_kwargs: None,
                    start=lambda: None,
                    stop=lambda: None,
                    addChild=lambda _component: None,
                    removeChild=lambda _component: None,
                    setFocus=lambda _component: None,
                    showOverlay=lambda component, _options=None: SimpleNamespace(
                        hide=lambda: None,
                        focus=lambda: None,
                        component=component,
                    ),
                    hideOverlay=lambda: None,
                    invalidate=lambda: None,
                    terminal=SimpleNamespace(
                        setProgress=lambda *_args, **_kwargs: None,
                        setTitle=lambda *_args, **_kwargs: None,
                    ),
                )
        self.chatContainer = getattr(self, "chatContainer", Container())
        self.pendingMessagesContainer = getattr(self, "pendingMessagesContainer", Container())
        self.statusContainer = getattr(self, "statusContainer", Container())
        self.headerContainer = getattr(self, "headerContainer", Container())
        self.widgetContainerAbove = getattr(self, "widgetContainerAbove", Container())
        self.widgetContainerBelow = getattr(self, "widgetContainerBelow", Container())
        self.editorContainer = getattr(self, "editorContainer", Container())
        self.defaultEditor = getattr(self, "defaultEditor", None)
        self.editor = getattr(self, "editor", None)
        self.keybindings = getattr(self, "keybindings", None)
        if self.keybindings is None:
            self.keybindings = KeybindingsManager.create() if runtime_session is not None else KeybindingsManager()
        if self.defaultEditor is None:
            if runtime_session is not None:
                self.defaultEditor = CustomEditor(
                    self.ui,
                    interactive_theme.get_editor_theme(),
                    self.keybindings,
                    {
                        "paddingX": _safe_call_int(self.settingsManager, "getEditorPaddingX", 0),
                        "autocompleteMaxVisible": _safe_call_int(
                            self.settingsManager, "getAutocompleteMaxVisible", 5
                        ),
                    },
                )
            else:
                self.defaultEditor = SimpleNamespace(
                    onEscape=None,
                    onCtrlD=None,
                    onPasteImage=None,
                    onExtensionShortcut=None,
                    onChange=None,
                    onSubmit=None,
                    borderColor=lambda text: text,
                    actionHandlers={},
                    paddingX=0,
                    autocompleteMaxVisible=5,
                    addToHistory=lambda _text: None,
                    setAutocompleteProvider=lambda _provider: None,
                    setText=lambda _text: None,
                    getText=lambda: "",
                    getExpandedText=lambda: "",
                    setPaddingX=lambda _padding: None,
                    setAutocompleteMaxVisible=lambda _visible: None,
                    onAction=lambda _action, _handler: None,
                )
        if self.editor is None:
            self.editor = self.defaultEditor
        if getattr(self.editorContainer, "children", None) == []:
            add_child = _callable_attr(self.editorContainer, "addChild")
            if add_child is not None:
                add_child(self.editor)

        self.footerDataProvider = getattr(self, "footerDataProvider", None)
        if self.footerDataProvider is None:
            self.footerDataProvider = FooterDataProvider(self.sessionManager.getCwd())
        self.footer = getattr(self, "footer", None)
        if self.footer is None:
            self.footer = FooterComponent(self.session, self.footerDataProvider)
        self.lastStatusSpacer = getattr(self, "lastStatusSpacer", None)
        self.lastStatusText = getattr(self, "lastStatusText", None)
        self.autocompleteProviderWrappers = list(getattr(self, "autocompleteProviderWrappers", []))
        self.autocompleteProvider = getattr(self, "autocompleteProvider", None)
        self.toolOutputExpanded = bool(getattr(self, "toolOutputExpanded", False))
        self.customHeader = getattr(self, "customHeader", None)
        self.builtInHeader = getattr(self, "builtInHeader", None)
        self.customFooter = getattr(self, "customFooter", None)
        self.editorComponentFactory = getattr(self, "editorComponentFactory", None)
        self.loadingAnimation = getattr(self, "loadingAnimation", None)
        self.autoCompactionEscapeHandler = getattr(self, "autoCompactionEscapeHandler", None)
        self.autoCompactionLoader = getattr(self, "autoCompactionLoader", None)
        self.extensionWidgetsAbove: dict[str, Any] = dict(getattr(self, "extensionWidgetsAbove", {}))
        self.extensionWidgetsBelow: dict[str, Any] = dict(getattr(self, "extensionWidgetsBelow", {}))
        self.extensionTerminalInputUnsubscribers: set[Callable[[], None]] = set(
            getattr(self, "extensionTerminalInputUnsubscribers", set())
        )
        self.skillCommands: dict[str, str] = dict(getattr(self, "skillCommands", {}))
        self.fdPath = getattr(self, "fdPath", None)
        self.anthropicSubscriptionWarningShown = bool(
            getattr(self, "anthropicSubscriptionWarningShown", False)
        )
        self.hideThinkingBlock = bool(
            getattr(self, "hideThinkingBlock", _safe_call_bool(self.settingsManager, "getHideThinkingBlock"))
        )
        self.version = getattr(self, "version", VERSION)
        self.isInitialized = bool(getattr(self, "isInitialized", False))
        self.lastSigintTime = float(getattr(self, "lastSigintTime", 0))
        self.onInputCallback = getattr(self, "onInputCallback", None)
        self.defaultWorkingMessage = getattr(self, "defaultWorkingMessage", "Working...")
        self.workingMessage = getattr(self, "workingMessage", None)
        self.workingVisible = bool(getattr(self, "workingVisible", True))
        self.workingIndicatorOptions = getattr(self, "workingIndicatorOptions", None)
        self.defaultHiddenThinkingLabel = getattr(self, "defaultHiddenThinkingLabel", "Thinking...")
        self.hiddenThinkingLabel = getattr(self, "hiddenThinkingLabel", self.defaultHiddenThinkingLabel)
        self.compactionQueuedMessages = list(getattr(self, "compactionQueuedMessages", []))
        self.pendingBashComponents = list(getattr(self, "pendingBashComponents", []))
        self.bashComponent = getattr(self, "bashComponent", None)
        self.streamingComponent = getattr(self, "streamingComponent", None)
        self.streamingMessage = getattr(self, "streamingMessage", None)
        self.retryEscapeHandler = getattr(self, "retryEscapeHandler", None)
        self.retryCountdown = getattr(self, "retryCountdown", None)
        self.retryLoader = getattr(self, "retryLoader", None)
        self.shutdownRequested = bool(getattr(self, "shutdownRequested", False))
        self.isShuttingDown = bool(getattr(self, "isShuttingDown", False))
        self.signalCleanupHandlers: list[Callable[[], None]] = list(
            getattr(self, "signalCleanupHandlers", [])
        )
        self._shutdownFuture: asyncio.Future[int] | None = None
        self._backgroundTasks: set[asyncio.Task[Any]] = set()
        self._sessionUnsubscribe: Callable[[], None] | None = None
        self._activeSelectorHandle: Any | None = None
        self._toolComponentsById: dict[str, ToolExecutionComponent] = {}
        self._handleClearCount = 0
        self.lastEscapeTime = float(getattr(self, "lastEscapeTime", 0))
        self.changelogMarkdown = getattr(self, "changelogMarkdown", None)
        self.startupNoticesShown = bool(getattr(self, "startupNoticesShown", False))

        if callable(_callable_attr(self.footer, "setAutoCompactEnabled")):
            self.footer.setAutoCompactEnabled(bool(getattr(self.session, "autoCompactionEnabled", False)))

        if runtime_host is not None:
            before_invalidate = _callable_attr(runtime_host, "setBeforeSessionInvalidate")
            if before_invalidate is not None:
                before_invalidate(lambda: self._clear_selector())
            set_rebind = _callable_attr(runtime_host, "setRebindSession")
            if set_rebind is not None:
                set_rebind(self.rebindCurrentSession)

    def _request_render(self, force: bool | None = None) -> None:
        request_render = _callable_attr(self.ui, "requestRender")
        if request_render is None:
            return
        if force is None:
            request_render()
        else:
            request_render(force)

    def prefixAutocompleteDescription(self, description: str | None, source_info: Any = None) -> str | None:
        source_path = _value(source_info, "path")
        if description and source_path:
            return f"{description} [{source_path}]"
        return description or source_path

    def createBaseAutocompleteProvider(self) -> AutocompleteProvider:
        builtin_commands = [
            SlashCommand(name=command.name, description=command.description) for command in BUILTIN_SLASH_COMMANDS
        ]
        builtin_commands.extend(
            SlashCommand(name=command.name, description=command.description) for command in _LOCAL_ALIAS_SLASH_COMMANDS
        )
        model_command = next((command for command in builtin_commands if command.name == "model"), None)
        if model_command is not None:
            model_command.getArgumentCompletions = lambda prefix: _model_argument_completions(
                self.session,
                prefix,
            )

        commands = list(builtin_commands)
        seen_names = {command.name for command in commands}
        self.skillCommands.clear()
        get_slash_commands = _callable_attr(self.session, "getSlashCommands") or _callable_attr(
            self.session, "getCommands"
        )
        if get_slash_commands is not None:
            get_enable_skill_commands = _callable_attr(self.settingsManager, "getEnableSkillCommands")
            skill_commands_enabled = get_enable_skill_commands is None or bool(get_enable_skill_commands())
            skill_paths: dict[str, str] = {}
            resource_loader = getattr(self.session, "resourceLoader", None)
            get_skills = _callable_attr(resource_loader, "getSkills")
            if get_skills is not None:
                skill_result = get_skills() or {}
                for skill in _value(skill_result, "skills", []) or []:
                    skill_name = str(_value(skill, "name", "")).strip()
                    if skill_name:
                        skill_paths[f"skill:{skill_name}"] = str(_value(skill, "filePath", ""))
            for command_info in get_slash_commands() or []:
                command_name = str(_value(command_info, "name", "")).strip()
                if not command_name or command_name in seen_names:
                    continue
                if command_name.startswith("skill:") and not skill_commands_enabled:
                    continue
                seen_names.add(command_name)
                commands.append(
                    SlashCommand(
                        name=command_name,
                        description=self.prefixAutocompleteDescription(
                            _value(command_info, "description"),
                            _value(command_info, "sourceInfo"),
                        ),
                    )
                )
                if command_name in skill_paths:
                    self.skillCommands[command_name] = skill_paths[command_name]

        cwd = str(self.sessionManager.getCwd())
        return CombinedAutocompleteProvider(commands, cwd, self.fdPath)

    def setupAutocompleteProvider(self) -> None:
        provider = self.createBaseAutocompleteProvider()
        for wrap_provider in self.autocompleteProviderWrappers:
            provider = wrap_provider(provider)

        self.autocompleteProvider = provider
        set_default = _callable_attr(self.defaultEditor, "setAutocompleteProvider")
        if set_default is not None:
            set_default(provider)
        if self.editor is not self.defaultEditor:
            set_current = _callable_attr(self.editor, "setAutocompleteProvider")
            if set_current is not None:
                set_current(provider)

    def showStartupNoticesIfNeeded(self) -> None:
        if self.startupNoticesShown:
            return
        self.startupNoticesShown = True

        if not self.changelogMarkdown:
            return

        if getattr(self.chatContainer, "children", []):
            self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DynamicBorder())
        if _safe_call_bool(self.settingsManager, "getCollapseChangelog", True):
            version_match = re.search(r"##\s+\[?(\d+\.\d+\.\d+)\]?", self.changelogMarkdown)
            latest_version = version_match.group(1) if version_match is not None else self.version
            condensed_text = (
                f"Updated to v{latest_version}. "
                f"Use {interactive_theme.theme.bold('/changelog')} to view full changelog."
            )
            self.chatContainer.addChild(Text(condensed_text, 1, 0))
        else:
            self.chatContainer.addChild(
                Text(
                    interactive_theme.theme.bold(interactive_theme.theme.fg("accent", "What's New")),
                    1,
                    0,
                )
            )
            self.chatContainer.addChild(Spacer(1))
            self.chatContainer.addChild(
                Markdown(
                    self.changelogMarkdown.strip(),
                    1,
                    0,
                    self.getMarkdownThemeWithSettings(),
                )
            )
            self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DynamicBorder())
        self._request_render()

    def createExtensionUIContext(self) -> _ExtensionUIContext:
        return _ExtensionUIContext(self)

    def showStatus(self, message: str) -> None:
        children = getattr(self.chatContainer, "children", [])
        last = children[-1] if len(children) > 0 else None
        second_last = children[-2] if len(children) > 1 else None

        if last is self.lastStatusText and second_last is self.lastStatusSpacer and last is not None:
            set_text = _callable_attr(last, "setText")
            if set_text is not None:
                set_text(interactive_theme.theme.fg("dim", message))
            self._request_render()
            return

        spacer = Spacer(1)
        text = Text(interactive_theme.theme.fg("dim", message), 1, 0)
        self.chatContainer.addChild(spacer)
        self.chatContainer.addChild(text)
        self.lastStatusSpacer = spacer
        self.lastStatusText = text
        self._request_render()

    def _append_notice(self, message: str, color: str) -> None:
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(Text(interactive_theme.theme.fg(color, message), 1, 0))
        self._request_render()

    def showError(self, message: str) -> None:
        self._append_notice(message, "error")

    def showWarning(self, message: str) -> None:
        self._append_notice(message, "warning")

    def showNewVersionNotification(self, release: LatestPiRelease) -> None:
        update_instruction = get_update_instruction(release.packageName or PACKAGE_NAME)
        changelog_url = "https://pi.dev/changelog"
        changelog_link = (
            hyperlink(interactive_theme.theme.fg("accent", "open changelog"), changelog_url)
            if getattr(getCapabilities(), "hyperlinks", False)
            else interactive_theme.theme.fg("accent", changelog_url)
        )
        changelog_line = interactive_theme.theme.fg("muted", "Changelog: ") + changelog_link
        note = (release.note or "").strip()

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DynamicBorder(lambda text: interactive_theme.theme.fg("warning", text)))
        self.chatContainer.addChild(
            Text(
                "\n".join(
                    [
                        interactive_theme.theme.bold(interactive_theme.theme.fg("warning", "Update Available")),
                        interactive_theme.theme.fg(
                            "muted",
                            f"New version {release.version} is available. {update_instruction}",
                        ),
                    ]
                ),
                1,
                0,
            )
        )
        if note:
            self.chatContainer.addChild(Spacer(1))
            self.chatContainer.addChild(
                Markdown(
                    note,
                    1,
                    0,
                    self.getMarkdownThemeWithSettings(),
                    DefaultTextStyle(color=lambda text: interactive_theme.theme.fg("muted", text)),
                )
            )
            self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(Text(changelog_line, 1, 0))
        self.chatContainer.addChild(DynamicBorder(lambda text: interactive_theme.theme.fg("warning", text)))
        self._request_render()

    def showPackageUpdateNotification(self, packages: list[str]) -> None:
        package_lines = "\n".join(f"- {package_name}" for package_name in packages)
        action = interactive_theme.theme.fg("accent", f"{APP_NAME} update")
        update_instruction = interactive_theme.theme.fg("muted", "Package updates are available. Run ") + action

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DynamicBorder(lambda text: interactive_theme.theme.fg("warning", text)))
        self.chatContainer.addChild(
            Text(
                "\n".join(
                    [
                        interactive_theme.theme.bold(
                            interactive_theme.theme.fg("warning", "Package Updates Available")
                        ),
                        update_instruction,
                        interactive_theme.theme.fg("muted", "Packages:"),
                        package_lines,
                    ]
                ),
                1,
                0,
            )
        )
        self.chatContainer.addChild(DynamicBorder(lambda text: interactive_theme.theme.fg("warning", text)))
        self._request_render()

    def getAllQueuedMessages(self) -> dict[str, list[str]]:
        get_steering = _callable_attr(self.session, "getSteeringMessages")
        get_follow_up = _callable_attr(self.session, "getFollowUpMessages")
        steering = list(get_steering() or []) if get_steering is not None else []
        follow_up = list(get_follow_up() or []) if get_follow_up is not None else []
        return {
            "steering": [
                *steering,
                *[
                    str(_value(message, "text", ""))
                    for message in self.compactionQueuedMessages
                    if _value(message, "mode") == "steer"
                ],
            ],
            "followUp": [
                *follow_up,
                *[
                    str(_value(message, "text", ""))
                    for message in self.compactionQueuedMessages
                    if _value(message, "mode") == "followUp"
                ],
            ],
        }

    def clearAllQueues(self) -> dict[str, list[str]]:
        clear_queue = _callable_attr(self.session, "clearQueue")
        cleared = clear_queue() if clear_queue is not None else {}
        steering = list(_value(cleared, "steering", []) or [])
        follow_up = list(_value(cleared, "followUp", []) or [])
        compaction_steering = [
            str(_value(message, "text", ""))
            for message in self.compactionQueuedMessages
            if _value(message, "mode") == "steer"
        ]
        compaction_follow_up = [
            str(_value(message, "text", ""))
            for message in self.compactionQueuedMessages
            if _value(message, "mode") == "followUp"
        ]
        self.compactionQueuedMessages = []
        return {
            "steering": [*steering, *compaction_steering],
            "followUp": [*follow_up, *compaction_follow_up],
        }

    def getAppKeyDisplay(self, action: str) -> str:
        return key_display_text(action)

    def updatePendingMessagesDisplay(self) -> None:
        clear = _callable_attr(self.pendingMessagesContainer, "clear")
        if clear is not None:
            clear()
        queued = self.getAllQueuedMessages()
        steering_messages = list(queued.get("steering", []))
        follow_up_messages = list(queued.get("followUp", []))
        if not steering_messages and not follow_up_messages:
            return

        add_child = _callable_attr(self.pendingMessagesContainer, "addChild")
        if add_child is None:
            return

        add_child(Spacer(1))
        for message in steering_messages:
            add_child(TruncatedText(interactive_theme.theme.fg("dim", f"Steering: {message}"), 1, 0))
        for message in follow_up_messages:
            add_child(TruncatedText(interactive_theme.theme.fg("dim", f"Follow-up: {message}"), 1, 0))
        dequeue_hint = self.getAppKeyDisplay("app.message.dequeue")
        add_child(
            TruncatedText(interactive_theme.theme.fg("dim", f"↳ {dequeue_hint} to edit all queued messages"), 1, 0)
        )

    def restoreQueuedMessagesToEditor(self, options: dict[str, Any] | None = None) -> int:
        cleared = self.clearAllQueues()
        all_queued = [*list(cleared.get("steering", [])), *list(cleared.get("followUp", []))]
        if not all_queued:
            self.updatePendingMessagesDisplay()
            if bool(_value(options, "abort", False)):
                abort = _callable_attr(getattr(self.session, "agent", None), "abort")
                if abort is not None:
                    abort()
            return 0

        queued_text = "\n\n".join(all_queued)
        current_text = _value(options, "currentText")
        if current_text is None:
            get_text = _callable_attr(self.editor, "getText")
            current_text = str(get_text() or "") if get_text is not None else ""
        combined_text = "\n\n".join(part for part in (queued_text, str(current_text)) if str(part).strip())
        self._set_editor_text(combined_text)
        self.updatePendingMessagesDisplay()
        if bool(_value(options, "abort", False)):
            abort = _callable_attr(getattr(self.session, "agent", None), "abort")
            if abort is not None:
                abort()
        return len(all_queued)

    def queueCompactionMessage(self, text: str, mode: str) -> None:
        self.compactionQueuedMessages.append({"text": text, "mode": mode})
        add_history = _callable_attr(self.editor, "addToHistory")
        if add_history is not None:
            add_history(text)
        self._set_editor_text("")
        self.updatePendingMessagesDisplay()
        self.showStatus("Queued message for after compaction")

    def isExtensionCommand(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        extension_runner = getattr(self.session, "extensionRunner", None)
        get_command = _callable_attr(extension_runner, "getCommand") or _callable_attr(extension_runner, "get_command")
        if get_command is None:
            return False
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        return bool(get_command(command_name))

    async def flushCompactionQueue(self, options: dict[str, Any] | None = None) -> None:
        if not self.compactionQueuedMessages:
            return

        queued_messages = list(self.compactionQueuedMessages)
        self.compactionQueuedMessages = []
        self.updatePendingMessagesDisplay()
        restored = False

        def restore_queue(error: Exception | str) -> None:
            nonlocal restored
            if restored:
                return
            restored = True
            clear_queue = _callable_attr(self.session, "clearQueue")
            if clear_queue is not None:
                clear_queue()
            self.compactionQueuedMessages = queued_messages
            self.updatePendingMessagesDisplay()
            error_message = error if isinstance(error, str) else str(error)
            suffix = "s" if len(queued_messages) > 1 else ""
            self.showError(f"Failed to send queued message{suffix}: {error_message}")

        async def dispatch_message(message: Any) -> None:
            text = str(_value(message, "text", ""))
            if self.isExtensionCommand(text):
                await self.session.prompt(text)
            elif _value(message, "mode") == "followUp":
                await self.session.followUp(text)
            else:
                await self.session.steer(text)

        try:
            if bool(_value(options, "willRetry", False)):
                for message in queued_messages:
                    await dispatch_message(message)
                self.updatePendingMessagesDisplay()
                return

            first_prompt_index = next(
                (index for index, message in enumerate(queued_messages) if not self.isExtensionCommand(str(_value(message, "text", "")))),
                -1,
            )
            if first_prompt_index == -1:
                for message in queued_messages:
                    await self.session.prompt(str(_value(message, "text", "")))
                return

            pre_commands = queued_messages[:first_prompt_index]
            first_prompt = queued_messages[first_prompt_index]
            rest = queued_messages[first_prompt_index + 1 :]

            for message in pre_commands:
                await self.session.prompt(str(_value(message, "text", "")))

            first_prompt_text = str(_value(first_prompt, "text", ""))
            task = asyncio.create_task(self.session.prompt(first_prompt_text))
            self._backgroundTasks.add(task)

            def _finish_prompt(prompt_task: asyncio.Task[Any]) -> None:
                self._backgroundTasks.discard(prompt_task)
                try:
                    prompt_task.result()
                except asyncio.CancelledError:
                    return
                except Exception as error:  # noqa: BLE001
                    restore_queue(error)

            task.add_done_callback(_finish_prompt)

            for message in rest:
                await dispatch_message(message)
            self.updatePendingMessagesDisplay()
        except Exception as error:  # noqa: BLE001
            restore_queue(error)

    def flushPendingBashComponents(self) -> None:
        remove_child = _callable_attr(self.pendingMessagesContainer, "removeChild")
        add_child = _callable_attr(self.chatContainer, "addChild")
        if add_child is None:
            return
        for component in self.pendingBashComponents:
            if remove_child is not None:
                remove_child(component)
            add_child(component)
        self.pendingBashComponents = []

    def showExtensionNotify(self, message: str, type: str | None = None) -> None:
        if type == "error":
            self.showError(message)
            return
        if type == "warning":
            self.showWarning(message)
            return
        self.showStatus(message)

    def showExtensionError(self, extensionPath: str, error: str, stack: str | None = None) -> None:
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(
            Text(interactive_theme.theme.fg("error", f'Extension "{extensionPath}" error: {error}'), 1, 0)
        )
        if stack:
            stack_lines = [
                interactive_theme.theme.fg("dim", f"  {line.strip()}")
                for line in stack.splitlines()[1:]
                if line.strip()
            ]
            if stack_lines:
                self.chatContainer.addChild(Text("\n".join(stack_lines), 1, 0))
        self._request_render()

    def updateTerminalTitle(self) -> None:
        set_title = _callable_attr(getattr(self.ui, "terminal", None), "setTitle")
        if set_title is None:
            return
        cwd_name = os.path.basename(self.sessionManager.getCwd()) or self.sessionManager.getCwd()
        session_name = self.sessionManager.getSessionName()
        if session_name:
            set_title(f"{APP_TITLE} - {session_name} - {cwd_name}")
            return
        set_title(f"{APP_TITLE} - {cwd_name}")

    def setExtensionStatus(self, key: str, text: str | None = None) -> None:
        self.footerDataProvider.setExtensionStatus(key, text)
        self._request_render()

    def getWorkingLoaderMessage(self) -> str:
        return str(self.workingMessage or self.defaultWorkingMessage)

    def createWorkingLoader(self) -> Loader:
        return Loader(
            self.ui,
            lambda spinner: interactive_theme.theme.fg("accent", spinner),
            lambda text: interactive_theme.theme.fg("muted", text),
            self.getWorkingLoaderMessage(),
            self.workingIndicatorOptions,
        )

    def stopWorkingLoader(self) -> None:
        if self.loadingAnimation is not None:
            stop = _callable_attr(self.loadingAnimation, "stop")
            if stop is not None:
                stop()
            self.loadingAnimation = None
        clear_status = _callable_attr(self.statusContainer, "clear")
        if clear_status is not None:
            clear_status()

    def setWorkingVisible(self, visible: bool) -> None:
        self.workingVisible = visible
        if not visible:
            self.stopWorkingLoader()
            self._request_render()
            return
        if bool(getattr(self.session, "isStreaming", False)) and self.loadingAnimation is None:
            clear_status = _callable_attr(self.statusContainer, "clear")
            if clear_status is not None:
                clear_status()
            self.loadingAnimation = self.createWorkingLoader()
            add_child = _callable_attr(self.statusContainer, "addChild")
            if add_child is not None:
                add_child(self.loadingAnimation)
        self._request_render()

    def setWorkingIndicator(self, options: LoaderIndicatorOptions | None = None) -> None:
        self.workingIndicatorOptions = options
        if self.loadingAnimation is not None:
            self.loadingAnimation.setIndicator(options)
        self._request_render()

    def setHiddenThinkingLabel(self, label: str | None = None) -> None:
        self.hiddenThinkingLabel = label or self.defaultHiddenThinkingLabel
        for child in getattr(self.chatContainer, "children", []):
            set_hidden_label = _callable_attr(child, "setHiddenThinkingLabel")
            if set_hidden_label is not None:
                set_hidden_label(self.hiddenThinkingLabel)
        streaming_component = getattr(self, "streamingComponent", None)
        set_streaming_label = _callable_attr(streaming_component, "setHiddenThinkingLabel")
        if set_streaming_label is not None:
            set_streaming_label(self.hiddenThinkingLabel)
        self._request_render()

    def setExtensionWidget(self, key: str, content: Any, options: dict[str, Any] | None = None) -> None:
        placement = str(_value(options, "placement", "aboveEditor"))

        def _remove_existing(widgets: dict[str, Any]) -> None:
            existing = widgets.pop(key, None)
            dispose = _callable_attr(existing, "dispose")
            if dispose is not None:
                dispose()

        _remove_existing(self.extensionWidgetsAbove)
        _remove_existing(self.extensionWidgetsBelow)

        if content is None:
            self.renderWidgets()
            return

        if isinstance(content, list):
            component = Container()
            for line in content[: self.MAX_WIDGET_LINES]:
                component.addChild(Text(str(line), 1, 0))
            if len(content) > self.MAX_WIDGET_LINES:
                component.addChild(Text(interactive_theme.theme.fg("muted", "... (widget truncated)"), 1, 0))
        else:
            component = content(self.ui, interactive_theme.theme)

        target = self.extensionWidgetsBelow if placement == "belowEditor" else self.extensionWidgetsAbove
        target[key] = component
        self.renderWidgets()

    def clearExtensionWidgets(self) -> None:
        for widget in [*self.extensionWidgetsAbove.values(), *self.extensionWidgetsBelow.values()]:
            dispose = _callable_attr(widget, "dispose")
            if dispose is not None:
                dispose()
        self.extensionWidgetsAbove.clear()
        self.extensionWidgetsBelow.clear()
        self.renderWidgets()

    def renderWidgetContainer(
        self,
        container: Any,
        widgets: dict[str, Any],
        spacerWhenEmpty: bool,
        leadingSpacer: bool,
    ) -> None:
        clear = _callable_attr(container, "clear")
        if clear is not None:
            clear()
        if not widgets:
            if spacerWhenEmpty:
                add_child = _callable_attr(container, "addChild")
                if add_child is not None:
                    add_child(Spacer(1))
            return

        add_child = _callable_attr(container, "addChild")
        if add_child is None:
            return
        if leadingSpacer:
            add_child(Spacer(1))
        for component in widgets.values():
            add_child(component)

    def renderWidgets(self) -> None:
        self.renderWidgetContainer(self.widgetContainerAbove, self.extensionWidgetsAbove, True, True)
        self.renderWidgetContainer(self.widgetContainerBelow, self.extensionWidgetsBelow, False, False)
        self._request_render()

    def setExtensionFooter(self, factory: Any) -> None:
        dispose = _callable_attr(self.customFooter, "dispose")
        if dispose is not None:
            dispose()

        remove_child = _callable_attr(self.ui, "removeChild")
        if remove_child is not None:
            if self.customFooter is not None:
                remove_child(self.customFooter)
            else:
                remove_child(self.footer)

        if factory is not None:
            self.customFooter = factory(self.ui, interactive_theme.theme, self.footerDataProvider)
            add_child = _callable_attr(self.ui, "addChild")
            if add_child is not None:
                add_child(self.customFooter)
        else:
            self.customFooter = None
            add_child = _callable_attr(self.ui, "addChild")
            if add_child is not None:
                add_child(self.footer)
        self._request_render()

    def setExtensionHeader(self, factory: Any) -> None:
        if self.builtInHeader is None:
            return

        dispose = _callable_attr(self.customHeader, "dispose")
        if dispose is not None:
            dispose()

        current_header = self.customHeader or self.builtInHeader
        children = getattr(self.headerContainer, "children", [])
        try:
            index = children.index(current_header)
        except ValueError:
            index = -1

        if factory is not None:
            self.customHeader = factory(self.ui, interactive_theme.theme)
            set_expanded = _callable_attr(self.customHeader, "setExpanded")
            if set_expanded is not None:
                set_expanded(self.toolOutputExpanded)
            if index >= 0:
                children[index] = self.customHeader
            else:
                children.insert(0, self.customHeader)
        else:
            self.customHeader = None
            set_expanded = _callable_attr(self.builtInHeader, "setExpanded")
            if set_expanded is not None:
                set_expanded(self.toolOutputExpanded)
            if index >= 0:
                children[index] = self.builtInHeader

        self._request_render()

    def addExtensionTerminalInputListener(self, handler: Any) -> Callable[[], None]:
        add_input_listener = _callable_attr(self.ui, "addInputListener")
        if add_input_listener is None:
            return lambda: None
        unsubscribe = add_input_listener(handler)
        if callable(unsubscribe):
            self.extensionTerminalInputUnsubscribers.add(unsubscribe)

            def _wrapped_unsubscribe() -> None:
                unsubscribe()
                self.extensionTerminalInputUnsubscribers.discard(unsubscribe)

            return _wrapped_unsubscribe
        return lambda: None

    def clearExtensionTerminalInputListeners(self) -> None:
        for unsubscribe in list(self.extensionTerminalInputUnsubscribers):
            unsubscribe()
        self.extensionTerminalInputUnsubscribers.clear()

    def resetExtensionUI(self) -> None:
        self._clear_selector()
        hide_overlay = _callable_attr(self.ui, "hideOverlay")
        if hide_overlay is not None:
            hide_overlay()
        self.clearExtensionTerminalInputListeners()
        self.setExtensionFooter(None)
        self.setExtensionHeader(None)
        self.clearExtensionWidgets()
        clear_statuses = _callable_attr(self.footerDataProvider, "clearExtensionStatuses")
        if clear_statuses is not None:
            clear_statuses()
        invalidate_footer = _callable_attr(self.footer, "invalidate")
        if invalidate_footer is not None:
            invalidate_footer()
        self.autocompleteProviderWrappers = []
        self.setCustomEditorComponent(None)
        self.setupAutocompleteProvider()
        self.defaultEditor.onExtensionShortcut = None
        if self.editor is not self.defaultEditor and hasattr(self.editor, "onExtensionShortcut"):
            self.editor.onExtensionShortcut = None
        self.updateTerminalTitle()
        self.workingMessage = None
        self.setWorkingIndicator(None)
        self.setWorkingVisible(True)
        self.setHiddenThinkingLabel(None)

    def setCustomEditorComponent(self, factory: Any) -> None:
        self.editorComponentFactory = factory
        get_text = _callable_attr(self.editor, "getText")
        current_text = str(get_text() or "") if get_text is not None else ""

        clear = _callable_attr(self.editorContainer, "clear")
        add_child = _callable_attr(self.editorContainer, "addChild")
        set_focus = _callable_attr(self.ui, "setFocus")
        if clear is not None:
            clear()

        if factory is not None:
            new_editor = factory(self.ui, interactive_theme.get_editor_theme(), self.keybindings)
            if hasattr(new_editor, "onSubmit"):
                new_editor.onSubmit = self.defaultEditor.onSubmit
            if hasattr(new_editor, "onChange"):
                new_editor.onChange = self.defaultEditor.onChange
            set_text = _callable_attr(new_editor, "setText")
            if set_text is not None:
                set_text(current_text)
            if hasattr(new_editor, "borderColor") and hasattr(self.defaultEditor, "borderColor"):
                new_editor.borderColor = self.defaultEditor.borderColor
            default_padding = getattr(self.defaultEditor, "paddingX", None)
            set_padding = _callable_attr(new_editor, "setPaddingX")
            if set_padding is not None and default_padding is not None:
                set_padding(int(default_padding))
            set_provider = _callable_attr(new_editor, "setAutocompleteProvider")
            if set_provider is not None and self.autocompleteProvider is not None:
                set_provider(self.autocompleteProvider)
            action_handlers = getattr(new_editor, "actionHandlers", None)
            default_handlers = getattr(self.defaultEditor, "actionHandlers", None)
            if isinstance(action_handlers, dict):
                if getattr(new_editor, "onEscape", None) is None:
                    new_editor.onEscape = lambda: self.defaultEditor.onEscape() if self.defaultEditor.onEscape else None
                if getattr(new_editor, "onCtrlD", None) is None:
                    new_editor.onCtrlD = lambda: self.defaultEditor.onCtrlD() if self.defaultEditor.onCtrlD else None
                if getattr(new_editor, "onPasteImage", None) is None:
                    new_editor.onPasteImage = (
                        lambda: self.defaultEditor.onPasteImage() if self.defaultEditor.onPasteImage else None
                    )
                if getattr(new_editor, "onExtensionShortcut", None) is None:
                    new_editor.onExtensionShortcut = (
                        lambda data: self.defaultEditor.onExtensionShortcut(data)
                        if self.defaultEditor.onExtensionShortcut is not None
                        else False
                    )
                if isinstance(default_handlers, dict):
                    action_handlers.update(default_handlers)
            self.editor = new_editor
        else:
            default_set_text = _callable_attr(self.defaultEditor, "setText")
            if default_set_text is not None:
                default_set_text(current_text)
            self.editor = self.defaultEditor

        if add_child is not None:
            add_child(self.editor)
        if set_focus is not None:
            set_focus(self.editor)
        self._request_render()

    async def showExtensionCustom(self, factory: Any, options: dict[str, Any] | None = None) -> Any:
        saved_text = self._get_editor_text()
        use_overlay = bool(_value(options, "overlay", False))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def restore_editor() -> None:
            clear = _callable_attr(self.editorContainer, "clear")
            add_child = _callable_attr(self.editorContainer, "addChild")
            set_focus = _callable_attr(self.ui, "setFocus")
            if clear is not None:
                clear()
            if add_child is not None:
                add_child(self.editor)
            self._set_editor_text(saved_text)
            if set_focus is not None:
                set_focus(self.editor)
            self._request_render()

        component: Any = None
        closed = False

        def done(result: Any) -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            if use_overlay:
                hide_overlay = _callable_attr(self.ui, "hideOverlay")
                if hide_overlay is not None:
                    hide_overlay()
            else:
                restore_editor()
            if not future.done():
                future.set_result(result)
            dispose = _callable_attr(component, "dispose")
            if dispose is not None:
                with contextlib.suppress(Exception):
                    dispose()

        try:
            component = await _maybe_await(factory(self.ui, interactive_theme.theme, self.keybindings, done))
            if closed:
                return await future
            if use_overlay:
                overlay_options = _value(options, "overlayOptions")
                resolved_options = overlay_options() if callable(overlay_options) else overlay_options
                handle = self.ui.showOverlay(component, resolved_options or {})
                on_handle = _value(options, "onHandle")
                if callable(on_handle):
                    on_handle(handle)
            else:
                clear = _callable_attr(self.editorContainer, "clear")
                add_child = _callable_attr(self.editorContainer, "addChild")
                set_focus = _callable_attr(self.ui, "setFocus")
                if clear is not None:
                    clear()
                if add_child is not None:
                    add_child(component)
                if set_focus is not None:
                    set_focus(component)
                self._request_render()
        except Exception:
            if not use_overlay:
                restore_editor()
            raise

        return await future

    def setToolsExpanded(self, expanded: bool) -> None:
        self.toolOutputExpanded = expanded
        active_header = self.customHeader or self.builtInHeader
        set_header_expanded = _callable_attr(active_header, "setExpanded")
        if set_header_expanded is not None:
            set_header_expanded(expanded)
        for child in getattr(self.chatContainer, "children", []):
            set_expanded = _callable_attr(child, "setExpanded")
            if set_expanded is not None:
                set_expanded(expanded)
        self._request_render()

    async def maybeWarnAboutAnthropicSubscriptionAuth(self, model: Any | None = None) -> None:
        warnings = {}
        get_warnings = _callable_attr(self.settingsManager, "getWarnings")
        if get_warnings is not None:
            warnings = dict(get_warnings() or {})
        if warnings.get("anthropicExtraUsage") is False:
            return
        if self.anthropicSubscriptionWarningShown:
            return

        resolved_model = model if model is not None else getattr(self.session, "model", None)
        if _value(resolved_model, "provider") != "anthropic":
            return

        model_registry = getattr(self.session, "modelRegistry", None)
        auth_storage = getattr(model_registry, "authStorage", None)
        stored_credential = None
        get_auth = _callable_attr(auth_storage, "get")
        if get_auth is not None:
            stored_credential = get_auth("anthropic")
        if isinstance(stored_credential, dict) and stored_credential.get("type") == "oauth":
            self.anthropicSubscriptionWarningShown = True
            self.showWarning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)
            return

        get_api_key = _callable_attr(model_registry, "getApiKeyForProvider")
        if get_api_key is None:
            return
        try:
            api_key = await _maybe_await(get_api_key("anthropic"))
        except Exception:
            return
        if not is_anthropic_subscription_auth_key(api_key):
            return
        self.anthropicSubscriptionWarningShown = True
        self.showWarning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)

    def handleCtrlZ(self) -> None:
        if sys.platform == "win32":
            self.showStatus("Suspend to background is not supported on Windows")
            return

        keep_alive = threading.Timer(2**30, lambda: None)
        keep_alive.start()

        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigcont = signal.getsignal(signal.SIGCONT)

        def ignore_sigint(_signum: int, _frame: Any) -> None:
            return None

        def resume(_signum: int, _frame: Any) -> None:
            keep_alive.cancel()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGCONT, previous_sigcont)
            start = _callable_attr(self.ui, "start")
            if start is not None:
                start()
            self._request_render(True)

        signal.signal(signal.SIGINT, ignore_sigint)
        signal.signal(signal.SIGCONT, resume)

        try:
            stop = _callable_attr(self.ui, "stop")
            if stop is not None:
                stop()
            os.kill(0, signal.SIGTSTP)
        except Exception:
            keep_alive.cancel()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGCONT, previous_sigcont)
            raise

    def getPathCommandArgument(self, text: str, command: str) -> str | None:
        if text == command:
            return None
        if not text.startswith(f"{command} "):
            return None

        args_string = text[len(command) + 1 :].lstrip()
        if not args_string:
            return None

        first_char = args_string[0]
        if first_char in {'"', "'"}:
            closing_quote_index = args_string.find(first_char, 1)
            if closing_quote_index < 0:
                return None
            return args_string[1:closing_quote_index]

        for index, char in enumerate(args_string):
            if char.isspace():
                return args_string[:index]
        return args_string

    def formatDisplayPath(self, path: str) -> str:
        cwd = self.sessionManager.getCwd()
        try:
            relative = os.path.relpath(path, cwd)
        except ValueError:
            relative = path
        if relative == ".":
            return "."
        if not relative.startswith(f"..{os.sep}") and relative != "..":
            return relative
        home = os.path.expanduser("~")
        if path.startswith(f"{home}{os.sep}"):
            return f"~/{os.path.relpath(path, home)}"
        return path

    def formatContextPath(self, path: str) -> str:
        return os.path.basename(path) or self.formatDisplayPath(path)

    def formatExtensionDisplayPath(self, path: str) -> str:
        return self.formatDisplayPath(path)

    def getShortPath(self, path: str, sourceInfo: Any = None) -> str:
        base_dir = _value(sourceInfo, "baseDir")
        if base_dir:
            try:
                relative = os.path.relpath(path, str(base_dir))
            except ValueError:
                relative = path
            if relative != "." and not relative.startswith(f"..{os.sep}") and relative != "..":
                return relative
        return self.formatDisplayPath(path)

    def _display_source_label(self, sourceInfo: Any = None) -> str:
        source = str(_value(sourceInfo, "source", "local"))
        scope = str(_value(sourceInfo, "scope", "project"))
        if source == "local":
            if scope == "user":
                return "user"
            if scope == "project":
                return "project"
            return "path"
        if source == "cli":
            return "path"
        if scope in {"user", "project", "temporary"}:
            scope_label = "temp" if scope == "temporary" else scope
            return f"{source} ({scope_label})"
        return source

    def _scope_group(self, sourceInfo: Any = None) -> str:
        source = str(_value(sourceInfo, "source", "local"))
        scope = str(_value(sourceInfo, "scope", "project"))
        if source == "cli" or scope == "temporary":
            return "path"
        if scope == "user":
            return "user"
        if scope == "project":
            return "project"
        return "path"

    def _is_package_source(self, sourceInfo: Any = None) -> bool:
        source = str(_value(sourceInfo, "source", ""))
        return source.startswith("npm:") or source.startswith("git:")

    def getCompactExtensionLabels(self, extensions: list[dict[str, Any]]) -> list[str]:
        counts: dict[str, int] = {}
        base_labels: list[str] = []
        for extension in extensions:
            path = str(_value(extension, "path", ""))
            label = Path(path).stem
            if label == "index":
                label = Path(path).parent.name or label
            counts[label] = counts.get(label, 0) + 1
            base_labels.append(label)

        labels: list[str] = []
        for extension, label in zip(extensions, base_labels, strict=False):
            if counts.get(label, 0) > 1:
                labels.append(self.formatExtensionDisplayPath(str(_value(extension, "path", ""))))
            else:
                labels.append(label)
        return labels

    def buildScopeGroups(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups = {
            "project": {"scope": "project", "paths": [], "packages": {}},
            "user": {"scope": "user", "paths": [], "packages": {}},
            "path": {"scope": "path", "paths": [], "packages": {}},
        }
        for item in items:
            source_info = _value(item, "sourceInfo")
            group = groups[self._scope_group(source_info)]
            if self._is_package_source(source_info):
                source = str(_value(source_info, "source", "package"))
                group["packages"].setdefault(source, []).append(item)
            else:
                group["paths"].append(item)
        return [
            group
            for group in (groups["project"], groups["user"], groups["path"])
            if group["paths"] or group["packages"]
        ]

    def formatScopeGroups(self, groups: list[dict[str, Any]], options: dict[str, Any]) -> str:
        lines: list[str] = []
        format_path = options["formatPath"]
        format_package_path = options["formatPackagePath"]

        for group in groups:
            lines.append(f"  {interactive_theme.theme.fg('accent', group['scope'])}")
            for item in sorted(group["paths"], key=lambda entry: str(_value(entry, "path", ""))):
                lines.append(interactive_theme.theme.fg("dim", f"    {format_path(item)}"))
            for source, items in sorted(group["packages"].items()):
                lines.append(f"    {interactive_theme.theme.fg('accent', source)}")
                for item in sorted(items, key=lambda entry: str(_value(entry, "path", ""))):
                    lines.append(interactive_theme.theme.fg("dim", f"      {format_package_path(item, source)}"))
        return "\n".join(lines)

    def findSourceInfoForPath(self, path: str, sourceInfos: dict[str, Any]) -> Any:
        exact = sourceInfos.get(path)
        if exact is not None:
            return exact
        current = path
        while "/" in current:
            current = current.rsplit("/", 1)[0]
            parent = sourceInfos.get(current)
            if parent is not None:
                return parent
        return None

    def formatPathWithSource(self, path: str, sourceInfo: Any = None) -> str:
        if sourceInfo is None:
            return self.formatDisplayPath(path)
        return f"{self._display_source_label(sourceInfo)} {self.getShortPath(path, sourceInfo)}"

    def formatDiagnostics(self, diagnostics: list[Any], sourceInfos: dict[str, Any]) -> str:
        lines: list[str] = []
        collisions: dict[str, list[Any]] = {}
        other_diagnostics: list[Any] = []

        for diagnostic in diagnostics:
            collision = _value(diagnostic, "collision")
            if _value(diagnostic, "type") == "collision" and collision is not None:
                name = str(_value(collision, "name", _value(diagnostic, "message", "collision")))
                collisions.setdefault(name, []).append(diagnostic)
            else:
                other_diagnostics.append(diagnostic)

        for name, entries in collisions.items():
            first_collision = _value(entries[0], "collision")
            if first_collision is None:
                continue
            winner_path = str(_value(first_collision, "winnerPath", ""))
            lines.append(interactive_theme.theme.fg("warning", f'  "{name}" collision:'))
            lines.append(
                interactive_theme.theme.fg(
                    "dim",
                    f"    {interactive_theme.theme.fg('accent', 'winner')} "
                    f"{self.formatPathWithSource(winner_path, self.findSourceInfoForPath(winner_path, sourceInfos))}",
                )
            )
            for diagnostic in entries:
                collision = _value(diagnostic, "collision")
                loser_path = str(_value(collision, "loserPath", ""))
                lines.append(
                    interactive_theme.theme.fg(
                        "dim",
                        f"    {interactive_theme.theme.fg('warning', 'skipped')} "
                        f"{self.formatPathWithSource(loser_path, self.findSourceInfoForPath(loser_path, sourceInfos))}",
                    )
                )

        for diagnostic in other_diagnostics:
            path = _value(diagnostic, "path")
            color = "error" if _value(diagnostic, "type") == "error" else "warning"
            if path:
                formatted_path = self.formatPathWithSource(str(path), self.findSourceInfoForPath(str(path), sourceInfos))
                lines.append(interactive_theme.theme.fg(color, f"  {formatted_path}"))
                lines.append(interactive_theme.theme.fg(color, f"    {_value(diagnostic, 'message', '')}"))
            else:
                lines.append(interactive_theme.theme.fg(color, f"  {_value(diagnostic, 'message', '')}"))
        return "\n".join(lines)

    def showLoadedResources(self, options: dict[str, Any] | None = None) -> None:
        show_listing = bool(
            _value(options, "force", False) or self.options.verbose or not _safe_call_bool(self.settingsManager, "getQuietStartup")
        )
        show_diagnostics = show_listing or bool(_value(options, "showDiagnosticsWhenQuiet", False))
        if not show_listing and not show_diagnostics:
            return

        resource_loader = getattr(self.session, "resourceLoader", None)
        if resource_loader is None:
            return

        get_skills = _callable_attr(resource_loader, "getSkills")
        get_prompts = _callable_attr(resource_loader, "getPrompts")
        get_themes = _callable_attr(resource_loader, "getThemes")
        get_agents_files = _callable_attr(resource_loader, "getAgentsFiles")
        get_extensions = _callable_attr(resource_loader, "getExtensions")

        skills_result = get_skills() if get_skills is not None else {"skills": [], "diagnostics": []}
        prompts_result = get_prompts() if get_prompts is not None else {"prompts": [], "diagnostics": []}
        themes_result = get_themes() if get_themes is not None else {"themes": [], "diagnostics": []}
        extensions_result = (
            get_extensions() if get_extensions is not None else SimpleNamespace(extensions=[], errors=[])
        )
        extensions = _value(options, "extensions")
        if extensions is None:
            extensions = [
                {"path": _value(extension, "path"), "sourceInfo": _value(extension, "sourceInfo")}
                for extension in _value(extensions_result, "extensions", []) or []
            ]

        source_infos: dict[str, Any] = {}
        for extension in extensions:
            source_info = _value(extension, "sourceInfo")
            path = _value(extension, "path")
            if source_info is not None and path:
                source_infos[str(path)] = source_info
        for skill in _value(skills_result, "skills", []) or []:
            source_infos[str(_value(skill, "filePath", ""))] = _value(skill, "sourceInfo")
        for prompt in _value(prompts_result, "prompts", []) or []:
            source_infos[str(_value(prompt, "filePath", ""))] = _value(prompt, "sourceInfo")
        for loaded_theme in _value(themes_result, "themes", []) or []:
            source_path = _value(loaded_theme, "sourcePath")
            source_info = _value(loaded_theme, "sourceInfo")
            if source_path and source_info is not None:
                source_infos[str(source_path)] = source_info

        def add_loaded_section(name: str, collapsed_body: str, expanded_body: str | None = None) -> None:
            body = expanded_body or collapsed_body
            section = ExpandableText(
                lambda: f"{interactive_theme.theme.fg('accent', f'[{name}]')}\n{collapsed_body}",
                lambda: f"{interactive_theme.theme.fg('accent', f'[{name}]')}\n{body}",
                self.getStartupExpansionState(),
                0,
                0,
            )
            self.chatContainer.addChild(section)
            self.chatContainer.addChild(Spacer(1))

        if show_listing:
            context_files = _value(get_agents_files() if get_agents_files is not None else {}, "agentsFiles", []) or []
            if context_files:
                add_loaded_section(
                    "Context",
                    interactive_theme.theme.fg(
                        "dim",
                        "  " + ", ".join(self.formatContextPath(str(_value(item, "path", ""))) for item in context_files),
                    ),
                    "\n".join(
                        interactive_theme.theme.fg("dim", f"  {self.formatDisplayPath(str(_value(item, 'path', '')))}")
                        for item in context_files
                    ),
                )

            skills = _value(skills_result, "skills", []) or []
            if skills:
                skill_items = [
                    {"path": str(_value(skill, "filePath", "")), "sourceInfo": _value(skill, "sourceInfo")}
                    for skill in skills
                ]
                add_loaded_section(
                    "Skills",
                    interactive_theme.theme.fg(
                        "dim",
                        "  " + ", ".join(sorted(str(_value(skill, "name", "")) for skill in skills)),
                    ),
                    self.formatScopeGroups(
                        self.buildScopeGroups(skill_items),
                        {
                            "formatPath": lambda item: self.formatDisplayPath(str(_value(item, "path", ""))),
                            "formatPackagePath": lambda item, _source: self.getShortPath(
                                str(_value(item, "path", "")),
                                _value(item, "sourceInfo"),
                            ),
                        },
                    ),
                )

            templates = list(getattr(self.session, "promptTemplates", []) or [])
            if templates:
                template_by_path = {str(_value(template, "filePath", "")): template for template in templates}
                prompt_items = [
                    {"path": str(_value(template, "filePath", "")), "sourceInfo": _value(template, "sourceInfo")}
                    for template in templates
                ]
                add_loaded_section(
                    "Prompts",
                    interactive_theme.theme.fg(
                        "dim",
                        "  " + ", ".join(sorted(f"/{_value(template, 'name', '')}" for template in templates)),
                    ),
                    self.formatScopeGroups(
                        self.buildScopeGroups(prompt_items),
                        {
                            "formatPath": lambda item: f"/{_value(template_by_path.get(str(_value(item, 'path', ''))), 'name', Path(str(_value(item, 'path', ''))).stem)}",
                            "formatPackagePath": lambda item, _source: f"/{_value(template_by_path.get(str(_value(item, 'path', ''))), 'name', Path(str(_value(item, 'path', ''))).stem)}",
                        },
                    ),
                )

            if extensions:
                add_loaded_section(
                    "Extensions",
                    interactive_theme.theme.fg("dim", "  " + ", ".join(sorted(self.getCompactExtensionLabels(extensions)))),
                    self.formatScopeGroups(
                        self.buildScopeGroups(list(extensions)),
                        {
                            "formatPath": lambda item: self.formatExtensionDisplayPath(str(_value(item, "path", ""))),
                            "formatPackagePath": lambda item, _source: self.formatExtensionDisplayPath(
                                self.getShortPath(str(_value(item, "path", "")), _value(item, "sourceInfo"))
                            ),
                        },
                    ),
                )

            custom_themes = [
                item for item in (_value(themes_result, "themes", []) or []) if _value(item, "sourcePath")
            ]
            if custom_themes:
                theme_items = [
                    {"path": str(_value(item, "sourcePath", "")), "sourceInfo": _value(item, "sourceInfo")}
                    for item in custom_themes
                ]
                add_loaded_section(
                    "Themes",
                    interactive_theme.theme.fg(
                        "dim",
                        "  "
                        + ", ".join(
                            sorted(
                                str(_value(item, "name", Path(str(_value(item, "sourcePath", ""))).stem))
                                for item in custom_themes
                            )
                        ),
                    ),
                    self.formatScopeGroups(
                        self.buildScopeGroups(theme_items),
                        {
                            "formatPath": lambda item: self.formatDisplayPath(str(_value(item, "path", ""))),
                            "formatPackagePath": lambda item, _source: self.getShortPath(
                                str(_value(item, "path", "")),
                                _value(item, "sourceInfo"),
                            ),
                        },
                    ),
                )

        if show_diagnostics:
            diagnostic_sections = [
                ("Skill conflicts", _value(skills_result, "diagnostics", []) or []),
                ("Prompt conflicts", _value(prompts_result, "diagnostics", []) or []),
            ]

            extension_diagnostics: list[Any] = []
            for error in _value(extensions_result, "errors", []) or []:
                extension_diagnostics.append(
                    SimpleNamespace(type="error", path=_value(error, "path"), message=_value(error, "error"))
                )
            get_command_diagnostics = _callable_attr(self.session.extensionRunner, "get_command_diagnostics") or _callable_attr(
                self.session.extensionRunner, "getCommandDiagnostics"
            )
            if get_command_diagnostics is not None:
                extension_diagnostics.extend(get_command_diagnostics() or [])
            get_shortcut_diagnostics = _callable_attr(
                self.session.extensionRunner, "get_shortcut_diagnostics"
            ) or _callable_attr(self.session.extensionRunner, "getShortcutDiagnostics")
            if get_shortcut_diagnostics is not None:
                extension_diagnostics.extend(get_shortcut_diagnostics() or [])
            diagnostic_sections.append(("Extension issues", extension_diagnostics))
            diagnostic_sections.append(("Theme conflicts", _value(themes_result, "diagnostics", []) or []))

            for title, diagnostics in diagnostic_sections:
                if not diagnostics:
                    continue
                formatted = self.formatDiagnostics(list(diagnostics), source_infos)
                if not formatted:
                    continue
                self.chatContainer.addChild(
                    Text(f"{interactive_theme.theme.fg('warning', f'[{title}]')}\n{formatted}", 0, 0)
                )
                self.chatContainer.addChild(Spacer(1))

    def getChangelogForDisplay(self) -> str | None:
        if list(_value(self.session.state, "messages", []) or []):
            return None

        get_last_changelog_version = _callable_attr(self.settingsManager, "getLastChangelogVersion")
        set_last_changelog_version = _callable_attr(self.settingsManager, "setLastChangelogVersion")
        last_seen_value = get_last_changelog_version() if get_last_changelog_version is not None else None
        last_seen_version = last_seen_value.strip() if isinstance(last_seen_value, str) else None
        entries = parse_changelog(get_changelog_path())

        if not last_seen_version:
            if set_last_changelog_version is not None:
                set_last_changelog_version(VERSION)
            self.reportInstallTelemetry(VERSION)
            return None

        new_entries = get_new_entries(entries, last_seen_version)
        if new_entries:
            if set_last_changelog_version is not None:
                set_last_changelog_version(VERSION)
            self.reportInstallTelemetry(VERSION)
            return "\n\n".join(entry.content for entry in new_entries)

        return None

    def reportInstallTelemetry(self, version: str) -> None:
        if os.environ.get("PI_OFFLINE"):
            return

        get_install_telemetry = _callable_attr(self.settingsManager, "getEnableInstallTelemetry")
        if get_install_telemetry is None:
            return

        try:
            if not is_install_telemetry_enabled(self.settingsManager):
                return
        except Exception:
            return

        async def _report() -> None:
            def _send() -> None:
                try:
                    request = Request(
                        f"https://pi.dev/api/report-install?version={quote(version)}",
                        headers={"User-Agent": f"pi/{version}"},
                    )
                    with urlopen(request, timeout=5):
                        return
                except Exception:
                    return

            await asyncio.to_thread(_send)

        self._schedule_task(_report())

    async def checkForPackageUpdates(self) -> list[str]:
        if os.environ.get("PI_OFFLINE"):
            return []

        try:
            package_manager = DefaultPackageManager(
                {
                    "cwd": self.sessionManager.getCwd(),
                    "agentDir": get_agent_dir(),
                    "settingsManager": self.settingsManager,
                }
            )
            updates = await package_manager.checkForAvailableUpdates()
        except Exception:
            return []

        return [
            display_name
            for update in updates
            if (display_name := str(_value(update, "displayName", "")).strip())
        ]

    async def checkTmuxKeyboardSetup(self) -> str | None:
        if not os.environ.get("TMUX"):
            return None

        async def _run_tmux_show(option: str) -> str | None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "tmux",
                    "show",
                    "-gv",
                    option,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception:
                return None

            try:
                stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=2)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                return None

            if proc.returncode != 0:
                return None
            return stdout.decode("utf-8", errors="replace").strip()

        extended_keys, extended_keys_format = await asyncio.gather(
            _run_tmux_show("extended-keys"),
            _run_tmux_show("extended-keys-format"),
        )

        if extended_keys is None:
            return None
        if extended_keys not in {"on", "always"}:
            return (
                "tmux extended-keys is off. Modified Enter keys may not work. Add `set -g extended-keys on` "
                "to ~/.tmux.conf and restart tmux."
            )
        if extended_keys_format == "xterm":
            return (
                "tmux extended-keys-format is xterm. Pi works best with csi-u. "
                "Add `set -g extended-keys-format csi-u` to ~/.tmux.conf and restart tmux."
            )
        return None

    async def promptForMissingSessionCwd(self, error: MissingSessionCwdError) -> str | None:
        confirmed = await _maybe_await(
            self.showExtensionConfirm(
                "Session cwd not found",
                format_missing_session_cwd_prompt(error.issue),
            )
        )
        return error.issue.fallbackCwd if confirmed else None

    async def showExtensionSelector(
        self,
        title: str,
        options: list[str],
        opts: dict[str, Any] | None = None,
    ) -> str | None:
        signal = _value(opts, "signal")
        if _is_signal_aborted(signal):
            return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()

        def finish(value: str | None) -> None:
            if not future.done():
                future.set_result(value)

        def abort() -> None:
            self._clear_selector()
            finish(None)

        unregister_abort = _register_abort_handler(signal, abort)

        def select(option: str) -> None:
            unregister_abort()
            self._clear_selector()
            finish(option)

        def cancel() -> None:
            unregister_abort()
            self._clear_selector()
            finish(None)

        self.showSelector(
            lambda _done: {
                "component": ExtensionSelectorComponent(
                    title,
                    options,
                    select,
                    cancel,
                    {
                        "tui": self.ui,
                        "timeout": _value(opts, "timeout"),
                        "onToggleToolsExpanded": self.toggleToolOutputExpansion,
                    },
                ),
                "focus": True,
            }
        )
        return await future

    async def showExtensionConfirm(
        self,
        title: str,
        message: str,
        opts: dict[str, Any] | None = None,
    ) -> bool:
        result = await self.showExtensionSelector(f"{title}\n{message}", ["Yes", "No"], opts)
        return result == "Yes"

    async def showExtensionInput(
        self,
        title: str,
        placeholder: str | None = None,
        opts: dict[str, Any] | None = None,
    ) -> str | None:
        signal = _value(opts, "signal")
        if _is_signal_aborted(signal):
            return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()

        def finish(value: str | None) -> None:
            if not future.done():
                future.set_result(value)

        def abort() -> None:
            self._clear_selector()
            finish(None)

        unregister_abort = _register_abort_handler(signal, abort)

        def submit(value: str) -> None:
            unregister_abort()
            self._clear_selector()
            finish(value)

        def cancel() -> None:
            unregister_abort()
            self._clear_selector()
            finish(None)

        self.showSelector(
            lambda _done: {
                "component": ExtensionInputComponent(
                    title,
                    placeholder,
                    submit,
                    cancel,
                    {"tui": self.ui, "timeout": _value(opts, "timeout")},
                ),
                "focus": True,
            }
        )
        return await future

    async def showExtensionEditor(self, title: str, prefill: str | None = None) -> str | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()

        def finish(value: str | None) -> None:
            if not future.done():
                future.set_result(value)

        def submit(value: str) -> None:
            self._clear_selector()
            finish(value)

        def cancel() -> None:
            self._clear_selector()
            finish(None)

        self.showSelector(
            lambda _done: {
                "component": ExtensionEditorComponent(
                    self.ui,
                    self.keybindings,
                    title,
                    prefill,
                    submit,
                    cancel,
                ),
                "focus": True,
            }
        )
        return await future

    async def handleFatalRuntimeError(self, prefix: str, error: Exception | BaseException | Any) -> None:
        message = str(error) if error is not None else "Unknown error"
        raise RuntimeError(f"{prefix}: {message}") from (error if isinstance(error, BaseException) else None)

    def rebuildChatFromMessages(self) -> None:
        context_builder = _callable_attr(self.sessionManager, "buildSessionContext")
        context = (
            context_builder()
            if context_builder is not None
            else SimpleNamespace(messages=self.session.state.messages)
        )
        messages = list(_value(context, "messages", getattr(self.session.state, "messages", [])) or [])
        pending_tools: dict[str, ToolExecutionComponent] = {}
        self._toolComponentsById = pending_tools

        for message in messages:
            role = _message_role(message)
            if role == "toolResult":
                tool_call_id = str(_value(message, "toolCallId", ""))
                component = pending_tools.get(tool_call_id)
                if component is not None:
                    component.updateResult(
                        {
                            "content": list(_value(message, "content", []) or []),
                            "details": _value(message, "details"),
                            "isError": bool(_value(message, "isError", False)),
                        },
                        False,
                    )
                continue
            self.addMessageToChat(message, pending_tools)

    def renderCurrentSessionState(self) -> None:
        clear = _callable_attr(self.chatContainer, "clear")
        if clear is not None:
            clear()
        self.lastStatusSpacer = None
        self.lastStatusText = None
        self.rebuildChatFromMessages()
        self._request_render()

    def addMessageToChat(
        self,
        message: Any,
        pendingTools: dict[str, ToolExecutionComponent] | None = None,
    ) -> None:
        role = _message_role(message)
        markdown_theme = self.getMarkdownThemeWithSettings()
        pending_tools = pendingTools if pendingTools is not None else self._toolComponentsById

        if role == "user":
            text = _extract_user_text(message)
            if text:
                self.chatContainer.addChild(UserMessageComponent(text, markdown_theme))
            return

        if role == "assistant":
            assistant = AssistantMessageComponent(
                message,
                self.hideThinkingBlock,
                markdown_theme,
                self.hiddenThinkingLabel,
            )
            self.chatContainer.addChild(assistant)
            for block in list(_value(message, "content", []) or []):
                if _value(block, "type") != "toolCall":
                    continue
                tool_call_id = str(_value(block, "id", ""))
                tool_name = str(_value(block, "name", ""))
                component = ToolExecutionComponent(
                    tool_name,
                    tool_call_id,
                    _value(block, "arguments", {}),
                    {"showImages": _safe_call_bool(self.settingsManager, "getShowImages", True)},
                    _tool_definition(self.session, tool_name),
                    self.ui,
                    self.sessionManager.getCwd(),
                )
                component.markExecutionStarted()
                component.setArgsComplete()
                component.setExpanded(self.toolOutputExpanded)
                pending_tools[tool_call_id] = component
                self.chatContainer.addChild(component)
            return

        if role == "bashExecution":
            component = BashExecutionComponent(
                str(_value(message, "command", "")),
                self.ui,
                bool(_value(message, "excludeFromContext", False)),
            )
            output = str(_value(message, "output", ""))
            if output:
                component.appendOutput(output)
            component.setComplete(
                _value(message, "exitCode"),
                bool(_value(message, "cancelled", False)),
                None,
                _value(message, "fullOutputPath"),
            )
            component.setExpanded(self.toolOutputExpanded)
            self.chatContainer.addChild(component)
            return

        if role == "custom":
            if not bool(_value(message, "display", True)):
                return
            custom_type = str(_value(message, "customType", ""))
            if custom_type == "skill":
                parsed = parse_skill_block(_extract_custom_text(message))
                if parsed is not None:
                    component = SkillInvocationMessageComponent(parsed, markdown_theme)
                    component.setExpanded(self.toolOutputExpanded)
                    self.chatContainer.addChild(component)
                    return
            runner = getattr(self.session, "extensionRunner", None)
            get_renderer = _callable_attr(runner, "get_message_renderer") or _callable_attr(
                runner, "getMessageRenderer"
            )
            renderer = get_renderer(custom_type) if get_renderer is not None else None
            component = CustomMessageComponent(message, renderer, markdown_theme)
            component.setExpanded(self.toolOutputExpanded)
            self.chatContainer.addChild(component)
            return

        if role == "branchSummary":
            component = BranchSummaryMessageComponent(message, markdown_theme)
            component.setExpanded(self.toolOutputExpanded)
            self.chatContainer.addChild(component)
            return

        if role == "compactionSummary":
            component = CompactionSummaryMessageComponent(message, markdown_theme)
            component.setExpanded(self.toolOutputExpanded)
            self.chatContainer.addChild(component)
            return

    async def handleImportCommand(self, text: str) -> None:
        input_path = self.getPathCommandArgument(text, "/import")
        if not input_path:
            self.showError("Usage: /import <path.jsonl>")
            return

        confirmed = await _maybe_await(
            self.showExtensionConfirm("Import session", f"Replace current session with {input_path}?")
        )
        if not confirmed:
            self.showStatus("Import cancelled")
            return

        try:
            if self.loadingAnimation is not None:
                stop = _callable_attr(self.loadingAnimation, "stop")
                if stop is not None:
                    stop()
                self.loadingAnimation = None
            clear = _callable_attr(self.statusContainer, "clear")
            if clear is not None:
                clear()
            result = await self.runtimeHost.importFromJsonl(input_path)
            if result.get("cancelled"):
                self.showStatus("Import cancelled")
                return
            self.renderCurrentSessionState()
            self.showStatus(f"Session imported from: {input_path}")
        except MissingSessionCwdError as error:
            selected_cwd = await _maybe_await(self.promptForMissingSessionCwd(error))
            if not selected_cwd:
                self.showStatus("Import cancelled")
                return
            result = await self.runtimeHost.importFromJsonl(input_path, selected_cwd)
            if result.get("cancelled"):
                self.showStatus("Import cancelled")
                return
            self.renderCurrentSessionState()
            self.showStatus(f"Session imported from: {input_path}")
        except SessionImportFileNotFoundError as error:
            self.showError(f"Failed to import session: {error}")
        except Exception as error:  # noqa: BLE001
            await self.handleFatalRuntimeError("Failed to import session", error)

    async def handleCloneCommand(self) -> None:
        leaf_id = None
        get_leaf_id = _callable_attr(self.sessionManager, "getLeafId")
        if get_leaf_id is not None:
            leaf_id = get_leaf_id()
        if not leaf_id:
            self.showStatus("Nothing to clone yet")
            return

        try:
            result = await self.runtimeHost.fork(leaf_id, {"position": "at"})
            if result.get("cancelled"):
                self._request_render()
                return
            self.renderCurrentSessionState()
            set_text = _callable_attr(self.editor, "setText")
            if set_text is not None:
                set_text("")
            self.showStatus("Cloned to new session")
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))

    def handleChangelogCommand(self) -> None:
        entries = parse_changelog(get_changelog_path())
        changelog_markdown = (
            "\n\n".join(entry.content for entry in reversed(entries))
            if entries
            else "No changelog entries found."
        )

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DynamicBorder())
        self.chatContainer.addChild(
            Text(
                interactive_theme.theme.bold(interactive_theme.theme.fg("accent", "What's New")),
                1,
                0,
            )
        )
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(
            Markdown(
                changelog_markdown,
                1,
                1,
                self.getMarkdownThemeWithSettings(),
            )
        )
        self.chatContainer.addChild(DynamicBorder())
        self._request_render()

    async def handleShareCommand(self) -> None:
        gh_path = shutil.which("gh")
        if gh_path is None:
            self.showError("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/")
            return

        try:
            auth_result = await asyncio.to_thread(
                subprocess.run,
                [gh_path, "auth", "status"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as error:  # noqa: BLE001
            self.showError(f"Failed to check GitHub CLI auth: {error}")
            return
        if auth_result.returncode != 0:
            self.showError("GitHub CLI is not logged in. Run 'gh auth login' first.")
            return

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="session-", suffix=".html", delete=False) as handle:
                tmp_path = handle.name
            await self.session.exportToHtml(tmp_path)
            self.showStatus("Creating gist...")
            result = await asyncio.to_thread(
                subprocess.run,
                [gh_path, "gist", "create", "--public=false", tmp_path],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                message = (result.stderr or "Unknown error").strip()
                self.showError(f"Failed to create gist: {message}")
                return
            gist_url = (result.stdout or "").strip()
            gist_id = gist_url.rsplit("/", 1)[-1] if gist_url else ""
            if not gist_id:
                self.showError("Failed to parse gist ID from gh output")
                return
            self.showStatus(f"Share URL: {get_share_viewer_url(gist_id)}\nGist: {gist_url}")
        except Exception as error:  # noqa: BLE001
            self.showError(f"Failed to create gist: {error}")
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    async def handleCopyCommand(self) -> None:
        get_last_assistant_text = _callable_attr(self.session, "getLastAssistantText")
        text = get_last_assistant_text() if get_last_assistant_text is not None else None
        if not isinstance(text, str) or not text:
            self.showError("No agent messages to copy yet.")
            return
        try:
            await copy_to_clipboard(text)
            self.showStatus("Copied last agent message to clipboard")
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))

    def handleNameCommand(self, text: str) -> None:
        name = re.sub(r"^/name\s*", "", text).strip()
        if not name:
            current_name = self.sessionManager.getSessionName()
            if current_name:
                self.chatContainer.addChild(Spacer(1))
                self.chatContainer.addChild(
                    Text(interactive_theme.theme.fg("dim", f"Session name: {current_name}"), 1, 0)
                )
                self._request_render()
                return
            self.showWarning("Usage: /name <name>")
            return

        self.session.setSessionName(name)
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(Text(interactive_theme.theme.fg("dim", f"Session name set: {name}"), 1, 0))
        self._request_render()

    def handleSessionCommand(self) -> None:
        stats = self.session.getSessionStats()
        lines = [interactive_theme.theme.bold("Session Info"), ""]
        session_name = self.sessionManager.getSessionName()
        if session_name:
            lines.append(f"{interactive_theme.theme.fg('dim', 'Name:')} {session_name}")
        lines.extend(
            [
                f"{interactive_theme.theme.fg('dim', 'File:')} {stats.sessionFile or 'In-memory'}",
                f"{interactive_theme.theme.fg('dim', 'ID:')} {stats.sessionId}",
                "",
                interactive_theme.theme.bold("Messages"),
                f"{interactive_theme.theme.fg('dim', 'User:')} {stats.userMessages}",
                f"{interactive_theme.theme.fg('dim', 'Assistant:')} {stats.assistantMessages}",
                f"{interactive_theme.theme.fg('dim', 'Tool Calls:')} {stats.toolCalls}",
                f"{interactive_theme.theme.fg('dim', 'Tool Results:')} {stats.toolResults}",
                f"{interactive_theme.theme.fg('dim', 'Total:')} {stats.totalMessages}",
                "",
                interactive_theme.theme.bold("Tokens"),
                f"{interactive_theme.theme.fg('dim', 'Input:')} {stats.tokens.input:,}",
                f"{interactive_theme.theme.fg('dim', 'Output:')} {stats.tokens.output:,}",
            ]
        )
        if stats.tokens.cacheRead > 0:
            lines.append(f"{interactive_theme.theme.fg('dim', 'Cache Read:')} {stats.tokens.cacheRead:,}")
        if stats.tokens.cacheWrite > 0:
            lines.append(f"{interactive_theme.theme.fg('dim', 'Cache Write:')} {stats.tokens.cacheWrite:,}")
        lines.append(f"{interactive_theme.theme.fg('dim', 'Total:')} {stats.tokens.total:,}")
        if stats.cost > 0:
            lines.extend(
                [
                    "",
                    interactive_theme.theme.bold("Cost"),
                    f"{interactive_theme.theme.fg('dim', 'Total:')} {stats.cost:.4f}",
                ]
            )

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(Text("\n".join(lines), 1, 0))
        self._request_render()

    def _format_keybinding_display(self, action: str) -> str:
        keys = list(self.keybindings.getKeys(action))
        if not keys:
            return "Unbound"
        return " / ".join(
            key.replace("ctrl+", "Ctrl+")
            .replace("alt+", "Alt+")
            .replace("shift+", "Shift+")
            .replace("pageUp", "PageUp")
            .replace("pageDown", "PageDown")
            for key in keys
        )

    def handleHotkeysCommand(self) -> None:
        sections = [
            ("App", ["app.interrupt", "app.clear", "app.exit", "app.model.select", "app.model.cycleForward"]),
            ("Session", ["app.session.resume", "app.session.tree", "app.session.fork", "app.session.new"]),
            ("Editor", ["tui.input.submit", "tui.input.newLine", "tui.editor.cursorUp", "tui.editor.cursorDown"]),
        ]
        lines = [interactive_theme.theme.bold("Hotkeys"), ""]
        for title, actions in sections:
            lines.append(interactive_theme.theme.bold(title))
            for action in actions:
                definition = KEYBINDINGS.get(action)
                description = definition.description if definition is not None else action
                lines.append(
                    f"{interactive_theme.theme.fg('dim', self._format_keybinding_display(action) + ':')} {description}"
                )
            lines.append("")

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(Text("\n".join(lines).rstrip(), 1, 0))
        self._request_render()

    def handleDebugCommand(self) -> None:
        width = int(getattr(self.ui.terminal, "columns", 0) or 0)
        height = int(getattr(self.ui.terminal, "rows", 0) or 0)
        render = _callable_attr(self.ui, "render")
        all_lines = list(render(width) if render is not None else [])
        messages = list(getattr(self.session, "messages", []) or [])

        def _json_default(value: Any) -> Any:
            if hasattr(value, "__dict__"):
                return value.__dict__
            return str(value)

        debug_data = "\n".join(
            [
                f"Debug output at {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}",
                f"Terminal: {width}x{height}",
                f"Total lines: {len(all_lines)}",
                "",
                "=== All rendered lines with visible widths ===",
                *[
                    f"[{idx}] (w={visibleWidth(line)}) {json.dumps(line)}"
                    for idx, line in enumerate(all_lines)
                ],
                "",
                "=== Agent messages (JSONL) ===",
                *[json.dumps(message, default=_json_default) for message in messages],
                "",
            ]
        )

        debug_log_path = Path(get_debug_log_path())
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_log_path.write_text(debug_data, encoding="utf-8")

        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(
            Text(
                f"{interactive_theme.theme.fg('accent', '✓ Debug log written')}\n"
                f"{interactive_theme.theme.fg('muted', str(debug_log_path))}",
                1,
                1,
            )
        )
        self._request_render()

    def handleArminSaysHi(self) -> None:
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(ArminComponent(self.ui))
        self._request_render()

    def handleDementedDelves(self) -> None:
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(EarendilAnnouncementComponent())
        self._request_render()

    def handleDaxnuts(self) -> None:
        self.chatContainer.addChild(Spacer(1))
        self.chatContainer.addChild(DaxnutsComponent(self.ui))
        self._request_render()

    def checkDaxnutsEasterEgg(self, model: Any) -> None:
        if str(_value(model, "provider", "")) == "opencode" and "kimi-k2.5" in str(_value(model, "id", "")).lower():
            self.handleDaxnuts()

    async def handleResumeSession(
        self,
        sessionPath: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        if self.loadingAnimation is not None:
            stop = _callable_attr(self.loadingAnimation, "stop")
            if stop is not None:
                stop()
            self.loadingAnimation = None
        clear = _callable_attr(self.statusContainer, "clear")
        if clear is not None:
            clear()
        try:
            result = await self.runtimeHost.switchSession(
                sessionPath,
                {"withSession": options.get("withSession")} if options else None,
            )
            if result.get("cancelled"):
                return result
            self.renderCurrentSessionState()
            self.showStatus("Resumed session")
            return result
        except MissingSessionCwdError as error:
            selected_cwd = await self.promptForMissingSessionCwd(error)
            if not selected_cwd:
                self.showStatus("Resume cancelled")
                return {"cancelled": True}
            result = await self.runtimeHost.switchSession(
                sessionPath,
                {
                    "cwdOverride": selected_cwd,
                    "withSession": options.get("withSession") if options else None,
                },
            )
            if result.get("cancelled"):
                return result
            self.renderCurrentSessionState()
            self.showStatus("Resumed session in current cwd")
            return result
        except Exception as error:  # noqa: BLE001
            await self.handleFatalRuntimeError("Failed to resume session", error)
            return {"cancelled": True}

    async def handleNewSession(self) -> dict[str, bool]:
        try:
            result = await self.runtimeHost.newSession()
            if result.get("cancelled"):
                return result
            self.renderCurrentSessionState()
            self.showStatus("Started new session")
            return result
        except Exception as error:  # noqa: BLE001
            await self.handleFatalRuntimeError("Failed to start new session", error)
            return {"cancelled": True}

    async def handleClearCommand(self) -> None:
        if self.loadingAnimation is not None:
            stop = _callable_attr(self.loadingAnimation, "stop")
            if stop is not None:
                stop()
            self.loadingAnimation = None
        clear = _callable_attr(self.statusContainer, "clear")
        if clear is not None:
            clear()
        try:
            result = await self.runtimeHost.newSession()
            if result.get("cancelled"):
                return
            self.renderCurrentSessionState()
            self.chatContainer.addChild(Spacer(1))
            self.chatContainer.addChild(
                Text(f"{interactive_theme.theme.fg('accent', '✓ New session started')}", 1, 1)
            )
            self._request_render()
        except Exception as error:  # noqa: BLE001
            await self.handleFatalRuntimeError("Failed to create session", error)

    async def handleBashCommand(self, command: str, excludeFromContext: bool = False) -> None:
        original_escape = getattr(self.defaultEditor, "onEscape", None)
        bash_component = BashExecutionComponent(command, self.ui, excludeFromContext)
        bash_component.setExpanded(self.toolOutputExpanded)
        is_deferred = bool(getattr(self.session, "isStreaming", False))
        if is_deferred:
            self.pendingMessagesContainer.addChild(bash_component)
            self.pendingBashComponents.append(bash_component)
        else:
            self.chatContainer.addChild(bash_component)
        self.bashComponent = bash_component
        self.defaultEditor.onEscape = lambda: _callable_attr(self.session, "abortBash") and self.session.abortBash()
        self._request_render()
        try:
            result = await self.session.executeBash(
                command,
                bash_component.appendOutput,
                {"excludeFromContext": excludeFromContext},
            )
            bash_component.setComplete(
                _value(result, "exitCode"),
                bool(_value(result, "cancelled", False)),
                None,
                _value(result, "fullOutputPath"),
            )
        except Exception as error:  # noqa: BLE001
            bash_component.setComplete(None, False)
            self.showError(f"Bash command failed: {error}")
        finally:
            self.defaultEditor.onEscape = original_escape
            self.bashComponent = None
            self._request_render()

    async def handleSubmittedText(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if text == "/resume":
            self._set_editor_text("")
            self.showSessionSelector()
            return
        if text == "/changelog":
            self._set_editor_text("")
            self.handleChangelogCommand()
            return
        if text == "/model" or text.startswith("/model "):
            search_term = text[7:].strip() if text.startswith("/model ") else None
            self._set_editor_text("")
            await self.handleModelCommand(search_term or None)
            return
        if text == "/scoped-models":
            self._set_editor_text("")
            await self.showModelsSelector()
            return
        if text == "/models":
            self._set_editor_text("")
            await self.showModelsSelector()
            return
        if text == "/settings":
            self._set_editor_text("")
            self.showSettingsSelector()
            return
        if text == "/export" or text.startswith("/export "):
            self._set_editor_text("")
            await self.handleExportCommand(text)
            return
        if text == "/theme":
            self._set_editor_text("")
            self.showThemeSelector()
            return
        if text.startswith("/theme "):
            self._set_editor_text("")
            await self.handleThemeCommand(text[7:].strip())
            return
        if text == "/import" or text.startswith("/import "):
            await self.handleImportCommand(text)
            self._set_editor_text("")
            return
        if text == "/clone":
            self._set_editor_text("")
            await self.handleCloneCommand()
            return
        if text == "/share":
            self._set_editor_text("")
            await self.handleShareCommand()
            return
        if text == "/copy":
            self._set_editor_text("")
            await self.handleCopyCommand()
            return
        if text == "/name" or text.startswith("/name "):
            self._set_editor_text("")
            self.handleNameCommand(text)
            return
        if text == "/session":
            self._set_editor_text("")
            self.handleSessionCommand()
            return
        if text == "/hotkeys":
            self._set_editor_text("")
            self.handleHotkeysCommand()
            return
        if text == "/fork":
            self._set_editor_text("")
            self.showUserMessageSelector()
            return
        if text == "/tree":
            self._set_editor_text("")
            self.showTreeSelector()
            return
        if text == "/login":
            self._set_editor_text("")
            await self.showOAuthSelector("login")
            return
        if text == "/logout":
            self._set_editor_text("")
            await self.showOAuthSelector("logout")
            return
        if text == "/new":
            self._set_editor_text("")
            await self.handleClearCommand()
            return
        if text == "/compact" or text.startswith("/compact "):
            custom_instructions = text[9:].strip() if text.startswith("/compact ") else None
            self._set_editor_text("")
            await self.handleCompactCommand(custom_instructions or None)
            return
        if text == "/reload":
            self._set_editor_text("")
            await self.handleReloadCommand()
            return
        if text == "/debug":
            self._set_editor_text("")
            self.handleDebugCommand()
            return
        if text == "/arminsayshi":
            self._set_editor_text("")
            self.handleArminSaysHi()
            return
        if text == "/dementedelves":
            self._set_editor_text("")
            self.handleDementedDelves()
            return
        if text == "/quit":
            self._set_editor_text("")
            await self.shutdown()
            return

        if text.startswith("!"):
            is_excluded = text.startswith("!!")
            command = text[2:].strip() if is_excluded else text[1:].strip()
            if command:
                if bool(getattr(self.session, "isBashRunning", False)):
                    self.showWarning("A bash command is already running. Press Esc to cancel it first.")
                    self._set_editor_text(text)
                    return
                add_history = _callable_attr(self.editor, "addToHistory")
                if add_history is not None:
                    add_history(text)
                self._set_editor_text("")
                await self.handleBashCommand(command, is_excluded)
                self.updateEditorBorderColor()
            return

        if bool(getattr(self.session, "isCompacting", False)):
            if self.isExtensionCommand(text):
                add_history = _callable_attr(self.editor, "addToHistory")
                if add_history is not None:
                    add_history(text)
                self._set_editor_text("")
                await self.session.prompt(text)
            else:
                self.queueCompactionMessage(text, "steer")
            return

        if bool(getattr(self.session, "isStreaming", False)):
            add_history = _callable_attr(self.editor, "addToHistory")
            if add_history is not None:
                add_history(text)
            self._set_editor_text("")
            await self.session.prompt(text, {"streamingBehavior": "steer"})
            self.updatePendingMessagesDisplay()
            self._request_render()
            return

        self.flushPendingBashComponents()
        if self.onInputCallback is not None:
            self.onInputCallback(text)
        add_history = _callable_attr(self.editor, "addToHistory")
        if add_history is not None:
            add_history(text)
        self._set_editor_text("")
        await self.session.prompt(text)

    async def handleThemeCommand(self, themeName: str) -> None:
        if not themeName:
            self.showThemeSelector()
            return
        result = interactive_theme.set_theme(themeName, True)
        if not result.get("success"):
            self.showError(str(result.get("error", f"Theme not found: {themeName}")))
            return
        set_theme = _callable_attr(self.settingsManager, "setTheme")
        if set_theme is not None:
            set_theme(themeName)
        self.updateEditorBorderColor()
        self._request_render()
        self.showStatus(f"Theme: {themeName}")

    async def handleReloadCommand(self) -> None:
        if bool(getattr(self.session, "isStreaming", False)):
            self.showWarning("Wait for the current response to finish before reloading.")
            return
        if bool(getattr(self.session, "isCompacting", False)):
            self.showWarning("Wait for compaction to finish before reloading.")
            return

        previous_editor = self.editor
        reload_box = Container()
        reload_box.addChild(DynamicBorder())
        reload_box.addChild(Spacer(1))
        reload_box.addChild(
            Text(
                interactive_theme.theme.fg(
                    "muted",
                    "Reloading keybindings, extensions, skills, prompts, and themes...",
                ),
                1,
                0,
            )
        )
        reload_box.addChild(Spacer(1))
        reload_box.addChild(DynamicBorder())

        self.editorContainer.clear()
        self.editorContainer.addChild(reload_box)
        set_focus = _callable_attr(self.ui, "setFocus")
        if set_focus is not None:
            set_focus(reload_box)
        self._request_render(True)
        await asyncio.sleep(0)

        def dismiss(editor: Any) -> None:
            self.editorContainer.clear()
            self.editorContainer.addChild(editor)
            if set_focus is not None:
                set_focus(editor)
            self._request_render()

        try:
            await self.session.reload()
            self.keybindings.reload()
            setKeybindings(self.keybindings)
            await self.rebindCurrentSession()
            dismiss(self.editor)
            self.showStatus("Reloaded keybindings, extensions, skills, prompts, themes")
        except Exception as error:  # noqa: BLE001
            dismiss(previous_editor)
            self.showError(f"Reload failed: {error}")

    async def handleExportCommand(self, text: str) -> None:
        output_path = self.getPathCommandArgument(text, "/export")
        try:
            if output_path and output_path.endswith(".jsonl"):
                file_path = self.session.exportToJsonl(output_path)
            else:
                file_path = await self.session.exportToHtml(output_path)
            self.showStatus(f"Session exported to: {file_path}")
        except Exception as error:  # noqa: BLE001
            self.showError(f"Failed to export session: {error}")

    def getLoginProviderOptions(self, authType: str | None = None) -> list[AuthSelectorProvider]:
        auth_storage = self.session.modelRegistry.authStorage
        oauth_providers = list(auth_storage.getOAuthProviders())
        oauth_provider_ids = {provider.id for provider in oauth_providers}
        options = [
            AuthSelectorProvider(id=str(provider.id), name=str(provider.name), authType="oauth")
            for provider in oauth_providers
        ]

        get_all = _callable_attr(self.session.modelRegistry, "getAll")
        model_providers = {
            str(_value(model, "provider", ""))
            for model in (get_all() or [])
            if _value(model, "provider", "")
        }
        for provider_id in model_providers:
            if provider_id in oauth_provider_ids:
                continue
            options.append(
                AuthSelectorProvider(
                    id=provider_id,
                    name=self.session.modelRegistry.getProviderDisplayName(provider_id),
                    authType="api_key",
                )
            )

        filtered = [option for option in options if authType is None or option.authType == authType]
        return sorted(filtered, key=lambda option: option.name.lower())

    def getLogoutProviderOptions(self) -> list[AuthSelectorProvider]:
        auth_storage = self.session.modelRegistry.authStorage
        options: list[AuthSelectorProvider] = []
        for provider_id in auth_storage.list():
            credential = auth_storage.get(provider_id)
            if not isinstance(credential, dict):
                continue
            options.append(
                AuthSelectorProvider(
                    id=provider_id,
                    name=self.session.modelRegistry.getProviderDisplayName(provider_id),
                    authType=str(credential.get("type", "api_key")),
                )
            )
        return sorted(options, key=lambda option: option.name.lower())

    def showLoginAuthTypeSelector(self) -> None:
        subscription_label = "Use a subscription"
        api_key_label = "Use an API key"
        self.showSelector(
            lambda done: {
                "component": ExtensionSelectorComponent(
                    "Select authentication method:",
                    [subscription_label, api_key_label],
                    lambda option: (
                        done(),
                        self.showLoginProviderSelector("oauth" if option == subscription_label else "api_key"),
                    ),
                    lambda: (done(), self._request_render()),
                ),
                "focus": True,
            }
        )

    def showLoginProviderSelector(self, authType: str) -> None:
        provider_options = self.getLoginProviderOptions(authType)
        if not provider_options:
            self.showStatus(
                "No subscription providers available."
                if authType == "oauth"
                else "No API key providers available."
            )
            return

        self.showSelector(
            lambda done: {
                "component": OAuthSelectorComponent(
                    "login",
                    self.session.modelRegistry.authStorage,
                    provider_options,
                    lambda provider_id: self._schedule_task(
                        self._handle_login_provider_select(provider_options, provider_id, done)
                    ),
                    lambda: (done(), self.showLoginAuthTypeSelector()),
                    lambda provider_id: self.session.modelRegistry.getProviderAuthStatus(provider_id),
                ),
                "focus": True,
            }
        )

    async def _handle_login_provider_select(
        self,
        provider_options: list[AuthSelectorProvider],
        provider_id: str,
        done: Callable[[], None],
    ) -> None:
        done()
        provider = next((item for item in provider_options if item.id == provider_id), None)
        if provider is None:
            return
        if provider.authType == "oauth":
            await self.showLoginDialog(provider.id, provider.name)
            return
        if provider.id == _BEDROCK_PROVIDER_ID:
            self.showBedrockSetupDialog(provider.id, provider.name)
            return
        await self.showApiKeyLoginDialog(provider.id, provider.name)

    async def showOAuthSelector(self, mode: str) -> None:
        if mode == "login":
            self.showLoginAuthTypeSelector()
            return

        provider_options = self.getLogoutProviderOptions()
        if not provider_options:
            self.showStatus(
                "No stored credentials to remove. /logout only removes credentials saved by /login; "
                "environment variables and models.json config are unchanged."
            )
            return

        self.showSelector(
            lambda done: {
                "component": OAuthSelectorComponent(
                    mode,
                    self.session.modelRegistry.authStorage,
                    provider_options,
                    lambda provider_id: self._schedule_task(
                        self._handle_logout_provider_select(provider_options, provider_id, done)
                    ),
                    lambda: (done(), self._request_render()),
                ),
                "focus": True,
            }
        )

    async def _handle_logout_provider_select(
        self,
        provider_options: list[AuthSelectorProvider],
        provider_id: str,
        done: Callable[[], None],
    ) -> None:
        done()
        provider = next((item for item in provider_options if item.id == provider_id), None)
        if provider is None:
            return
        try:
            self.session.modelRegistry.authStorage.logout(provider.id)
            self.session.modelRegistry.refresh()
            self.updateAvailableProviderCount()
            message = (
                f"Logged out of {provider.name}"
                if provider.authType == "oauth"
                else (
                    f"Removed stored API key for {provider.name}. Environment variables and models.json config are "
                    "unchanged."
                )
            )
            self.showStatus(message)
        except Exception as error:  # noqa: BLE001
            self.showError(f"Logout failed: {error}")

    async def completeProviderAuthentication(
        self,
        provider_id: str,
        provider_name: str,
        auth_type: str,
        previous_model: Any = None,
    ) -> None:
        self.session.modelRegistry.refresh()
        action_label = f"Logged in to {provider_name}" if auth_type == "oauth" else f"Saved API key for {provider_name}"

        selected_model = None
        selection_error: str | None = None
        if _is_unknown_model(previous_model):
            available_models = list(await _maybe_await(self.session.modelRegistry.getAvailable()))
            provider_models = [model for model in available_models if _value(model, "provider") == provider_id]
            if provider_id not in defaultModelPerProvider:
                selection_error = (
                    f'{action_label}, but no default model is configured for provider "{provider_id}". '
                    "Use /model to select a model."
                )
            elif not provider_models:
                selection_error = (
                    f"{action_label}, but no models are available for that provider. "
                    "Use /model to select a model."
                )
            else:
                default_model_id = defaultModelPerProvider[provider_id]
                selected_model = next(
                    (model for model in provider_models if _value(model, "id") == default_model_id),
                    None,
                )
                if selected_model is None:
                    selection_error = (
                        f'{action_label}, but its default model "{default_model_id}" is not available. '
                        "Use /model to select a model."
                    )
                else:
                    try:
                        await _maybe_await(self.session.setModel(selected_model))
                    except Exception as error:  # noqa: BLE001
                        selected_model = None
                        selection_error = (
                            f"{action_label}, but selecting its default model failed: {error}. "
                            "Use /model to select a model."
                        )

        self.updateAvailableProviderCount()
        self.footer.invalidate()
        self.updateEditorBorderColor()
        if selected_model is not None:
            self.showStatus(
                f"{action_label}. Selected {_value(selected_model, 'id')}. Credentials saved to {get_auth_path()}"
            )
            await self.maybeWarnAboutAnthropicSubscriptionAuth(selected_model)
            self.checkDaxnutsEasterEgg(selected_model)
            return

        self.showStatus(f"{action_label}. Credentials saved to {get_auth_path()}")
        if selection_error is not None:
            self.showError(selection_error)
        else:
            await self.maybeWarnAboutAnthropicSubscriptionAuth()

    def showBedrockSetupDialog(self, providerId: str, providerName: str) -> None:
        set_focus = _callable_attr(self.ui, "setFocus")

        def restore_editor() -> None:
            self.editorContainer.clear()
            self.editorContainer.addChild(self.editor)
            if set_focus is not None:
                set_focus(self.editor)
            self._request_render()

        dialog = LoginDialogComponent(
            self.ui,
            providerId,
            lambda _success, _message: restore_editor(),
            providerName,
            "Amazon Bedrock setup",
        )
        dialog.showInfo(
            [
                interactive_theme.theme.fg(
                    "text", "Amazon Bedrock uses AWS credentials instead of a single API key."
                ),
                interactive_theme.theme.fg(
                    "text", "Configure an AWS profile, IAM keys, bearer token, or role-based credentials."
                ),
                interactive_theme.theme.fg("muted", "See:"),
                interactive_theme.theme.fg("accent", f"  {os.path.join(get_docs_path(), 'providers.md')}"),
            ]
        )

        self.editorContainer.clear()
        self.editorContainer.addChild(dialog)
        if set_focus is not None:
            set_focus(dialog)
        self._request_render()

    async def showApiKeyLoginDialog(self, providerId: str, providerName: str) -> None:
        previous_model = getattr(self.session, "model", None)
        dialog = LoginDialogComponent(self.ui, providerId, lambda _success, _message: None, providerName)
        self.editorContainer.clear()
        self.editorContainer.addChild(dialog)
        set_focus = _callable_attr(self.ui, "setFocus")
        if set_focus is not None:
            set_focus(dialog)
        self._request_render()

        def restore_editor() -> None:
            self.editorContainer.clear()
            self.editorContainer.addChild(self.editor)
            if set_focus is not None:
                set_focus(self.editor)
            self._request_render()

        try:
            api_key = str((await dialog.showPrompt("Enter API key:")).strip())
            if not api_key:
                raise RuntimeError("API key cannot be empty.")
            self.session.modelRegistry.authStorage.set(providerId, {"type": "api_key", "key": api_key})
            restore_editor()
            await self.completeProviderAuthentication(providerId, providerName, "api_key", previous_model)
        except Exception as error:  # noqa: BLE001
            restore_editor()
            if str(error) != "Login cancelled":
                self.showError(f"Failed to save API key for {providerName}: {error}")

    def showOAuthLoginSelect(self, dialog: LoginDialogComponent, prompt: Any) -> Awaitable[str | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()

        def restore_dialog() -> None:
            self.editorContainer.clear()
            self.editorContainer.addChild(dialog)
            set_focus = _callable_attr(self.ui, "setFocus")
            if set_focus is not None:
                set_focus(dialog)
            self._request_render()

        labels = [str(_value(option, "label", "")) for option in _value(prompt, "options", []) or []]
        selector = ExtensionSelectorComponent(
            str(_value(prompt, "message", "")),
            labels,
            lambda option_label: (
                restore_dialog(),
                future.set_result(
                    next(
                        (
                            str(_value(option, "id"))
                            for option in _value(prompt, "options", []) or []
                            if str(_value(option, "label", "")) == option_label
                        ),
                        None,
                    )
                ),
            ),
            lambda: (restore_dialog(), future.set_result(None)),
        )
        self.editorContainer.clear()
        self.editorContainer.addChild(selector)
        set_focus = _callable_attr(self.ui, "setFocus")
        if set_focus is not None:
            set_focus(selector)
        self._request_render()
        return future

    async def showLoginDialog(self, providerId: str, providerName: str) -> None:
        previous_model = getattr(self.session, "model", None)
        dialog = LoginDialogComponent(self.ui, providerId, lambda _success, _message: None, providerName)
        self.editorContainer.clear()
        self.editorContainer.addChild(dialog)
        set_focus = _callable_attr(self.ui, "setFocus")
        if set_focus is not None:
            set_focus(dialog)
        self._request_render()

        manual_code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def restore_editor() -> None:
            self.editorContainer.clear()
            self.editorContainer.addChild(self.editor)
            if set_focus is not None:
                set_focus(self.editor)
            self._request_render()

        try:
            def _handle_auth(info: Any) -> None:
                dialog.showAuth(
                    str(_value(info, "url", "")),
                    _value(info, "instructions"),
                )

                async def _collect_manual_code() -> None:
                    try:
                        value = await dialog.showManualInput(
                            "Paste redirect URL below, or complete login in browser:"
                        )
                        if not manual_code_future.done():
                            manual_code_future.set_result(value)
                    except Exception as error:  # noqa: BLE001
                        if not manual_code_future.done():
                            manual_code_future.set_exception(
                                error if isinstance(error, Exception) else RuntimeError(str(error))
                            )

                self._schedule_task(_collect_manual_code())

            await self.session.modelRegistry.authStorage.login(
                providerId,
                {
                    "onAuth": _handle_auth,
                    "onDeviceCode": dialog.showDeviceCode,
                    "onPrompt": lambda prompt: dialog.showPrompt(
                        str(_value(prompt, "message", "")),
                        _value(prompt, "placeholder"),
                    ),
                    "onProgress": dialog.showProgress,
                    "onSelect": lambda prompt: self.showOAuthLoginSelect(dialog, prompt),
                    "onManualCodeInput": lambda: manual_code_future,
                    "signal": dialog.signal,
                },
            )
            restore_editor()
            await self.completeProviderAuthentication(providerId, providerName, "oauth", previous_model)
        except Exception as error:  # noqa: BLE001
            restore_editor()
            if str(error) != "Login cancelled":
                self.showError(f"Failed to login to {providerName}: {error}")

    async def handleCompactCommand(self, customInstructions: str | None = None) -> None:
        entries = self.sessionManager.getEntries()
        message_count = sum(1 for entry in entries if entry.get("type") == "message")
        if message_count < 2:
            self.showWarning("Nothing to compact (no messages yet)")
            return
        if self.loadingAnimation is not None:
            stop = _callable_attr(self.loadingAnimation, "stop")
            if stop is not None:
                stop()
            self.loadingAnimation = None
        clear_status = _callable_attr(self.statusContainer, "clear")
        if clear_status is not None:
            clear_status()
        try:
            await self.session.compact(customInstructions)
        except Exception:
            return

    async def checkShutdownRequested(self) -> None:
        if not self.shutdownRequested:
            return
        await self.shutdown()

    async def handleEvent(self, event: dict[str, Any] | Any) -> None:
        if not self.isInitialized:
            await self.init()

        event_type = _value(event, "type")
        if event_type == "agent_start":
            self._toolComponentsById.clear()
            get_progress = _callable_attr(self.settingsManager, "getShowTerminalProgress")
            terminal = getattr(self.ui, "terminal", None)
            set_progress = _callable_attr(terminal, "setProgress")
            if get_progress is not None and bool(get_progress()) and set_progress is not None:
                set_progress(True)

            if self.retryEscapeHandler is not None:
                self.defaultEditor.onEscape = self.retryEscapeHandler
                self.retryEscapeHandler = None
            if self.retryCountdown is not None:
                dispose = _callable_attr(self.retryCountdown, "dispose")
                if dispose is not None:
                    dispose()
                self.retryCountdown = None
            if self.retryLoader is not None:
                stop = _callable_attr(self.retryLoader, "stop")
                if stop is not None:
                    stop()
                self.retryLoader = None
            self.stopWorkingLoader()
            if self.workingVisible:
                self.loadingAnimation = self.createWorkingLoader()
                add_child = _callable_attr(self.statusContainer, "addChild")
                if add_child is not None:
                    add_child(self.loadingAnimation)
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "queue_update":
            self.updatePendingMessagesDisplay()
            self._request_render()
            return
        if event_type == "compaction_start":
            get_progress = _callable_attr(self.settingsManager, "getShowTerminalProgress")
            terminal = getattr(self.ui, "terminal", None)
            set_progress = _callable_attr(terminal, "setProgress")
            if get_progress is not None and bool(get_progress()) and set_progress is not None:
                set_progress(True)

            self.autoCompactionEscapeHandler = getattr(self.defaultEditor, "onEscape", None)
            self.defaultEditor.onEscape = lambda: _callable_attr(self.session, "abortCompaction") and self.session.abortCompaction()
            clear_status = _callable_attr(self.statusContainer, "clear")
            if clear_status is not None:
                clear_status()
            cancel_hint = f"({key_text('app.interrupt')} to cancel)"
            reason = str(_value(event, "reason", "manual"))
            label = (
                f"Compacting context... {cancel_hint}"
                if reason == "manual"
                else f"{'Context overflow detected, ' if reason == 'overflow' else ''}Auto-compacting... {cancel_hint}"
            )
            self.autoCompactionLoader = Loader(
                self.ui,
                lambda spinner: interactive_theme.theme.fg("accent", spinner),
                lambda text: interactive_theme.theme.fg("muted", text),
                label,
            )
            add_child = _callable_attr(self.statusContainer, "addChild")
            if add_child is not None:
                add_child(self.autoCompactionLoader)
            self._request_render()
            return
        if event_type == "message_start":
            message = _value(event, "message")
            role = _message_role(message)
            if role == "custom":
                self.addMessageToChat(message)
                self.footer.invalidate()
                self._request_render()
                return
            if role == "user":
                self.addMessageToChat(message)
                self.updatePendingMessagesDisplay()
                self.footer.invalidate()
                self._request_render()
                return
            if role == "assistant":
                self.streamingComponent = AssistantMessageComponent(
                    None,
                    self.hideThinkingBlock,
                    self.getMarkdownThemeWithSettings(),
                    self.hiddenThinkingLabel,
                )
                self.streamingMessage = message
                self.chatContainer.addChild(self.streamingComponent)
                self.streamingComponent.updateContent(message)
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "message_update":
            message = _value(event, "message")
            if self.streamingComponent is not None and _message_role(message) == "assistant":
                self.streamingMessage = message
                self.streamingComponent.updateContent(message)

                for content in list(_value(message, "content", []) or []):
                    if _value(content, "type") != "toolCall":
                        continue
                    tool_call_id = str(_value(content, "id", ""))
                    component = self._toolComponentsById.get(tool_call_id)
                    if component is None:
                        component = ToolExecutionComponent(
                            str(_value(content, "name", "")),
                            tool_call_id,
                            _value(content, "arguments", {}),
                            {
                                "showImages": _safe_call_bool(self.settingsManager, "getShowImages", True),
                                "imageWidthCells": _safe_call_int(self.settingsManager, "getImageWidthCells", 56),
                            },
                            _tool_definition(self.session, str(_value(content, "name", ""))),
                            self.ui,
                            self.sessionManager.getCwd(),
                        )
                        component.setExpanded(self.toolOutputExpanded)
                        self.chatContainer.addChild(component)
                        self._toolComponentsById[tool_call_id] = component
                    else:
                        component.updateArgs(_value(content, "arguments", {}))
                self.footer.invalidate()
                self._request_render()
            return
        if event_type == "message_end":
            message = _value(event, "message")
            if _message_role(message) == "user":
                return
            if self.loadingAnimation is not None:
                self.stopWorkingLoader()
            if self.streamingComponent is not None and _message_role(message) == "assistant":
                self.streamingMessage = message
                error_message: str | None = None
                if _value(message, "stopReason") == "aborted":
                    retry_attempt = int(getattr(self.session, "retryAttempt", 0) or 0)
                    error_message = (
                        f"Aborted after {retry_attempt} retry attempt{'s' if retry_attempt > 1 else ''}"
                        if retry_attempt > 0
                        else "Operation aborted"
                    )
                    if isinstance(message, dict):
                        message["errorMessage"] = error_message
                    else:
                        setattr(message, "errorMessage", error_message)
                self.streamingComponent.updateContent(message)

                if _value(message, "stopReason") in {"aborted", "error"}:
                    final_error = error_message or str(_value(message, "errorMessage", "") or "Error")
                    for component in list(self._toolComponentsById.values()):
                        component.updateResult(
                            {"content": [{"type": "text", "text": final_error}], "isError": True}
                        )
                    self._toolComponentsById.clear()
                else:
                    for component in list(self._toolComponentsById.values()):
                        component.setArgsComplete()
                self.streamingComponent = None
                self.streamingMessage = None
                self.footer.invalidate()
                self._request_render()
                return
            self.renderCurrentSessionState()
            return
        if event_type == "tool_execution_start":
            tool_call_id = str(_value(event, "toolCallId", ""))
            component = self._toolComponentsById.get(tool_call_id)
            if component is None:
                tool_name = str(_value(event, "toolName", ""))
                component = ToolExecutionComponent(
                    tool_name,
                    tool_call_id,
                    _value(event, "args", {}),
                    {
                        "showImages": _safe_call_bool(self.settingsManager, "getShowImages", True),
                        "imageWidthCells": _safe_call_int(self.settingsManager, "getImageWidthCells", 56),
                    },
                    _tool_definition(self.session, tool_name),
                    self.ui,
                    self.sessionManager.getCwd(),
                )
                component.setExpanded(self.toolOutputExpanded)
                self.chatContainer.addChild(component)
                self._toolComponentsById[tool_call_id] = component
            component.markExecutionStarted()
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "tool_execution_update":
            component = self._toolComponentsById.get(str(_value(event, "toolCallId", "")))
            if component is not None:
                partial = dict(_value(event, "partialResult", {}) or {})
                partial["isError"] = False
                component.updateResult(partial, True)
                self.footer.invalidate()
                self._request_render()
            return
        if event_type == "tool_execution_end":
            tool_call_id = str(_value(event, "toolCallId", ""))
            component = self._toolComponentsById.get(tool_call_id)
            if component is not None:
                result = dict(_value(event, "result", {}) or {})
                result["isError"] = bool(_value(event, "isError", False))
                component.updateResult(result)
                self._toolComponentsById.pop(tool_call_id, None)
                self.footer.invalidate()
                self._request_render()
            return
        if event_type == "agent_end":
            get_progress = _callable_attr(self.settingsManager, "getShowTerminalProgress")
            terminal = getattr(self.ui, "terminal", None)
            set_progress = _callable_attr(terminal, "setProgress")
            if get_progress is not None and bool(get_progress()) and set_progress is not None:
                set_progress(False)
            if self.loadingAnimation is not None:
                self.stopWorkingLoader()
            if self.streamingComponent is not None:
                remove_child = _callable_attr(self.chatContainer, "removeChild")
                if remove_child is not None:
                    remove_child(self.streamingComponent)
                self.streamingComponent = None
                self.streamingMessage = None
            self._toolComponentsById.clear()
            await self.checkShutdownRequested()
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "session_info_changed":
            self.updateTerminalTitle()
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "thinking_level_changed":
            self.updateEditorBorderColor()
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "auto_retry_start":
            self.retryEscapeHandler = getattr(self.defaultEditor, "onEscape", None)
            self.defaultEditor.onEscape = lambda: _callable_attr(self.session, "abortRetry") and self.session.abortRetry()
            clear_status = _callable_attr(self.statusContainer, "clear")
            if clear_status is not None:
                clear_status()
            if self.retryCountdown is not None:
                dispose = _callable_attr(self.retryCountdown, "dispose")
                if dispose is not None:
                    dispose()

            def retry_message(seconds: int) -> str:
                return (
                    f"Retrying ({int(_value(event, 'attempt', 0))}/{int(_value(event, 'maxAttempts', 0))}) in {seconds}s... "
                    f"({key_text('app.interrupt')} to cancel)"
                )

            self.retryLoader = Loader(
                self.ui,
                lambda spinner: interactive_theme.theme.fg("warning", spinner),
                lambda text: interactive_theme.theme.fg("muted", text),
                retry_message(int((_value(event, "delayMs", 0) + 999) // 1000)),
            )
            self.retryCountdown = CountdownTimer(
                int(_value(event, "delayMs", 0)),
                self.ui,
                lambda seconds: self.retryLoader.setMessage(retry_message(seconds)) if self.retryLoader is not None else None,
                lambda: setattr(self, "retryCountdown", None),
            )
            add_child = _callable_attr(self.statusContainer, "addChild")
            if add_child is not None:
                add_child(self.retryLoader)
            self.footer.invalidate()
            self._request_render()
            return
        if event_type == "auto_retry_end":
            if self.retryEscapeHandler is not None:
                self.defaultEditor.onEscape = self.retryEscapeHandler
                self.retryEscapeHandler = None
            if self.retryCountdown is not None:
                dispose = _callable_attr(self.retryCountdown, "dispose")
                if dispose is not None:
                    dispose()
                self.retryCountdown = None
            if self.retryLoader is not None:
                stop = _callable_attr(self.retryLoader, "stop")
                if stop is not None:
                    stop()
                self.retryLoader = None
                clear_status = _callable_attr(self.statusContainer, "clear")
                if clear_status is not None:
                    clear_status()
            if not bool(_value(event, "success", False)):
                self.showError(
                    f"Retry failed after {int(_value(event, 'attempt', 0))} attempts: "
                    f"{_value(event, 'finalError', None) or 'Unknown error'}"
                )
            self.footer.invalidate()
            self._request_render()
            return
        if event_type != "compaction_end":
            self.footer.invalidate()
            self._request_render()
            return

        get_progress = _callable_attr(self.settingsManager, "getShowTerminalProgress")
        terminal = getattr(self.ui, "terminal", None)
        set_progress = _callable_attr(terminal, "setProgress")
        if get_progress is not None and bool(get_progress()) and set_progress is not None:
            set_progress(False)

        if self.autoCompactionEscapeHandler is not None:
            self.defaultEditor.onEscape = self.autoCompactionEscapeHandler
            self.autoCompactionEscapeHandler = None
        if self.autoCompactionLoader is not None:
            stop = _callable_attr(self.autoCompactionLoader, "stop")
            if stop is not None:
                stop()
            self.autoCompactionLoader = None
            clear_status = _callable_attr(self.statusContainer, "clear")
            if clear_status is not None:
                clear_status()

        if bool(_value(event, "aborted")):
            if _value(event, "reason") == "manual":
                self.showError("Compaction cancelled")
            else:
                self.showStatus("Auto-compaction cancelled")
        else:
            result = _value(event, "result")
            if result is not None:
                clear_chat = _callable_attr(self.chatContainer, "clear")
                if clear_chat is not None:
                    clear_chat()
                self.rebuildChatFromMessages()
                self.addMessageToChat(
                    createCompactionSummaryMessage(
                        str(_value(result, "summary", "")),
                        int(_value(result, "tokensBefore", 0)),
                        datetime.now(UTC).isoformat(),
                    )
                )
                invalidate_footer = _callable_attr(self.footer, "invalidate")
                if invalidate_footer is not None:
                    invalidate_footer()
            elif _value(event, "errorMessage"):
                if _value(event, "reason") == "manual":
                    self.showError(str(_value(event, "errorMessage")))
                else:
                    self.chatContainer.addChild(Spacer(1))
                    self.chatContainer.addChild(
                        Text(
                            interactive_theme.theme.fg("error", str(_value(event, "errorMessage"))),
                            1,
                            0,
                        )
                    )

        flush_queue = _callable_attr(self, "flushCompactionQueue")
        if flush_queue is not None:
            await _maybe_await(flush_queue({"willRetry": bool(_value(event, "willRetry", False))}))
        self._request_render()

    async def rebindCurrentSession(self, session: Any | None = None) -> None:
        if session is not None:
            self.session = session
        elif getattr(self.runtimeHost, "session", None) is not None:
            self.session = self.runtimeHost.session

        self.sessionManager = getattr(self.session, "sessionManager", self.sessionManager)
        self.settingsManager = getattr(self.session, "settingsManager", self.settingsManager)
        self.footer.setSession(self.session)
        if hasattr(self.footerDataProvider, "setCwd"):
            self.footerDataProvider.setCwd(self.sessionManager.getCwd())
        if hasattr(self.footer, "setAutoCompactEnabled"):
            self.footer.setAutoCompactEnabled(bool(getattr(self.session, "autoCompactionEnabled", False)))

        self.hideThinkingBlock = _safe_call_bool(self.settingsManager, "getHideThinkingBlock", self.hideThinkingBlock)
        get_themes = _callable_attr(self.session.resourceLoader, "getThemes")
        themes_result = get_themes() if get_themes is not None else {}
        themes = _value(themes_result, "themes", [])
        interactive_theme.set_registered_themes(themes)
        current_theme = _callable_attr(self.settingsManager, "getTheme")
        interactive_theme.init_theme(current_theme() if current_theme is not None else None, True)
        self.resetExtensionUI()
        self.updateEditorBorderColor()
        self.updateAvailableProviderCount()
        self.setupAutocompleteProvider()
        await self.session.bindExtensions(
            {
                "uiContext": self.createExtensionUIContext(),
                "abortHandler": lambda: self.session.agent.abort() if getattr(self.session, "agent", None) else None,
                "commandContextActions": self._build_command_context_actions(),
                "shutdownHandler": self.requestShutdown,
                "onError": lambda error: self.showExtensionError(
                    str(_value(error, "extensionPath", "<extension>")),
                    str(_value(error, "error", error)),
                    _value(error, "stack"),
                ),
            }
        )
        self.setupExtensionShortcuts(self.session.extensionRunner)
        self.subscribeToSession()
        clear_chat = _callable_attr(self.chatContainer, "clear")
        if clear_chat is not None:
            clear_chat()
        self.lastStatusSpacer = None
        self.lastStatusText = None
        self.showLoadedResources({"force": False, "showDiagnosticsWhenQuiet": True})
        self.rebuildChatFromMessages()
        self._request_render()
        self.changelogMarkdown = self.getChangelogForDisplay()
        self.showStartupNoticesIfNeeded()
        self.updateTerminalTitle()

    def subscribeToSession(self) -> None:
        if self._sessionUnsubscribe is not None:
            self._sessionUnsubscribe()
            self._sessionUnsubscribe = None
        subscribe = _callable_attr(self.session, "subscribe")
        if subscribe is None:
            return

        def _listener(event: Any) -> None:
            self._schedule_task(self.handleEvent(event))

        self._sessionUnsubscribe = subscribe(_listener)

    def setupExtensionShortcuts(self, extensionRunner: Any) -> None:
        get_shortcuts = _callable_attr(extensionRunner, "get_shortcuts") or _callable_attr(
            extensionRunner, "getShortcuts"
        )
        if get_shortcuts is None:
            self.defaultEditor.onExtensionShortcut = None
            return

        get_effective_config = _callable_attr(self.keybindings, "getEffectiveConfig")
        effective_config = get_effective_config() if get_effective_config is not None else {}
        shortcuts = get_shortcuts(effective_config) or {}
        if not shortcuts:
            self.defaultEditor.onExtensionShortcut = None
            return

        create_context = _callable_attr(extensionRunner, "create_context") or _callable_attr(
            extensionRunner, "createContext"
        )

        async def _run_shortcut(handler: Any, ctx: Any) -> None:
            try:
                await _maybe_await(handler(ctx))
            except Exception as error:  # noqa: BLE001
                self.showError(f"Shortcut handler error: {error}")

        def _on_extension_shortcut(data: str) -> bool:
            for shortcut_str, shortcut in shortcuts.items():
                if not matchesKey(data, shortcut_str):
                    continue
                handler = _value(shortcut, "handler")
                if callable(handler):
                    context = create_context() if create_context is not None else None
                    self._schedule_task(_run_shortcut(handler, context))
                return True
            return False

        self.defaultEditor.onExtensionShortcut = _on_extension_shortcut
        if self.editor is not self.defaultEditor and hasattr(self.editor, "onExtensionShortcut"):
            self.editor.onExtensionShortcut = _on_extension_shortcut

    def setupKeyHandlers(self) -> None:
        def _on_escape() -> None:
            if bool(getattr(self.session, "isStreaming", False)):
                restore_queued = _callable_attr(self, "restoreQueuedMessagesToEditor")
                if restore_queued is not None:
                    restore_queued({"abort": True})
                else:
                    agent = getattr(self.session, "agent", None)
                    abort = _callable_attr(agent, "abort")
                    if abort is not None:
                        abort()
            elif bool(getattr(self.session, "isBashRunning", False)):
                abort_bash = _callable_attr(self.session, "abortBash")
                if abort_bash is not None:
                    abort_bash()
            elif _is_bash_mode(self.editor):
                self._set_editor_text("")
                self.updateEditorBorderColor()
            elif not self._get_editor_text().strip():
                action = _safe_call_str(self.settingsManager, "getDoubleEscapeAction", "none")
                if action == "none":
                    return
                now = time.monotonic() * 1000
                if now - self.lastEscapeTime < 500:
                    if action == "tree":
                        self.showTreeSelector()
                    else:
                        self.showUserMessageSelector()
                    self.lastEscapeTime = 0
                else:
                    self.lastEscapeTime = now

        self.defaultEditor.onEscape = _on_escape
        if callable(_callable_attr(self.defaultEditor, "onAction")):
            self.defaultEditor.onAction("app.clear", self.handleCtrlC)
            self.defaultEditor.onAction("app.suspend", self.handleCtrlZ)
            self.defaultEditor.onAction(
                "app.thinking.cycle",
                lambda: self._schedule_task(self._cycle_thinking_level()),
            )
            self.defaultEditor.onAction(
                "app.model.cycleForward",
                lambda: self._schedule_task(self._cycle_model("forward")),
            )
            self.defaultEditor.onAction(
                "app.model.cycleBackward",
                lambda: self._schedule_task(self._cycle_model("backward")),
            )
            self.defaultEditor.onAction("app.model.select", lambda: self.showModelSelector())
            self.defaultEditor.onAction("app.tools.expand", self.toggleToolOutputExpansion)
            self.defaultEditor.onAction("app.thinking.toggle", self.toggleThinkingBlockVisibility)
            self.defaultEditor.onAction("app.editor.external", lambda: self._schedule_task(self.openExternalEditor()))
            self.defaultEditor.onAction("app.message.followUp", lambda: self._schedule_task(self.handleFollowUp()))
            self.defaultEditor.onAction("app.message.dequeue", self.handleDequeue)
            self.defaultEditor.onAction("app.session.fork", self.showUserMessageSelector)
            self.defaultEditor.onAction("app.session.tree", self.showTreeSelector)
            self.defaultEditor.onAction("app.session.resume", lambda: self.showSessionSelector())
            self.defaultEditor.onAction("app.session.new", lambda: self._schedule_task(self.handleClearCommand()))
        debug_handler = _callable_attr(self, "handleDebugCommand")
        if debug_handler is not None:
            self.ui.onDebug = debug_handler
        self.defaultEditor.onCtrlD = self.handleCtrlD
        self.defaultEditor.onPasteImage = lambda: self._schedule_task(self.handleClipboardImagePaste())
        self.defaultEditor.onChange = lambda text: self._on_editor_change(text)

    def setupEditorSubmitHandler(self) -> None:
        self.defaultEditor.onSubmit = lambda text: self._schedule_task(self.handleSubmittedText(text))

    async def handleClipboardImagePaste(self) -> None:
        try:
            image = await read_clipboard_image()
            if image is None:
                return

            tmp_dir = Path(tempfile.gettempdir())
            extension = extension_for_image_mime_type(image.mimeType) or "png"
            file_path = tmp_dir / f"pi-clipboard-{uuid4()}.{extension}"
            file_path.write_bytes(image.bytes)

            insert_text_at_cursor = _callable_attr(self.editor, "insertTextAtCursor")
            if insert_text_at_cursor is not None:
                insert_text_at_cursor(str(file_path))
            else:
                set_text = _callable_attr(self.editor, "setText")
                get_text = _callable_attr(self.editor, "getText")
                if set_text is not None:
                    current_text = str(get_text() or "") if get_text is not None else ""
                    set_text(current_text + str(file_path))

            self.ui.requestRender()
        except Exception:
            return

    def updateAvailableProviderCount(self) -> None:
        model_registry = getattr(self.session, "modelRegistry", None)
        get_available = _callable_attr(model_registry, "getAvailable")
        if get_available is None:
            return
        providers = {
            str(_value(model, "provider", ""))
            for model in (get_available() or [])
            if _value(model, "provider")
        }
        self.footerDataProvider.setAvailableProviderCount(len(providers))

    def getMarkdownThemeWithSettings(self) -> Any:
        markdown_theme = interactive_theme.get_markdown_theme()
        indent = _safe_call_str(self.settingsManager, "getCodeBlockIndent", "  ")
        try:
            return replace(markdown_theme, codeBlockIndent=indent)
        except TypeError:
            markdown_theme.codeBlockIndent = indent
            return markdown_theme

    def updateEditorBorderColor(self) -> None:
        border = (
            interactive_theme.theme.getBashModeBorderColor()
            if _is_bash_mode(self.editor)
            else interactive_theme.theme.getThinkingBorderColor(
                str(_value(self.session.state, "thinkingLevel", "off"))
            )
        )
        if hasattr(self.editor, "borderColor"):
            self.editor.borderColor = border
        self._request_render()

    def toggleToolOutputExpansion(self) -> None:
        self.setToolsExpanded(not self.toolOutputExpanded)

    def toggleThinkingBlockVisibility(self) -> None:
        self.hideThinkingBlock = not self.hideThinkingBlock
        set_hide = _callable_attr(self.settingsManager, "setHideThinkingBlock")
        if set_hide is not None:
            set_hide(self.hideThinkingBlock)
        clear_chat = _callable_attr(self.chatContainer, "clear")
        if clear_chat is not None:
            clear_chat()
        self.rebuildChatFromMessages()

        if self.streamingComponent is not None and self.streamingMessage is not None:
            self.streamingComponent.setHideThinkingBlock(self.hideThinkingBlock)
            self.streamingComponent.updateContent(self.streamingMessage)
            add_child = _callable_attr(self.chatContainer, "addChild")
            if add_child is not None:
                add_child(self.streamingComponent)

        self.showStatus(f"Thinking blocks: {'hidden' if self.hideThinkingBlock else 'visible'}")

    async def openExternalEditor(self) -> None:
        editor_cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor_cmd:
            self.showWarning("No editor configured. Set $VISUAL or $EDITOR environment variable.")
            return

        current_text = self._get_editor_text()
        tmp_file = Path(tempfile.gettempdir()) / f"pi-editor-{int(time.time() * 1000)}.pi.md"

        try:
            tmp_file.write_text(current_text, encoding="utf-8")

            stop = _callable_attr(self.ui, "stop")
            if stop is not None:
                stop()

            parts = [part for part in editor_cmd.split(" ") if part]
            editor = parts[0]
            editor_args = parts[1:]
            sys.stdout.write(f"Launching external editor: {editor_cmd}\nPi will resume when the editor exits.\n")
            try:
                status = await asyncio.to_thread(
                    subprocess.run,
                    [editor, *editor_args, str(tmp_file)],
                    check=False,
                )
            except OSError:
                status = None

            if status is not None and status.returncode == 0:
                new_content = re.sub(r"\n$", "", tmp_file.read_text(encoding="utf-8"))
                self._set_editor_text(new_content)
        finally:
            with contextlib.suppress(OSError):
                tmp_file.unlink()

            start = _callable_attr(self.ui, "start")
            if start is not None:
                start()
            self._request_render(True)

    def clearEditor(self) -> None:
        self._set_editor_text("")
        self._request_render()

    def handleCtrlC(self) -> None:
        now = time.time() * 1000
        if now - self.lastSigintTime < 500:
            self._schedule_task(self.shutdown())
        else:
            self.clearEditor()
            self.lastSigintTime = now

    def handleCtrlD(self) -> None:
        self._schedule_task(self.shutdown())

    async def handleFollowUp(self) -> None:
        text = self._get_editor_text().strip()
        if not text:
            return

        if bool(getattr(self.session, "isCompacting", False)):
            if self.isExtensionCommand(text):
                add_history = _callable_attr(self.editor, "addToHistory")
                if add_history is not None:
                    add_history(text)
                self._set_editor_text("")
                await self.session.prompt(text)
            else:
                self.queueCompactionMessage(text, "followUp")
            return

        if bool(getattr(self.session, "isStreaming", False)):
            add_history = _callable_attr(self.editor, "addToHistory")
            if add_history is not None:
                add_history(text)
            self._set_editor_text("")
            await self.session.prompt(text, {"streamingBehavior": "followUp"})
            self.updatePendingMessagesDisplay()
            self._request_render()
            return

        on_submit = getattr(self.editor, "onSubmit", None)
        if callable(on_submit):
            self._set_editor_text("")
            await _maybe_await(on_submit(text))

    def handleDequeue(self) -> None:
        restored = self.restoreQueuedMessagesToEditor()
        if restored == 0:
            self.showStatus("No queued messages to restore")
            return
        suffix = "s" if restored > 1 else ""
        self.showStatus(f"Restored {restored} queued message{suffix} to editor")

    def showModelSelector(self, initialSearchInput: str | None = None) -> None:
        self.showSelector(
            lambda done: {
                "component": ModelSelectorComponent(
                    self.ui,
                    getattr(self.session, "model", None),
                    self.settingsManager,
                    self.session.modelRegistry,
                    [
                        ScopedModelItem(
                            model=item["model"] if isinstance(item, dict) else item.model,
                            thinkingLevel=(
                                item.get("thinkingLevel") if isinstance(item, dict) else item.thinkingLevel
                            ),
                        )
                        for item in list(getattr(self.session, "scopedModels", []) or [])
                    ],
                    lambda model: self._schedule_task(self._handle_model_select(model, done)),
                    lambda: (done(), self._request_render()),
                    initialSearchInput,
                ),
            }
        )

    def showThemeSelector(self) -> None:
        original_theme = interactive_theme.theme.name or _safe_call_str(
            self.settingsManager, "getTheme", interactive_theme.get_default_theme()
        )

        def _preview(theme_name: str) -> None:
            interactive_theme.set_theme(theme_name, True)
            self.updateEditorBorderColor()
            self._request_render()

        def _cancel() -> None:
            interactive_theme.set_theme(original_theme, True)
            self.updateEditorBorderColor()
            done()
            self._request_render()

        def _select(theme_name: str) -> None:
            self._schedule_task(self._handle_theme_select(theme_name, done))

        def done() -> None:
            self._clear_selector()

        selector = ThemeSelectorComponent(original_theme, _select, _cancel, _preview)
        self._activeSelectorHandle = self.ui.showOverlay(selector, {})

    def showSessionSelector(self) -> None:
        self.showSelector(
            lambda done: {
                "component": SessionSelectorComponent(
                    lambda onProgress=None: SessionManager.list(
                        self.sessionManager.getCwd(),
                        self.sessionManager.getSessionDir(),
                        onProgress,
                    ),
                    SessionManager.listAll,
                    lambda sessionPath: self._schedule_task(self._handle_session_select(sessionPath, done)),
                    lambda: (done(), self._request_render()),
                    self.requestShutdown,
                    lambda: self._request_render(),
                    {
                        "renameSession": lambda sessionFilePath, nextName: _rename_session_file(
                            sessionFilePath,
                            nextName,
                        ),
                        "showRenameHint": True,
                        "keybindings": self.keybindings,
                    },
                    self.sessionManager.getSessionFile(),
                ),
            }
        )

    def showTreeSelector(self, initialSelectedId: str | None = None) -> None:
        get_tree = _callable_attr(self.sessionManager, "getTree")
        tree = list(get_tree() or []) if get_tree is not None else []
        get_leaf_id = _callable_attr(self.sessionManager, "getLeafId")
        real_leaf_id = get_leaf_id() if get_leaf_id is not None else None
        initial_filter_mode = _safe_call_str(self.settingsManager, "getTreeFilterMode", "default") or "default"
        if not tree:
            self.showStatus("No entries in session")
            return

        terminal = getattr(self.ui, "terminal", None)
        terminal_height = int(_value(terminal, "rows", 24) or 24)

        self.showSelector(
            lambda done: {
                "component": TreeSelectorComponent(
                    tree,
                    real_leaf_id,
                    terminal_height,
                    lambda entry_id: self._schedule_task(
                        self._handle_tree_select(
                            str(entry_id),
                            str(real_leaf_id) if real_leaf_id is not None else None,
                            done,
                        )
                    ),
                    lambda: (done(), self._request_render()),
                    lambda entry_id, label: (
                        _callable_attr(self.sessionManager, "appendLabelChange")
                        and self.sessionManager.appendLabelChange(str(entry_id), label),
                        self._request_render(),
                    ),
                    initialSelectedId,
                    initial_filter_mode,
                ),
                "focus": True,
            }
        )

    def showSelector(self, builder: Callable[[Callable[[], None]], dict[str, Any]]) -> None:
        self._clear_selector()

        def done() -> None:
            self._clear_selector()

        built = builder(done)
        component = built["component"]
        options = built.get("options", {})
        self._activeSelectorHandle = self.ui.showOverlay(component, options)
        focus = built.get("focus")
        if focus is not None and hasattr(self._activeSelectorHandle, "focus"):
            self._activeSelectorHandle.focus()

    async def init(self) -> None:
        if self.isInitialized:
            return

        self.registerSignalHandlers()
        setKeybindings(self.keybindings)
        themes_result = {}
        get_themes = _callable_attr(getattr(self.session, "resourceLoader", None), "getThemes")
        if get_themes is not None:
            themes_result = get_themes() or {}
        interactive_theme.set_registered_themes(_value(themes_result, "themes", []))
        interactive_theme.init_theme(_safe_call_str(self.settingsManager, "getTheme"), True)
        self.updateEditorBorderColor()
        self.setupAutocompleteProvider()

        quiet_startup = _safe_call_bool(self.settingsManager, "getQuietStartup")
        self.headerContainer.clear()
        if not quiet_startup:
            logo = interactive_theme.theme.bold(
                interactive_theme.theme.fg("accent", APP_NAME)
            ) + interactive_theme.theme.fg(
                "dim",
                f" v{self.version}",
            )
            expanded_instructions = [
                key_hint("app.interrupt", "interrupt"),
                key_hint("app.clear", "clear"),
                key_hint("app.exit", "exit"),
                key_hint("app.model.select", "select model"),
                key_hint("app.session.resume", "resume"),
                raw_key_hint("/model", "model command"),
                raw_key_hint("/theme", "theme command"),
                raw_key_hint("/resume", "resume command"),
                raw_key_hint("!", "bash"),
            ]
            compact_instructions = [
                key_hint("app.interrupt", "interrupt"),
                raw_key_hint("/", "commands"),
                raw_key_hint("!", "bash"),
                key_hint("app.tools.expand", "expand"),
            ]
            self.builtInHeader = ExpandableText(
                lambda: f"{logo}\n" + interactive_theme.theme.fg("dim", " · ".join(compact_instructions)),
                lambda: f"{logo}\n" + "\n".join(expanded_instructions),
                self.getStartupExpansionState(),
                1,
                0,
            )
            self.headerContainer.addChild(Spacer(1))
            self.headerContainer.addChild(self.builtInHeader)
            self.headerContainer.addChild(Spacer(1))
        else:
            self.builtInHeader = Text("", 0, 0)
            self.headerContainer.addChild(self.builtInHeader)

        add_child = _callable_attr(self.ui, "addChild")
        if add_child is not None:
            for child in (
                self.headerContainer,
                self.chatContainer,
                self.pendingMessagesContainer,
                self.statusContainer,
                self.widgetContainerAbove,
                self.editorContainer,
                self.widgetContainerBelow,
                self.customFooter or self.footer,
            ):
                add_child(child)
        set_focus = _callable_attr(self.ui, "setFocus")
        if set_focus is not None:
            set_focus(self.editor)

        self.setupKeyHandlers()
        self.setupEditorSubmitHandler()
        self.renderWidgets()

        on_theme_change = getattr(interactive_theme, "on_theme_change", None)
        if callable(on_theme_change):
            on_theme_change(
                lambda: (
                    _callable_attr(self.ui, "invalidate") and self.ui.invalidate(),
                    self.updateEditorBorderColor(),
                    self._request_render(),
                )
            )
        on_branch_change = _callable_attr(self.footerDataProvider, "onBranchChange")
        if on_branch_change is not None:
            on_branch_change(lambda: self._request_render())

        start = _callable_attr(self.ui, "start")
        if start is not None:
            start()
        self.isInitialized = True
        await self.rebindCurrentSession()
        self.updateAvailableProviderCount()

    async def _check_for_new_version(self) -> None:
        release = await check_for_new_pi_version(self.version)
        if release is not None:
            self.showNewVersionNotification(release)

    async def _check_for_package_updates(self) -> None:
        updates = await self.checkForPackageUpdates()
        if updates:
            self.showPackageUpdateNotification(updates)

    async def _check_tmux_keyboard_setup(self) -> None:
        warning = await self.checkTmuxKeyboardSetup()
        if warning:
            self.showWarning(warning)

    async def run(self) -> int:
        await self.init()
        self._shutdownFuture = asyncio.get_running_loop().create_future()
        self._schedule_task(self._check_for_new_version())
        self._schedule_task(self._check_for_package_updates())
        self._schedule_task(self._check_tmux_keyboard_setup())

        migrated = list(self.options.migratedProviders or [])
        if migrated:
            self.showWarning(f"Migrated credentials to auth.json: {', '.join(migrated)}")
        get_model_registry_error = _callable_attr(getattr(self.session, "modelRegistry", None), "getError")
        model_registry_error = get_model_registry_error() if get_model_registry_error is not None else None
        if model_registry_error:
            self.showError(f"models.json error: {model_registry_error}")
        if self.options.modelFallbackMessage:
            self.showWarning(self.options.modelFallbackMessage)

        await self.maybeWarnAboutAnthropicSubscriptionAuth()

        if self.options.initialMessage:
            await self.session.prompt(
                self.options.initialMessage,
                {"images": list(self.options.initialImages or [])},
            )
            self.renderCurrentSessionState()
        for message in list(self.options.initialMessages or []):
            await self.session.prompt(message)
            self.renderCurrentSessionState()

        return await self._shutdownFuture

    async def shutdown(self) -> int:
        self.requestShutdown()
        if self._shutdownFuture is None:
            return 0
        return await asyncio.shield(self._shutdownFuture)

    def requestShutdown(self) -> None:
        if self._shutdownFuture is not None and self._shutdownFuture.done():
            return
        stop = _callable_attr(self.ui, "stop")
        if stop is not None:
            stop()
        if self._shutdownFuture is not None and not self._shutdownFuture.done():
            self._shutdownFuture.set_result(0)

    def dispose(self) -> None:
        if self._sessionUnsubscribe is not None:
            self._sessionUnsubscribe()
            self._sessionUnsubscribe = None
        stop_theme_watcher = getattr(interactive_theme, "stop_theme_watcher", None)
        if callable(stop_theme_watcher):
            stop_theme_watcher()
        dispose_footer = _callable_attr(self.footerDataProvider, "dispose")
        if dispose_footer is not None:
            dispose_footer()
        self._clear_selector()

    def _on_editor_change(self, text: str) -> None:
        self._handleClearCount = 0
        if hasattr(self.editor, "borderColor"):
            self.editor.borderColor = (
                interactive_theme.theme.getBashModeBorderColor()
                if text.lstrip().startswith("!")
                else interactive_theme.theme.getThinkingBorderColor(
                    str(_value(self.session.state, "thinkingLevel", "off"))
                )
            )
        self._request_render()

    def _schedule_task(self, awaitable: Awaitable[Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(awaitable)
            return
        task = loop.create_task(awaitable)
        self._backgroundTasks.add(task)
        task.add_done_callback(self._finish_background_task)

    def _finish_background_task(self, task: asyncio.Task[Any]) -> None:
        self._backgroundTasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))

    def _set_editor_text(self, text: str) -> None:
        set_text = _callable_attr(self.editor, "setText")
        if set_text is not None:
            set_text(text)

    def _get_editor_text(self) -> str:
        expanded = _callable_attr(self.editor, "getExpandedText")
        if expanded is not None:
            return str(expanded())
        get_text = _callable_attr(self.editor, "getText")
        if get_text is not None:
            return str(get_text())
        return ""

    def _clear_selector(self) -> None:
        handle = self._activeSelectorHandle
        self._activeSelectorHandle = None
        if handle is not None and hasattr(handle, "hide"):
            handle.hide()

    def getStartupExpansionState(self) -> bool:
        return bool(self.options.verbose or self.toolOutputExpanded)

    async def _cycle_model(self, direction: str) -> None:
        try:
            result = await self.session.cycleModel(direction)
            if result is None:
                scoped_models = list(getattr(self.session, "scopedModels", []) or [])
                self.showStatus("Only one model in scope" if scoped_models else "Only one model available")
                return
            self.footer.invalidate()
            self.updateEditorBorderColor()
            thinking_str = ""
            if bool(_value(result.model, "reasoning", False)) and _value(result, "thinkingLevel") != "off":
                thinking_str = f" (thinking: {_value(result, 'thinkingLevel')})"
            self.showStatus(f"Switched to {_value(result.model, 'name', None) or result.model.id}{thinking_str}")
            self.checkDaxnutsEasterEgg(result.model)
            await self.maybeWarnAboutAnthropicSubscriptionAuth(result.model)
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))

    async def _cycle_thinking_level(self) -> None:
        level = self.session.cycleThinkingLevel()
        if level is None:
            self.showStatus("Current model does not support thinking")
            return
        self.footer.invalidate()
        self.updateEditorBorderColor()
        self.showStatus(f"Thinking level: {level}")

    async def _handle_model_select(self, model: Any, done: Callable[[], None]) -> None:
        try:
            await self.session.setModel(model)
            self.footer.invalidate()
            self.updateAvailableProviderCount()
            self.updateEditorBorderColor()
            done()
            self._request_render()
            self.showStatus(f"Model: {_value(model, 'id', model)}")
            self.checkDaxnutsEasterEgg(model)
            await self.maybeWarnAboutAnthropicSubscriptionAuth(model)
        except Exception as error:  # noqa: BLE001
            done()
            self.showError(str(error))

    async def _handle_theme_select(self, theme_name: str, done: Callable[[], None]) -> None:
        result = interactive_theme.set_theme(theme_name, True)
        if not result.get("success"):
            self.showError(str(result.get("error", f"Theme not found: {theme_name}")))
            return
        set_theme = _callable_attr(self.settingsManager, "setTheme")
        if set_theme is not None:
            set_theme(theme_name)
        self.updateEditorBorderColor()
        done()
        self._request_render()
        self.showStatus(f"Theme: {theme_name}")

    async def _handle_session_select(self, sessionPath: str, done: Callable[[], None]) -> None:
        done()
        await self.handleResumeSession(sessionPath)

    async def handleModelCommand(self, searchTerm: str | None = None) -> None:
        if not searchTerm:
            self.showModelSelector()
            return

        model = findExactModelReferenceMatch(searchTerm, self.getModelCandidates())
        if model is None:
            self.showModelSelector(searchTerm)
            return

        try:
            await self.session.setModel(model)
            self.footer.invalidate()
            self.updateAvailableProviderCount()
            self.updateEditorBorderColor()
            self._request_render()
            self.showStatus(f"Model: {_value(model, 'id', model)}")
            self.checkDaxnutsEasterEgg(model)
            await self.maybeWarnAboutAnthropicSubscriptionAuth(model)
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))

    def getModelCandidates(self) -> list[Any]:
        scoped_models = list(getattr(self.session, "scopedModels", []) or [])
        if scoped_models:
            return [_value(item, "model", item) for item in scoped_models]

        model_registry = getattr(self.session, "modelRegistry", None)
        refresh = _callable_attr(model_registry, "refresh")
        if refresh is not None:
            refresh()
        get_available = _callable_attr(model_registry, "getAvailable")
        return list(get_available() or []) if get_available is not None else []

    def showSettingsSelector(self) -> None:
        available_themes = [
            str(_value(item, "name", item))
            for item in interactive_theme.get_available_themes_with_paths()
        ]
        get_available_thinking_levels = _callable_attr(self.session, "getAvailableThinkingLevels")
        get_warnings = _callable_attr(self.settingsManager, "getWarnings")

        self.showSelector(
            lambda done: {
                "component": SettingsSelectorComponent(
                    SettingsConfig(
                        autoCompact=bool(getattr(self.session, "autoCompactionEnabled", False)),
                        showImages=_safe_call_bool(self.settingsManager, "getShowImages", True),
                        imageWidthCells=_safe_call_int(self.settingsManager, "getImageWidthCells", 40),
                        autoResizeImages=_safe_call_bool(self.settingsManager, "getImageAutoResize", True),
                        blockImages=_safe_call_bool(self.settingsManager, "getBlockImages", False),
                        enableSkillCommands=_safe_call_bool(self.settingsManager, "getEnableSkillCommands", True),
                        steeringMode=str(getattr(self.session, "steeringMode", "one-at-a-time")),
                        followUpMode=str(getattr(self.session, "followUpMode", "one-at-a-time")),
                        transport=str(_safe_call_str(self.settingsManager, "getTransport", "sse")),
                        httpIdleTimeoutMs=_safe_call_int(self.settingsManager, "getHttpIdleTimeoutMs", 300_000),
                        thinkingLevel=str(_value(self.session.state, "thinkingLevel", "off")),
                        availableThinkingLevels=list(get_available_thinking_levels() or [])
                        if get_available_thinking_levels is not None
                        else [],
                        currentTheme=_safe_call_str(
                            self.settingsManager,
                            "getTheme",
                            interactive_theme.theme.name or interactive_theme.get_default_theme(),
                        )
                        or interactive_theme.get_default_theme(),
                        availableThemes=available_themes,
                        hideThinkingBlock=self.hideThinkingBlock,
                        collapseChangelog=_safe_call_bool(self.settingsManager, "getCollapseChangelog", True),
                        enableInstallTelemetry=_safe_call_bool(
                            self.settingsManager,
                            "getEnableInstallTelemetry",
                            True,
                        ),
                        doubleEscapeAction=_safe_call_str(self.settingsManager, "getDoubleEscapeAction", "tree")
                        or "tree",
                        treeFilterMode=_safe_call_str(self.settingsManager, "getTreeFilterMode", "default")
                        or "default",
                        showHardwareCursor=_safe_call_bool(self.settingsManager, "getShowHardwareCursor", False),
                        editorPaddingX=_safe_call_int(self.settingsManager, "getEditorPaddingX", 0),
                        autocompleteMaxVisible=_safe_call_int(
                            self.settingsManager,
                            "getAutocompleteMaxVisible",
                            5,
                        ),
                        quietStartup=_safe_call_bool(self.settingsManager, "getQuietStartup", False),
                        clearOnShrink=_safe_call_bool(self.settingsManager, "getClearOnShrink", False),
                        showTerminalProgress=_safe_call_bool(self.settingsManager, "getShowTerminalProgress", False),
                        warnings=dict(get_warnings() or {}) if get_warnings is not None else {},
                    ),
                    SettingsCallbacks(
                        onAutoCompactChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setCompactionEnabled")
                            and self.settingsManager.setCompactionEnabled(enabled),
                            _callable_attr(self.footer, "setAutoCompactEnabled")
                            and self.footer.setAutoCompactEnabled(enabled),
                        ),
                        onShowImagesChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setShowImages")
                            and self.settingsManager.setShowImages(enabled),
                            [
                                _callable_attr(child, "setShowImages") and child.setShowImages(enabled)
                                for child in getattr(self.chatContainer, "children", [])
                            ],
                        ),
                        onImageWidthCellsChange=lambda width: (
                            _callable_attr(self.settingsManager, "setImageWidthCells")
                            and self.settingsManager.setImageWidthCells(width),
                            [
                                _callable_attr(child, "setImageWidthCells") and child.setImageWidthCells(width)
                                for child in getattr(self.chatContainer, "children", [])
                            ],
                        ),
                        onAutoResizeImagesChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setImageAutoResize")
                            and self.settingsManager.setImageAutoResize(enabled)
                        ),
                        onBlockImagesChange=lambda blocked: (
                            _callable_attr(self.settingsManager, "setBlockImages")
                            and self.settingsManager.setBlockImages(blocked)
                        ),
                        onEnableSkillCommandsChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setEnableSkillCommands")
                            and self.settingsManager.setEnableSkillCommands(enabled),
                            self.setupAutocompleteProvider(),
                        ),
                        onSteeringModeChange=lambda mode: (
                            _callable_attr(self.session, "setSteeringMode") and self.session.setSteeringMode(mode)
                        ),
                        onFollowUpModeChange=lambda mode: (
                            _callable_attr(self.session, "setFollowUpMode") and self.session.setFollowUpMode(mode)
                        ),
                        onTransportChange=lambda transport: (
                            _callable_attr(self.settingsManager, "setTransport")
                            and self.settingsManager.setTransport(transport),
                            hasattr(getattr(self.session, "agent", None), "transport")
                            and setattr(self.session.agent, "transport", transport),
                        ),
                        onHttpIdleTimeoutMsChange=lambda timeout_ms: (
                            _callable_attr(self.settingsManager, "setHttpIdleTimeoutMs")
                            and self.settingsManager.setHttpIdleTimeoutMs(timeout_ms),
                            self.showStatus(
                                f"HTTP idle timeout: {timeout_ms / 1000:g} sec"
                                if timeout_ms
                                else "HTTP idle timeout: disabled"
                            ),
                        ),
                        onThinkingLevelChange=lambda level: (
                            _callable_attr(self.session, "setThinkingLevel") and self.session.setThinkingLevel(level),
                            self.footer.invalidate(),
                            self.updateEditorBorderColor(),
                        ),
                        onThemeChange=lambda theme_name: self._schedule_task(
                            self._handle_theme_select(theme_name, lambda: None)
                        ),
                        onThemePreview=lambda theme_name: (
                            interactive_theme.set_theme(theme_name, True),
                            self.updateEditorBorderColor(),
                            self._request_render(),
                        ),
                        onHideThinkingBlockChange=lambda hidden: (
                            setattr(self, "hideThinkingBlock", hidden),
                            _callable_attr(self.settingsManager, "setHideThinkingBlock")
                            and self.settingsManager.setHideThinkingBlock(hidden),
                            [
                                _callable_attr(child, "setHideThinkingBlock") and child.setHideThinkingBlock(hidden)
                                for child in getattr(self.chatContainer, "children", [])
                            ],
                            self.renderCurrentSessionState(),
                        ),
                        onCollapseChangelogChange=lambda collapsed: (
                            _callable_attr(self.settingsManager, "setCollapseChangelog")
                            and self.settingsManager.setCollapseChangelog(collapsed)
                        ),
                        onEnableInstallTelemetryChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setEnableInstallTelemetry")
                            and self.settingsManager.setEnableInstallTelemetry(enabled)
                        ),
                        onDoubleEscapeActionChange=lambda action: (
                            _callable_attr(self.settingsManager, "setDoubleEscapeAction")
                            and self.settingsManager.setDoubleEscapeAction(action)
                        ),
                        onTreeFilterModeChange=lambda mode: (
                            _callable_attr(self.settingsManager, "setTreeFilterMode")
                            and self.settingsManager.setTreeFilterMode(mode)
                        ),
                        onShowHardwareCursorChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setShowHardwareCursor")
                            and self.settingsManager.setShowHardwareCursor(enabled),
                            _callable_attr(self.ui, "setShowHardwareCursor")
                            and self.ui.setShowHardwareCursor(enabled),
                        ),
                        onEditorPaddingXChange=lambda padding: (
                            _callable_attr(self.settingsManager, "setEditorPaddingX")
                            and self.settingsManager.setEditorPaddingX(padding),
                            _callable_attr(self.defaultEditor, "setPaddingX")
                            and self.defaultEditor.setPaddingX(padding),
                            self.editor is not self.defaultEditor
                            and _callable_attr(self.editor, "setPaddingX")
                            and self.editor.setPaddingX(padding),
                        ),
                        onAutocompleteMaxVisibleChange=lambda max_visible: (
                            _callable_attr(self.settingsManager, "setAutocompleteMaxVisible")
                            and self.settingsManager.setAutocompleteMaxVisible(max_visible),
                            _callable_attr(self.defaultEditor, "setAutocompleteMaxVisible")
                            and self.defaultEditor.setAutocompleteMaxVisible(max_visible),
                            self.editor is not self.defaultEditor
                            and _callable_attr(self.editor, "setAutocompleteMaxVisible")
                            and self.editor.setAutocompleteMaxVisible(max_visible),
                        ),
                        onQuietStartupChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setQuietStartup")
                            and self.settingsManager.setQuietStartup(enabled)
                        ),
                        onClearOnShrinkChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setClearOnShrink")
                            and self.settingsManager.setClearOnShrink(enabled),
                            _callable_attr(self.ui, "setClearOnShrink") and self.ui.setClearOnShrink(enabled),
                        ),
                        onShowTerminalProgressChange=lambda enabled: (
                            _callable_attr(self.settingsManager, "setShowTerminalProgress")
                            and self.settingsManager.setShowTerminalProgress(enabled)
                        ),
                        onWarningsChange=lambda warnings: (
                            _callable_attr(self.settingsManager, "setWarnings")
                            and self.settingsManager.setWarnings(warnings)
                        ),
                        onCancel=lambda: (done(), self._request_render()),
                    ),
                ),
            }
        )

    async def showModelsSelector(self) -> None:
        all_models = self.getModelCandidates()
        if not all_models:
            self.showStatus("No models available")
            return

        session_scoped_models = list(getattr(self.session, "scopedModels", []) or [])
        if session_scoped_models:
            current_enabled_ids: list[str] | None = [
                f"{_value(item, 'model').provider}/{_value(item, 'model').id}"
                for item in session_scoped_models
                if _value(item, "model") is not None
            ]
        else:
            patterns = _callable_attr(self.settingsManager, "getEnabledModels")
            enabled_patterns = list(patterns() or []) if patterns is not None else []
            if enabled_patterns:
                resolved = await resolveModelScope(enabled_patterns, getattr(self.session, "modelRegistry", None))
                current_enabled_ids = [f"{item.model.provider}/{item.model.id}" for item in resolved]
            else:
                current_enabled_ids = None

        async def _update_session_models(enabled_ids: list[str] | None) -> None:
            if enabled_ids and len(enabled_ids) < len(all_models):
                resolved = await resolveModelScope(enabled_ids, getattr(self.session, "modelRegistry", None))
                self.session.setScopedModels(
                    [{"model": item.model, "thinkingLevel": item.thinkingLevel} for item in resolved]
                )
            else:
                self.session.setScopedModels([])
            self.updateAvailableProviderCount()
            self._request_render()

        self.showSelector(
            lambda done: {
                "component": ScopedModelsSelectorComponent(
                    ModelsConfig(
                        allModels=list(all_models),
                        enabledModelIds=current_enabled_ids,
                    ),
                    ModelsCallbacks(
                        onChange=lambda enabled_ids: self._schedule_task(_update_session_models(enabled_ids)),
                        onPersist=lambda enabled_ids: (
                            _callable_attr(self.settingsManager, "setEnabledModels")
                            and self.settingsManager.setEnabledModels(
                                None
                                if enabled_ids is None or len(enabled_ids) == len(all_models)
                                else list(enabled_ids)
                            ),
                            self.showStatus("Model selection saved to settings"),
                        ),
                        onCancel=lambda: (done(), self._request_render()),
                    ),
                ),
            }
        )

    def showUserMessageSelector(self) -> None:
        user_messages = list(_callable_attr(self.session, "getUserMessagesForForking")() or [])
        if not user_messages:
            self.showStatus("No messages to fork from")
            return

        initial_selected_id = _value(user_messages[-1], "entryId")
        self.showSelector(
            lambda done: {
                "component": UserMessageSelectorComponent(
                    [
                        UserMessageItem(id=str(_value(message, "entryId")), text=str(_value(message, "text", "")))
                        for message in user_messages
                    ],
                    lambda entry_id: self._schedule_task(self._handle_user_message_fork(entry_id, done)),
                    lambda: (done(), self._request_render()),
                    str(initial_selected_id) if initial_selected_id is not None else None,
                ),
            }
        )

    async def _handle_user_message_fork(self, entryId: str, done: Callable[[], None]) -> None:
        try:
            result = await self.runtimeHost.fork(entryId)
            if result.get("cancelled"):
                done()
                self._request_render()
                return
            self.renderCurrentSessionState()
            self._set_editor_text(str(result.get("selectedText") or ""))
            done()
            self.showStatus("Forked to new session")
        except Exception as error:  # noqa: BLE001
            done()
            self.showError(str(error))

    async def _handle_tree_select(
        self,
        entryId: str,
        realLeafId: str | None,
        done: Callable[[], None],
    ) -> None:
        if entryId == realLeafId:
            done()
            self.showStatus("Already at this point")
            return

        done()
        navigate_tree = _callable_attr(self.session, "navigateTree")
        abort_branch_summary = _callable_attr(self.session, "abortBranchSummary")
        if navigate_tree is None:
            self.showError("Session tree navigation is unavailable")
            return

        wants_summary = False
        custom_instructions: str | None = None
        if not _safe_call_bool(self.settingsManager, "getBranchSummarySkipPrompt", False):
            while True:
                summary_choice = await self.showExtensionSelector(
                    "Summarize branch?",
                    ["No summary", "Summarize", "Summarize with custom prompt"],
                )
                if summary_choice is None:
                    self.showTreeSelector(entryId)
                    return

                wants_summary = summary_choice != "No summary"
                if summary_choice == "Summarize with custom prompt":
                    custom_instructions = await self.showExtensionEditor("Custom summarization instructions")
                    if custom_instructions is None:
                        continue
                break

        original_escape = getattr(self.defaultEditor, "onEscape", None)
        summary_loader = None
        if wants_summary:
            self.defaultEditor.onEscape = (
                (lambda: abort_branch_summary()) if abort_branch_summary is not None else original_escape
            )
            clear_status = _callable_attr(self.statusContainer, "clear")
            if clear_status is not None:
                clear_status()
            summary_loader = BorderedLoader(
                self.ui,
                interactive_theme.theme,
                f"Summarizing branch... ({key_text('app.interrupt')} to cancel)",
                {"cancellable": False},
            )
            add_child = _callable_attr(self.statusContainer, "addChild")
            if add_child is not None:
                add_child(summary_loader)
            self._request_render()

        try:
            navigate_options = {
                "summarize": wants_summary,
                "customInstructions": custom_instructions,
            }
            result = await navigate_tree(entryId, navigate_options)
            if _value(result, "aborted", False):
                self.showStatus("Branch summarization cancelled")
                self.showTreeSelector(entryId)
                return
            if _value(result, "cancelled", False):
                self.showStatus("Navigation cancelled")
                return
            self.renderCurrentSessionState()
            get_text = _callable_attr(self.editor, "getText")
            current_text = str(get_text() or "") if get_text is not None else ""
            editor_text = _value(result, "editorText")
            if editor_text and not current_text.strip():
                self._set_editor_text(str(editor_text))
            self.showStatus("Navigated to selected point")
            flush_queue = _callable_attr(self, "flushCompactionQueue")
            if flush_queue is not None:
                await _maybe_await(flush_queue({"willRetry": False}))
        except Exception as error:  # noqa: BLE001
            self.showError(str(error))
        finally:
            self.defaultEditor.onEscape = original_escape
            if summary_loader is not None:
                dispose = _callable_attr(summary_loader, "dispose")
                if dispose is not None:
                    dispose()
                clear_status = _callable_attr(self.statusContainer, "clear")
                if clear_status is not None:
                    clear_status()
                self._request_render()

    def _build_command_context_actions(self) -> dict[str, Any]:
        return {
            "waitForIdle": lambda: self.session.agent.waitForIdle() if getattr(self.session, "agent", None) else None,
            "newSession": lambda new_session_options=None: self.runtimeHost.newSession(new_session_options),
            "fork": lambda entry_id, fork_options=None: self.runtimeHost.fork(entry_id, fork_options),
            "navigateTree": lambda target_id, tree_options=None: self._navigate_tree_from_command_context(
                target_id,
                tree_options,
            ),
            "switchSession": lambda session_path, switch_options=None: self.runtimeHost.switchSession(
                session_path,
                switch_options,
            ),
            "reload": self.handleReloadCommand,
        }

    async def _navigate_tree_from_command_context(
        self,
        targetId: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        navigate_tree = _callable_attr(self.session, "navigateTree")
        if navigate_tree is None:
            return {"cancelled": True}

        result = await navigate_tree(targetId, options or None)
        if _value(result, "cancelled", False):
            return dict(result) if isinstance(result, dict) else {"cancelled": True}

        self.renderCurrentSessionState()
        get_text = _callable_attr(self.editor, "getText")
        current_text = str(get_text() or "") if get_text is not None else ""
        editor_text = _value(result, "editorText")
        if editor_text and not current_text.strip():
            self._set_editor_text(str(editor_text))
        self.showStatus("Navigated to selected point")
        flush_queue = _callable_attr(self, "flushCompactionQueue")
        if flush_queue is not None:
            await _maybe_await(flush_queue({"willRetry": False}))
        return dict(result) if isinstance(result, dict) else {"cancelled": False}


def _safe_call_bool(obj: Any, name: str, default: bool = False) -> bool:
    getter = _callable_attr(obj, name)
    if getter is None:
        return default
    try:
        return bool(getter())
    except Exception:
        return default


def _safe_call_int(obj: Any, name: str, default: int = 0) -> int:
    getter = _callable_attr(obj, name)
    if getter is None:
        return default
    try:
        return int(getter())
    except Exception:
        return default


def _safe_call_str(obj: Any, name: str, default: str | None = None) -> str | None:
    getter = _callable_attr(obj, name)
    if getter is None:
        return default
    try:
        value = getter()
    except Exception:
        return default
    return str(value) if value is not None else default


def _is_bash_mode(editor: Any) -> bool:
    get_text = _callable_attr(editor, "getText")
    text = get_text() if get_text is not None else ""
    return str(text).lstrip().startswith("!")


def _extract_user_text(message: Any) -> str:
    content = _value(message, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if _value(block, "type") == "text":
            parts.append(str(_value(block, "text", "")))
    return "".join(parts)


def _extract_custom_text(message: Any) -> str:
    content = _value(message, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if _value(block, "type") == "text":
            parts.append(str(_value(block, "text", "")))
    return "\n".join(parts)


def _model_argument_completions(session: Any, prefix: str) -> list[dict[str, str]] | None:
    model_registry = getattr(session, "modelRegistry", None)
    get_available = _callable_attr(model_registry, "getAvailable")
    if get_available is None:
        return None
    models = get_available() or []
    if not models:
        return None
    items = [
        {"id": str(_value(model, "id", "")), "provider": str(_value(model, "provider", "")), "model": model}
        for model in models
    ]
    filtered = [
        item
        for item in items
        if prefix.lower() in f"{item['id']} {item['provider']} {item['provider']}/{item['id']}".lower()
    ]
    if not filtered:
        return None
    return [
        {
            "value": f"{item['provider']}/{item['id']}",
            "label": item["id"],
            "description": item["provider"],
        }
        for item in filtered
    ]


def _rename_session_file(session_file_path: str, next_name: str | None) -> None:
    next_value = (next_name or "").strip()
    if not next_value:
        return
    manager = SessionManager.open(session_file_path)
    manager.appendSessionInfo(next_value)


__all__ = [
    "InteractiveMode",
    "InteractiveModeOptions",
    "isApiKeyLoginProvider",
]
