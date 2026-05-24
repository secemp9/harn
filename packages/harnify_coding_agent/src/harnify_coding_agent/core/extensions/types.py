"""Extension type surface for coding-agent resource and tool loading."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeVar

from harnify_agent.harness.messages import CustomMessage
from harnify_agent.types import AgentMessage, AgentToolResult, AgentToolUpdateCallback, ThinkingLevel, ToolExecutionMode
from harnify_ai.types import (
    Api,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Context,
    ImageContent,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
)
from harnify_ai.utils.oauth.types import OAuthCredentials, OAuthLoginCallbacks
from harnify_tui import AutocompleteItem, AutocompleteProvider, Component, EditorComponent, EditorTheme, KeyId, TUI

from harnify_coding_agent.core.compaction import CompactionResult as SessionCompactionResult
from harnify_coding_agent.core.event_bus import EventBus
from harnify_coding_agent.core.exec import ExecOptions, ExecResult
from harnify_coding_agent.core.footer_data_provider import ReadonlyFooterDataProvider
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.model_registry import ModelRegistry
from harnify_coding_agent.core.session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    ReadonlySessionManager,
    SessionEntry,
    SessionManager,
)
from harnify_coding_agent.core.slash_commands import SlashCommandInfo
from harnify_coding_agent.core.source_info import SourceInfo
from harnify_coding_agent.core.system_prompt import BuildSystemPromptOptions
from harnify_coding_agent.modes.interactive.theme.theme import Theme

if TYPE_CHECKING:
    from harnify_coding_agent.core.bash_executor import BashResult
    from harnify_coding_agent.core.tools.bash import BashOperations, BashToolDetails, BashToolInput
    from harnify_coding_agent.core.tools.edit import EditToolDetails, EditToolInput
    from harnify_coding_agent.core.tools.find import FindToolDetails, FindToolInput
    from harnify_coding_agent.core.tools.grep import GrepToolDetails, GrepToolInput
    from harnify_coding_agent.core.tools.ls import LsToolDetails, LsToolInput
    from harnify_coding_agent.core.tools.read import ReadToolDetails, ReadToolInput
    from harnify_coding_agent.core.tools.write import WriteToolInput

TArgs = TypeVar("TArgs")
TDetails = TypeVar("TDetails")
TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")

type AppKeybinding = str
type WidgetPlacement = Literal["aboveEditor", "belowEditor"]
type ExtensionUIDialogOptions = dict[str, Any]
type ExtensionWidgetOptions = dict[str, WidgetPlacement]
type TerminalInputResult = dict[str, bool | str]
type TerminalInputHandler = Callable[[str], TerminalInputResult | None]
type WorkingIndicatorOptions = dict[str, Any]
type AutocompleteProviderFactory = Callable[[AutocompleteProvider], AutocompleteProvider]
type EditorFactory = Callable[[TUI, EditorTheme, KeybindingsManager], EditorComponent]
type ExtensionErrorListener = Callable[["ExtensionError"], None]
type NewSessionHandler = Callable[[dict[str, Any] | None], Awaitable[dict[str, bool]]]
type ForkHandler = Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
type NavigateTreeHandler = Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
type SwitchSessionHandler = Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
type ReloadHandler = Callable[[], Awaitable[None]]
type ShutdownHandler = Callable[[], None]
type ModelSelectSource = Literal["set", "cycle", "restore"]
type InputSource = Literal["interactive", "rpc", "extension"]
type MessageRenderer[TDetails] = Callable[[CustomMessage[TDetails], "MessageRenderOptions", Theme], Component | None]
type ExtensionHandler[TEvent, TResult] = Callable[
    [TEvent, "ExtensionContext"],
    Awaitable[TResult | None] | TResult | None,
]
type SendMessageHandler = Callable[
    [Any, dict[str, Any] | None],
    None,
]
type SendUserMessageHandler = Callable[
    [str | list[TextContent | ImageContent], dict[str, Any] | None],
    None,
]
type AppendEntryHandler = Callable[[str, Any], None]
type SetSessionNameHandler = Callable[[str], None]
type GetSessionNameHandler = Callable[[], str | None]
type SetLabelHandler = Callable[[str, str | None], None]
type GetActiveToolsHandler = Callable[[], list[str]]
type GetAllToolsHandler = Callable[[], list["ToolInfo"]]
type SetActiveToolsHandler = Callable[[list[str]], None]
type RefreshToolsHandler = Callable[[], None]
type GetCommandsHandler = Callable[[], list[SlashCommandInfo]]
type SetModelHandler = Callable[[Model[Any]], Awaitable[bool]]
type GetThinkingLevelHandler = Callable[[], ThinkingLevel]
type SetThinkingLevelHandler = Callable[[ThinkingLevel], None]
type RegisterProviderHandler = Callable[[str, "ProviderConfig", str | None], None]
type UnregisterProviderHandler = Callable[[str, str | None], None]
type ToolCallRenderer[TArgs] = Callable[[TArgs, Theme, "ToolRenderContext"], Component]
type ToolResultRenderer = Callable[
    [AgentToolResult | Mapping[str, Any], "ToolRenderResultOptions", Theme, "ToolRenderContext"],
    Component,
]
type ToolRenderShell = Literal["default", "self"]


@dataclass(slots=True)
class ToolDefinition[TArgs, TDetails]:
    name: str
    label: str
    description: str
    parameters: Any
    execute: Callable[
        [str, TArgs, Any | None, AgentToolUpdateCallback | None, ExtensionContext],
        Awaitable[AgentToolResult],
    ]
    prepareArguments: Callable[[Any], Any] | None = None
    executionMode: ToolExecutionMode | None = None
    promptSnippet: str | None = None
    promptGuidelines: list[str] = field(default_factory=list)
    renderCall: ToolCallRenderer[TArgs] | None = None
    renderResult: ToolResultRenderer | None = None
    renderShell: ToolRenderShell | None = None


@dataclass(slots=True)
class ToolInfo:
    name: str
    description: str
    parameters: Any
    sourceInfo: SourceInfo


@dataclass(slots=True)
class RegisteredTool:
    definition: ToolDefinition[Any, Any]
    sourceInfo: SourceInfo


@dataclass(slots=True)
class RegisteredCommand:
    name: str
    sourceInfo: SourceInfo
    description: str | None = None
    getArgumentCompletions: Callable[[str], list[Any] | None | Awaitable[list[Any] | None]] | None = None
    handler: Callable[[str, ExtensionCommandContext], Awaitable[None]] | None = None


@dataclass(slots=True)
class ResolvedCommand(RegisteredCommand):
    invocationName: str = ""


@dataclass(slots=True)
class ExtensionFlag:
    name: str
    extensionPath: str
    type: Literal["boolean", "string"]
    description: str | None = None
    default: bool | str | None = None


@dataclass(slots=True)
class ExtensionShortcut:
    shortcut: KeyId
    extensionPath: str
    handler: Callable[[ExtensionContext], Awaitable[None] | None]
    description: str | None = None


class ProviderModelConfig(TypedDict, total=False):
    id: str
    name: str
    api: Api
    baseUrl: str
    reasoning: bool
    thinkingLevelMap: dict[str, str | None]
    input: list[Literal["text", "image"]]
    cost: dict[str, float]
    contextWindow: int
    maxTokens: int
    headers: dict[str, str]
    compat: dict[str, Any]


class OAuthProviderConfig(TypedDict, total=False):
    name: str
    login: Callable[[OAuthLoginCallbacks], Awaitable[OAuthCredentials]]
    refreshToken: Callable[[OAuthCredentials], Awaitable[OAuthCredentials]]
    getApiKey: Callable[[OAuthCredentials], str]
    modifyModels: Callable[[list[Model[Any]], OAuthCredentials], list[Model[Any]]]


class ProviderConfig(TypedDict, total=False):
    name: str
    baseUrl: str
    apiKey: str
    api: Api
    streamSimple: Callable[[Model[Any], Context, SimpleStreamOptions | None], AssistantMessageEventStream]
    headers: dict[str, str]
    authHeader: bool
    models: list[ProviderModelConfig]
    oauth: OAuthProviderConfig


@dataclass(slots=True)
class PendingProviderRegistration:
    name: str
    config: ProviderConfig
    extensionPath: str


class ContextUsage(TypedDict):
    tokens: int | None
    contextWindow: int
    percent: float | None


class CompactOptions(TypedDict, total=False):
    customInstructions: str
    onComplete: Callable[[SessionCompactionResult], None]
    onError: Callable[[Exception], None]


class MessageRenderOptions(TypedDict):
    expanded: bool


class ToolRenderResultOptions(TypedDict):
    expanded: bool
    isPartial: bool


class ToolRenderContext(Protocol):
    args: Any
    toolCallId: str
    invalidate: Callable[[], None]
    lastComponent: Any | None
    state: Any
    cwd: str
    executionStarted: bool
    argsComplete: bool
    isPartial: bool
    expanded: bool
    showImages: bool
    isError: bool


class ResourcesDiscoverEvent(TypedDict):
    type: Literal["resources_discover"]
    cwd: str
    reason: Literal["startup", "reload"]


class ResourcesDiscoverResult(TypedDict, total=False):
    skillPaths: list[str]
    promptPaths: list[str]
    themePaths: list[str]


class SessionStartEvent(TypedDict, total=False):
    type: Literal["session_start"]
    reason: Literal["startup", "reload", "new", "resume", "fork"]
    previousSessionFile: str


class SessionBeforeSwitchEvent(TypedDict, total=False):
    type: Literal["session_before_switch"]
    reason: Literal["new", "resume"]
    targetSessionFile: str


class SessionBeforeForkEvent(TypedDict):
    type: Literal["session_before_fork"]
    entryId: str
    position: Literal["before", "at"]


class SessionBeforeCompactEvent(TypedDict, total=False):
    type: Literal["session_before_compact"]
    preparation: Any
    branchEntries: list[SessionEntry]
    customInstructions: str | None
    signal: Any


class SessionCompactEvent(TypedDict):
    type: Literal["session_compact"]
    compactionEntry: dict[str, Any]
    fromExtension: bool


class SessionShutdownEvent(TypedDict, total=False):
    type: Literal["session_shutdown"]
    reason: Literal["quit", "reload", "new", "resume", "fork"]
    targetSessionFile: str


class TreePreparation(TypedDict, total=False):
    targetId: str
    oldLeafId: str | None
    commonAncestorId: str | None
    entriesToSummarize: list[SessionEntry]
    userWantsSummary: bool
    customInstructions: str
    replaceInstructions: bool
    label: str


class SessionBeforeTreeEvent(TypedDict):
    type: Literal["session_before_tree"]
    preparation: TreePreparation
    signal: Any


class SessionTreeEvent(TypedDict, total=False):
    type: Literal["session_tree"]
    newLeafId: str | None
    oldLeafId: str | None
    summaryEntry: dict[str, Any]
    fromExtension: bool | None


type SessionEvent = (
    SessionStartEvent
    | SessionBeforeSwitchEvent
    | SessionBeforeForkEvent
    | SessionBeforeCompactEvent
    | SessionCompactEvent
    | SessionShutdownEvent
    | SessionBeforeTreeEvent
    | SessionTreeEvent
)


class ContextEvent(TypedDict):
    type: Literal["context"]
    messages: list[AgentMessage]


class BeforeProviderRequestEvent(TypedDict):
    type: Literal["before_provider_request"]
    payload: Any


class AfterProviderResponseEvent(TypedDict):
    type: Literal["after_provider_response"]
    status: int
    headers: dict[str, str]


class BeforeAgentStartEvent(TypedDict, total=False):
    type: Literal["before_agent_start"]
    prompt: str
    images: list[ImageContent] | None
    systemPrompt: str
    systemPromptOptions: BuildSystemPromptOptions


class AgentStartEvent(TypedDict):
    type: Literal["agent_start"]


class AgentEndEvent(TypedDict):
    type: Literal["agent_end"]
    messages: list[AgentMessage]


class TurnStartEvent(TypedDict, total=False):
    type: Literal["turn_start"]
    turnIndex: int
    timestamp: int


class TurnEndEvent(TypedDict, total=False):
    type: Literal["turn_end"]
    turnIndex: int
    message: AgentMessage
    toolResults: list[AgentMessage]


class MessageStartEvent(TypedDict):
    type: Literal["message_start"]
    message: AgentMessage


class MessageUpdateEvent(TypedDict):
    type: Literal["message_update"]
    message: AgentMessage
    assistantMessageEvent: Any


class MessageEndEvent(TypedDict):
    type: Literal["message_end"]
    message: AgentMessage


class ToolExecutionStartEvent(TypedDict):
    type: Literal["tool_execution_start"]
    toolCallId: str
    toolName: str
    args: Any


class ToolExecutionUpdateEvent(TypedDict):
    type: Literal["tool_execution_update"]
    toolCallId: str
    toolName: str
    args: Any
    partialResult: Any


class ToolExecutionEndEvent(TypedDict):
    type: Literal["tool_execution_end"]
    toolCallId: str
    toolName: str
    result: Any
    isError: bool


class ModelSelectEvent(TypedDict, total=False):
    type: Literal["model_select"]
    model: Model[Any]
    previousModel: Model[Any] | None
    source: ModelSelectSource


class ThinkingLevelSelectEvent(TypedDict):
    type: Literal["thinking_level_select"]
    level: ThinkingLevel
    previousLevel: ThinkingLevel


class UserBashEvent(TypedDict):
    type: Literal["user_bash"]
    command: str
    excludeFromContext: bool
    cwd: str


class InputEvent(TypedDict, total=False):
    type: Literal["input"]
    text: str
    images: list[ImageContent]
    source: InputSource


class InputEventContinueResult(TypedDict):
    action: Literal["continue"]


class InputEventTransformResult(TypedDict, total=False):
    action: Literal["transform"]
    text: str
    images: list[ImageContent]


class InputEventHandledResult(TypedDict):
    action: Literal["handled"]


type InputEventResult = InputEventContinueResult | InputEventTransformResult | InputEventHandledResult


class ToolCallEventBase(TypedDict):
    type: Literal["tool_call"]
    toolCallId: str
    toolName: str
    input: dict[str, Any]


class BashToolCallEvent(ToolCallEventBase):
    toolName: Literal["bash"]


class ReadToolCallEvent(ToolCallEventBase):
    toolName: Literal["read"]


class EditToolCallEvent(ToolCallEventBase):
    toolName: Literal["edit"]


class WriteToolCallEvent(ToolCallEventBase):
    toolName: Literal["write"]


class GrepToolCallEvent(ToolCallEventBase):
    toolName: Literal["grep"]


class FindToolCallEvent(ToolCallEventBase):
    toolName: Literal["find"]


class LsToolCallEvent(ToolCallEventBase):
    toolName: Literal["ls"]


class CustomToolCallEvent(ToolCallEventBase):
    toolName: str


type ToolCallEvent = (
    BashToolCallEvent
    | ReadToolCallEvent
    | EditToolCallEvent
    | WriteToolCallEvent
    | GrepToolCallEvent
    | FindToolCallEvent
    | LsToolCallEvent
    | CustomToolCallEvent
)


class ToolResultEventBase(TypedDict):
    type: Literal["tool_result"]
    toolCallId: str
    toolName: str
    input: dict[str, Any]
    content: list[TextContent | ImageContent]
    details: Any
    isError: bool


class BashToolResultEvent(ToolResultEventBase):
    toolName: Literal["bash"]


class ReadToolResultEvent(ToolResultEventBase):
    toolName: Literal["read"]


class EditToolResultEvent(ToolResultEventBase):
    toolName: Literal["edit"]


class WriteToolResultEvent(ToolResultEventBase):
    toolName: Literal["write"]


class GrepToolResultEvent(ToolResultEventBase):
    toolName: Literal["grep"]


class FindToolResultEvent(ToolResultEventBase):
    toolName: Literal["find"]


class LsToolResultEvent(ToolResultEventBase):
    toolName: Literal["ls"]


class CustomToolResultEvent(ToolResultEventBase):
    toolName: str


type ToolResultEvent = (
    BashToolResultEvent
    | ReadToolResultEvent
    | EditToolResultEvent
    | WriteToolResultEvent
    | GrepToolResultEvent
    | FindToolResultEvent
    | LsToolResultEvent
    | CustomToolResultEvent
)


type ExtensionEvent = (
    ResourcesDiscoverEvent
    | SessionEvent
    | ContextEvent
    | BeforeProviderRequestEvent
    | AfterProviderResponseEvent
    | BeforeAgentStartEvent
    | AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ModelSelectEvent
    | ThinkingLevelSelectEvent
    | UserBashEvent
    | InputEvent
    | ToolCallEvent
    | ToolResultEvent
)


class ContextEventResult(TypedDict, total=False):
    messages: list[AgentMessage]


type BeforeProviderRequestEventResult = Any


class ToolCallEventResult(TypedDict, total=False):
    block: bool
    reason: str


class UserBashEventResult(TypedDict, total=False):
    operations: Any
    result: Any


class ToolResultEventResult(TypedDict, total=False):
    content: list[TextContent | ImageContent]
    details: Any
    isError: bool


class MessageEndEventResult(TypedDict, total=False):
    message: AgentMessage


class BeforeAgentStartEventResult(TypedDict, total=False):
    message: Any
    systemPrompt: str


class SessionBeforeSwitchResult(TypedDict, total=False):
    cancel: bool


class SessionBeforeForkResult(TypedDict, total=False):
    cancel: bool
    skipConversationRestore: bool


class SessionBeforeCompactResult(TypedDict, total=False):
    cancel: bool
    compaction: SessionCompactionResult


class BranchSummaryPayload(TypedDict, total=False):
    summary: str
    details: Any


class SessionBeforeTreeResult(TypedDict, total=False):
    cancel: bool
    summary: BranchSummaryPayload
    customInstructions: str
    replaceInstructions: bool
    label: str


@dataclass(slots=True)
class ExtensionRuntime:
    sendMessage: SendMessageHandler
    sendUserMessage: SendUserMessageHandler
    appendEntry: AppendEntryHandler
    setSessionName: SetSessionNameHandler
    getSessionName: GetSessionNameHandler
    setLabel: SetLabelHandler
    getActiveTools: GetActiveToolsHandler
    getAllTools: GetAllToolsHandler
    setActiveTools: SetActiveToolsHandler
    refreshTools: RefreshToolsHandler
    getCommands: GetCommandsHandler
    setModel: SetModelHandler
    getThinkingLevel: GetThinkingLevelHandler
    setThinkingLevel: SetThinkingLevelHandler
    flagValues: dict[str, bool | str] = field(default_factory=dict)
    pendingProviderRegistrations: list[PendingProviderRegistration] = field(default_factory=list)
    assertActive: Callable[[], None] = lambda: None
    invalidate: Callable[[str | None], None] = lambda _message=None: None
    registerProvider: RegisterProviderHandler = lambda _name, _config, _extension_path=None: None
    unregisterProvider: UnregisterProviderHandler = lambda _name, _extension_path=None: None
    loadedModules: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Extension:
    path: str
    resolvedPath: str
    sourceInfo: SourceInfo
    handlers: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    messageRenderers: dict[str, MessageRenderer[Any]] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[KeyId, ExtensionShortcut] = field(default_factory=dict)
    skillPaths: list[str] = field(default_factory=list)
    promptPaths: list[str] = field(default_factory=list)
    themePaths: list[str] = field(default_factory=list)
    systemPrompt: str | None = None
    appendSystemPrompt: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadExtensionsResult:
    extensions: list[Extension]
    errors: list[dict[str, str]]
    runtime: ExtensionRuntime


@dataclass(slots=True)
class ExtensionError:
    extensionPath: str
    event: str
    error: str
    stack: str | None = None


class ExtensionContext(Protocol):
    def __getitem__(self, key: str) -> Any: ...

    def get(self, key: str, default: Any = None) -> Any: ...


class ExtensionCommandContext(ExtensionContext, Protocol):
    async def waitForIdle(self) -> None: ...

    async def newSession(self, options: dict[str, Any] | None = None) -> dict[str, bool]: ...

    async def fork(self, entryId: str, options: dict[str, Any] | None = None) -> dict[str, bool]: ...

    async def navigateTree(self, targetId: str, options: dict[str, Any] | None = None) -> dict[str, bool]: ...

    async def switchSession(self, sessionPath: str, options: dict[str, Any] | None = None) -> dict[str, bool]: ...

    async def reload(self) -> None: ...


class ExtensionActions(Protocol):
    sendMessage: SendMessageHandler
    sendUserMessage: SendUserMessageHandler
    appendEntry: AppendEntryHandler
    setSessionName: SetSessionNameHandler
    getSessionName: GetSessionNameHandler
    setLabel: SetLabelHandler
    getActiveTools: GetActiveToolsHandler
    getAllTools: GetAllToolsHandler
    setActiveTools: SetActiveToolsHandler
    refreshTools: RefreshToolsHandler
    getCommands: GetCommandsHandler
    setModel: SetModelHandler
    getThinkingLevel: GetThinkingLevelHandler
    setThinkingLevel: SetThinkingLevelHandler


class ExtensionContextActions(Protocol):
    getModel: Callable[[], Model[Any] | None]
    isIdle: Callable[[], bool]
    getSignal: Callable[[], Any | None]
    abort: Callable[[], None]
    hasPendingMessages: Callable[[], bool]
    shutdown: Callable[[], None]
    getContextUsage: Callable[[], ContextUsage | None]
    compact: Callable[[CompactOptions | None], None]
    getSystemPrompt: Callable[[], str]


class ExtensionCommandContextActions(Protocol):
    waitForIdle: Callable[[], Awaitable[None]]
    newSession: Callable[[dict[str, Any] | None], Awaitable[dict[str, bool]]]
    fork: Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
    navigateTree: Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
    switchSession: Callable[[str, dict[str, Any] | None], Awaitable[dict[str, bool]]]
    reload: Callable[[], Awaitable[None]]


class ExtensionAPI(Protocol):
    cwd: str
    extension: Extension
    events: Any

    def on(self, event: str, handler: Callable[..., Any]) -> None: ...

    def register_tool(
        self,
        definition: ToolDefinition[Any, Any],
        *,
        source_path: str | None = None,
        source_info: SourceInfo | None = None,
    ) -> None: ...

    def registerCommand(
        self,
        name: str,
        options: dict[str, Any],
    ) -> None: ...

    def registerShortcut(
        self,
        shortcut: KeyId,
        options: dict[str, Any],
    ) -> None: ...

    def registerFlag(
        self,
        name: str,
        options: dict[str, Any],
    ) -> None: ...

    def registerMessageRenderer(self, customType: str, renderer: MessageRenderer[Any]) -> None: ...

    def getFlag(self, name: str) -> bool | str | None: ...

    def sendMessage(self, message: Any, options: dict[str, Any] | None = None) -> None: ...

    def sendUserMessage(
        self,
        content: str | list[TextContent | ImageContent],
        options: dict[str, Any] | None = None,
    ) -> None: ...

    def appendEntry(self, customType: str, data: Any = None) -> None: ...

    def setSessionName(self, name: str) -> None: ...

    def getSessionName(self) -> str | None: ...

    def setLabel(self, entryId: str, label: str | None) -> None: ...

    async def exec(self, command: str, args: list[str], options: ExecOptions | None = None) -> ExecResult: ...

    def getActiveTools(self) -> list[str]: ...

    def getAllTools(self) -> list[ToolInfo]: ...

    def setActiveTools(self, toolNames: list[str]) -> None: ...

    def getCommands(self) -> list[SlashCommandInfo]: ...

    async def setModel(self, model: Model[Any]) -> bool: ...

    def getThinkingLevel(self) -> ThinkingLevel: ...

    def setThinkingLevel(self, level: ThinkingLevel) -> None: ...

    def registerProvider(self, name: str, config: ProviderConfig) -> None: ...

    def unregisterProvider(self, name: str) -> None: ...

    def add_skill_path(self, path: str) -> None: ...

    def add_prompt_path(self, path: str) -> None: ...

    def add_theme_path(self, path: str) -> None: ...

    def set_system_prompt(self, prompt: str | None) -> None: ...

    def append_system_prompt(self, prompt: str) -> None: ...


type ExtensionFactory = Callable[[ExtensionAPI], Awaitable[None] | None]


def is_tool_call_event_type(event_type: str) -> bool:
    return event_type == "tool_call"


__all__ = [
    "AfterProviderResponseEvent",
    "AgentEndEvent",
    "AgentMessage",
    "AgentStartEvent",
    "AppendEntryHandler",
    "BashToolCallEvent",
    "BashToolResultEvent",
    "BeforeAgentStartEvent",
    "BeforeAgentStartEventResult",
    "BeforeProviderRequestEvent",
    "BeforeProviderRequestEventResult",
    "BranchSummaryPayload",
    "BuildSystemPromptOptions",
    "CompactOptions",
    "ContextUsage",
    "ContextEvent",
    "ContextEventResult",
    "CustomMessage",
    "CustomToolCallEvent",
    "CustomToolResultEvent",
    "EditToolCallEvent",
    "EditToolResultEvent",
    "ExecOptions",
    "ExecResult",
    "Extension",
    "ExtensionAPI",
    "ExtensionActions",
    "ExtensionCommandContext",
    "ExtensionCommandContextActions",
    "ExtensionContext",
    "ExtensionContextActions",
    "ExtensionError",
    "ExtensionErrorListener",
    "ExtensionEvent",
    "ExtensionFactory",
    "ExtensionFlag",
    "ExtensionHandler",
    "ExtensionRuntime",
    "ExtensionShortcut",
    "ExtensionUIContext",
    "FindToolCallEvent",
    "FindToolResultEvent",
    "GetActiveToolsHandler",
    "GetAllToolsHandler",
    "GetCommandsHandler",
    "GetSessionNameHandler",
    "GetThinkingLevelHandler",
    "GrepToolCallEvent",
    "GrepToolResultEvent",
    "ImageContent",
    "InputEvent",
    "InputEventContinueResult",
    "InputEventHandledResult",
    "InputEventResult",
    "InputEventTransformResult",
    "InputSource",
    "KeyId",
    "LoadExtensionsResult",
    "LsToolCallEvent",
    "LsToolResultEvent",
    "MessageEndEvent",
    "MessageEndEventResult",
    "MessageRenderOptions",
    "MessageRenderer",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ModelSelectEvent",
    "ModelSelectSource",
    "PendingProviderRegistration",
    "ProviderConfig",
    "ProviderModelConfig",
    "ReadToolCallEvent",
    "ReadToolResultEvent",
    "RegisteredCommand",
    "RegisteredTool",
    "ResourcesDiscoverEvent",
    "ResourcesDiscoverResult",
    "ForkHandler",
    "NavigateTreeHandler",
    "NewSessionHandler",
    "ReloadHandler",
    "ResolvedCommand",
    "SessionBeforeCompactEvent",
    "SessionBeforeCompactResult",
    "SessionBeforeForkEvent",
    "SessionBeforeForkResult",
    "SessionBeforeSwitchEvent",
    "SessionBeforeSwitchResult",
    "SessionBeforeTreeEvent",
    "SessionBeforeTreeResult",
    "SessionCompactEvent",
    "SessionEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "SessionTreeEvent",
    "SendMessageHandler",
    "SendUserMessageHandler",
    "SetActiveToolsHandler",
    "SetLabelHandler",
    "SetModelHandler",
    "SetSessionNameHandler",
    "SetThinkingLevelHandler",
    "ShutdownHandler",
    "SlashCommandInfo",
    "SwitchSessionHandler",
    "TextContent",
    "ThinkingLevel",
    "ThinkingLevelSelectEvent",
    "ToolCallEvent",
    "ToolCallEventBase",
    "ToolCallRenderer",
    "ToolDefinition",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolInfo",
    "ToolExecutionMode",
    "ToolRenderContext",
    "ToolRenderResultOptions",
    "ToolRenderShell",
    "ToolResultEvent",
    "ToolResultEventBase",
    "ToolResultEventResult",
    "ToolResultRenderer",
    "TreePreparation",
    "TurnEndEvent",
    "TurnStartEvent",
    "UserBashEvent",
    "UserBashEventResult",
    "WriteToolCallEvent",
    "WriteToolResultEvent",
    "is_tool_call_event_type",
]
