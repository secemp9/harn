> harn can help you use the SDK. Ask it to build an integration for your use case.

# SDK

The SDK provides programmatic access to harn's agent capabilities. Use it to embed harn in other applications, build custom interfaces, or integrate with automated workflows.

**Example use cases:**
- Build a custom UI (web, desktop, mobile)
- Integrate agent capabilities into existing applications
- Create automated pipelines with agent reasoning
- Build custom tools that spawn sub-agents
- Test agent behavior programmatically

## Quick Start

```python
from harn_coding_agent import AuthStorage, create_agent_session, ModelRegistry, SessionManager

# Set up credential storage and model registry
auth_storage = AuthStorage.create()
model_registry = ModelRegistry.create(auth_storage)

result = await create_agent_session(
    session_manager=SessionManager.in_memory(),
    auth_storage=auth_storage,
    model_registry=model_registry,
)
session = result.session

def on_event(event):
    if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
        print(event.assistant_message_event.delta, end="", flush=True)

session.subscribe(on_event)

await session.prompt("What files are in the current directory?")
```

## Installation

```bash
pip install harn
```

The SDK is included in the main package. No separate installation needed.

## Core Concepts

### create_agent_session()

The main factory function for a single `AgentSession`.

`create_agent_session()` uses a `ResourceLoader` to supply extensions, skills, prompt templates, themes, and context files. If you do not provide one, it uses `DefaultResourceLoader` with standard discovery.

```python
from harn_coding_agent import create_agent_session, SessionManager

# Minimal: defaults with DefaultResourceLoader
result = await create_agent_session()
session = result.session

# Custom: override specific options
result = await create_agent_session(
    model=my_model,
    tools=["read", "bash"],
    session_manager=SessionManager.in_memory(),
)
session = result.session
```

### AgentSession

The session manages agent lifecycle, message history, model state, compaction, and event streaming.

```python
class AgentSession:
    # Send a prompt and wait for completion
    async def prompt(self, text: str, options: PromptOptions = None) -> None: ...

    # Queue messages during streaming
    async def steer(self, text: str) -> None: ...
    async def follow_up(self, text: str) -> None: ...

    # Subscribe to events (returns unsubscribe function)
    def subscribe(self, listener: Callable[[AgentSessionEvent], None]) -> Callable[[], None]: ...

    # Session info
    session_file: str | None
    session_id: str

    # Model control
    async def set_model(self, model: Model) -> None: ...
    def set_thinking_level(self, level: ThinkingLevel) -> None: ...
    async def cycle_model(self) -> ModelCycleResult | None: ...
    def cycle_thinking_level(self) -> ThinkingLevel | None: ...

    # State access
    agent: Agent
    model: Model | None
    thinking_level: ThinkingLevel
    messages: list[AgentMessage]
    is_streaming: bool

    # In-place tree navigation within the current session file
    async def navigate_tree(self, target_id: str, options: dict = None) -> dict: ...

    # Compaction
    async def compact(self, custom_instructions: str = None) -> CompactionResult: ...
    def abort_compaction(self) -> None: ...

    # Abort current operation
    async def abort(self) -> None: ...

    # Cleanup
    def dispose(self) -> None: ...
```

Session replacement APIs such as new-session, resume, fork, and import live on `AgentSessionRuntime`, not on `AgentSession`.

### create_agent_session_runtime() and AgentSessionRuntime

Use the runtime API when you need to replace the active session and rebuild cwd-bound runtime state.
This is the same layer used by the built-in interactive, print, and RPC modes.

```python
from harn_coding_agent import (
    create_agent_session_from_services,
    create_agent_session_runtime,
    create_agent_session_services,
    get_agent_dir,
    SessionManager,
)

async def create_runtime(cwd, session_manager, session_start_event):
    services = await create_agent_session_services(cwd=cwd)
    result = await create_agent_session_from_services(
        services=services,
        session_manager=session_manager,
        session_start_event=session_start_event,
    )
    return {
        **result,
        "services": services,
        "diagnostics": services.diagnostics,
    }

runtime = await create_agent_session_runtime(
    create_runtime,
    cwd=os.getcwd(),
    agent_dir=get_agent_dir(),
    session_manager=SessionManager.create(os.getcwd()),
)
```

`AgentSessionRuntime` owns replacement of the active runtime across:

- `new_session()`
- `switch_session()`
- `fork()`
- clone flows via `fork(entry_id, position="at")`
- `import_from_jsonl()`

Important behavior:

- `runtime.session` changes after those operations
- event subscriptions are attached to a specific `AgentSession`, so re-subscribe after replacement
- if you use extensions, call `runtime.session.bind_extensions(...)` again for the new session
- creation returns diagnostics on `runtime.diagnostics`
- if runtime creation or replacement fails, the method throws and the caller decides how to handle it

```python
session = runtime.session
unsubscribe = session.subscribe(lambda event: None)

await runtime.new_session()

unsubscribe()
session = runtime.session
unsubscribe = session.subscribe(lambda event: None)
```

### Prompting and Message Queueing

`PromptOptions` controls prompt expansion, queueing behavior while streaming, and prompt preflight notifications:

```python
@dataclass
class PromptOptions:
    expand_prompt_templates: bool = True
    images: list[ImageContent] = None
    streaming_behavior: str = None  # "steer" or "followUp"
    source: InputSource = None
    preflight_result: Callable[[bool], None] = None
```

The `prompt()` method handles prompt templates, extension commands, and message sending:

```python
# Basic prompt (when not streaming)
await session.prompt("What files are here?")

# With images
await session.prompt("What's in this image?", PromptOptions(
    images=[ImageContent(type="image", data="...", mime_type="image/png")]
))

# During streaming: must specify how to queue the message
await session.prompt("Stop and do this instead", PromptOptions(streaming_behavior="steer"))
await session.prompt("After you're done, also check X", PromptOptions(streaming_behavior="followUp"))
```

For explicit queueing during streaming:

```python
# Queue a steering message for delivery after the current assistant turn finishes its tool calls
await session.steer("New instruction")

# Wait for agent to finish (delivered only when agent stops)
await session.follow_up("After you're done, also do this")
```

### Agent and AgentState

The `Agent` class (from `harn_agent`) handles the core LLM interaction. Access it via `session.agent`.

```python
# Access current state
state = session.agent.state

# state.messages: list[AgentMessage] - conversation history
# state.model: Model - current model
# state.thinking_level: ThinkingLevel - current thinking level
# state.system_prompt: str - system prompt
# state.tools: list[AgentTool] - available tools
# state.streaming_message: AgentMessage | None - current partial assistant message
# state.error_message: str | None - latest assistant error

# Wait for agent to finish processing
await session.agent.wait_for_idle()
```

### Events

Subscribe to events to receive streaming output and lifecycle notifications.

```python
def on_event(event):
    match event.type:
        # Streaming text from assistant
        case "message_update":
            if event.assistant_message_event.type == "text_delta":
                print(event.assistant_message_event.delta, end="", flush=True)

        # Tool execution
        case "tool_execution_start":
            print(f"Tool: {event.tool_name}")
        case "tool_execution_end":
            print(f"Result: {'error' if event.is_error else 'success'}")

        # Agent lifecycle
        case "agent_end":
            # event.messages contains new messages
            pass

        # Session events (queue, compaction, retry)
        case "queue_update":
            print(event.steering, event.follow_up)

session.subscribe(on_event)
```

## Options Reference

### Directories

```python
result = await create_agent_session(
    # Working directory for DefaultResourceLoader discovery
    cwd=os.getcwd(),  # default

    # Global config directory
    agent_dir="~/.harn/agent",  # default (expands ~)
)
```

`cwd` is used by `DefaultResourceLoader` for:
- Project extensions (`.harn/extensions/`)
- Project skills:
  - `.harn/skills/`
  - `.agents/skills/` in `cwd` and ancestor directories (up to git repo root, or filesystem root when not in a repo)
- Project prompts (`.harn/prompts/`)
- Context files (`AGENTS.md` walking up from cwd)
- Session directory naming

`agent_dir` is used by `DefaultResourceLoader` for:
- Global extensions (`extensions/`)
- Global skills:
  - `skills/` under `agent_dir` (for example `~/.harn/agent/skills/`)
  - `~/.agents/skills/`
- Global prompts (`prompts/`)
- Global context file (`AGENTS.md`)
- Settings (`settings.json`)
- Custom models (`models.json`)
- Credentials (`auth.json`)
- Sessions (`sessions/`)

When you pass a custom `ResourceLoader`, `cwd` and `agent_dir` no longer control resource discovery. They still influence session naming and tool path resolution.

### Model

```python
from harn_ai import get_model
from harn_coding_agent import AuthStorage, ModelRegistry

auth_storage = AuthStorage.create()
model_registry = ModelRegistry.create(auth_storage)

# Find specific built-in model (doesn't check if API key exists)
opus = get_model("anthropic", "claude-opus-4-5")
if not opus:
    raise ValueError("Model not found")

# Find any model by provider/id, including custom models from models.json
# (doesn't check if API key exists)
custom_model = model_registry.find("my-provider", "my-model")

# Get only models that have valid API keys configured
available = await model_registry.get_available()

result = await create_agent_session(
    model=opus,
    thinking_level="medium",  # off, minimal, low, medium, high, xhigh

    # Models for cycling (Ctrl+P in interactive mode)
    scoped_models=[
        {"model": opus, "thinking_level": "high"},
        {"model": haiku, "thinking_level": "off"},
    ],

    auth_storage=auth_storage,
    model_registry=model_registry,
)
```

If no model is provided:
1. Tries to restore from session (if continuing)
2. Uses default from settings
3. Falls back to first available model

### API Keys and OAuth

API key resolution priority (handled by AuthStorage):
1. Runtime overrides (via `set_runtime_api_key`, not persisted)
2. Stored credentials in `auth.json` (API keys or OAuth tokens)
3. Environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.)
4. Fallback resolver (for custom provider keys from `models.json`)

```python
from harn_coding_agent import AuthStorage, ModelRegistry

# Default: uses ~/.harn/agent/auth.json and ~/.harn/agent/models.json
auth_storage = AuthStorage.create()
model_registry = ModelRegistry.create(auth_storage)

result = await create_agent_session(
    session_manager=SessionManager.in_memory(),
    auth_storage=auth_storage,
    model_registry=model_registry,
)

# Runtime API key override (not persisted to disk)
auth_storage.set_runtime_api_key("anthropic", "sk-my-temp-key")

# Custom auth storage location
custom_auth = AuthStorage.create("/my/app/auth.json")
custom_registry = ModelRegistry.create(custom_auth, "/my/app/models.json")
```

### System Prompt

Use a `ResourceLoader` to override the system prompt:

```python
from harn_coding_agent import create_agent_session, DefaultResourceLoader

loader = DefaultResourceLoader(
    system_prompt_override=lambda: "You are a helpful assistant.",
)
await loader.reload()

result = await create_agent_session(resource_loader=loader)
```

### Tools

Specify which built-in tools to enable:

- Built-in tool names: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`
- Default built-ins: `read`, `bash`, `edit`, `write`
- `no_tools="all"` disables all tools
- `no_tools="builtin"` disables default built-ins while keeping extension and custom tools enabled

```python
from harn_coding_agent import create_agent_session

# Read-only mode
result = await create_agent_session(
    tools=["read", "grep", "find", "ls"],
)

# Pick specific tools
result = await create_agent_session(
    tools=["read", "bash", "grep"],
)
```

### Custom Tools

```python
from pydantic import BaseModel, Field
from harn_coding_agent import create_agent_session, define_tool

class MyToolParams(BaseModel):
    input: str = Field(description="Input value")

async def my_tool_execute(tool_call_id, params):
    return {
        "content": [{"type": "text", "text": f"Result: {params.input}"}],
        "details": {},
    }

my_tool = define_tool(
    name="my_tool",
    label="My Tool",
    description="Does something useful",
    parameters=MyToolParams,
    execute=my_tool_execute,
)

# Pass custom tools directly
result = await create_agent_session(
    custom_tools=[my_tool],
)
```

Custom tools passed via `custom_tools` are combined with extension-registered tools. Extensions loaded by the ResourceLoader can also register tools via `harn.register_tool()`.

### Extensions

Extensions are loaded by the `ResourceLoader`. `DefaultResourceLoader` discovers extensions from `~/.harn/agent/extensions/`, `.harn/extensions/`, and settings.json extension sources.

```python
from harn_coding_agent import create_agent_session, DefaultResourceLoader

loader = DefaultResourceLoader(
    additional_extension_paths=["/path/to/my_extension.py"],
    extension_factories=[
        lambda harn: harn.on("agent_start", lambda: print("[Inline Extension] Agent starting")),
    ],
)
await loader.reload()

result = await create_agent_session(resource_loader=loader)
```

Extensions can register tools, subscribe to events, add commands, and more. See [extensions.md](extensions.md) for the full API.

**Event Bus:** Extensions can communicate via `harn.events`. Pass a shared `event_bus` to `DefaultResourceLoader` if you need to emit or listen from outside:

```python
from harn_coding_agent import create_event_bus, DefaultResourceLoader

event_bus = create_event_bus()
loader = DefaultResourceLoader(event_bus=event_bus)
await loader.reload()

event_bus.on("my-extension:status", lambda data: print(data))
```

### Skills

```python
from harn_coding_agent import create_agent_session, DefaultResourceLoader, Skill

custom_skill = Skill(
    name="my-skill",
    description="Custom instructions",
    file_path="/path/to/SKILL.md",
    base_dir="/path/to",
    source="custom",
)

loader = DefaultResourceLoader(
    skills_override=lambda current: {
        "skills": [*current.skills, custom_skill],
        "diagnostics": current.diagnostics,
    },
)
await loader.reload()

result = await create_agent_session(resource_loader=loader)
```

### Context Files

```python
from harn_coding_agent import create_agent_session, DefaultResourceLoader

loader = DefaultResourceLoader(
    agents_files_override=lambda current: {
        "agents_files": [
            *current.agents_files,
            {"path": "/virtual/AGENTS.md", "content": "# Guidelines\n\n- Be concise"},
        ],
    },
)
await loader.reload()

result = await create_agent_session(resource_loader=loader)
```

### Session Management

Sessions use a tree structure with `id`/`parentId` linking, enabling in-place branching.

```python
from harn_coding_agent import create_agent_session, SessionManager

# In-memory (no persistence)
result = await create_agent_session(
    session_manager=SessionManager.in_memory(),
)

# New persistent session
result = await create_agent_session(
    session_manager=SessionManager.create(os.getcwd()),
)

# Continue most recent
result = await create_agent_session(
    session_manager=SessionManager.continue_recent(os.getcwd()),
)

# Open specific file
result = await create_agent_session(
    session_manager=SessionManager.open("/path/to/session.jsonl"),
)

# List sessions
current_project_sessions = await SessionManager.list(os.getcwd())
all_sessions = await SessionManager.list_all(os.getcwd())
```

See [Session Format](session-format.md) for details.

### Settings Management

```python
from harn_coding_agent import create_agent_session, SettingsManager, SessionManager

# Default: loads from files (global + project merged)
result = await create_agent_session(
    settings_manager=SettingsManager.create(),
)

# With overrides
settings_manager = SettingsManager.create()
settings_manager.apply_overrides({
    "compaction": {"enabled": False},
    "retry": {"enabled": True, "maxRetries": 5},
})
result = await create_agent_session(settings_manager=settings_manager)

# In-memory (no file I/O, for testing)
result = await create_agent_session(
    settings_manager=SettingsManager.in_memory({"compaction": {"enabled": False}}),
    session_manager=SessionManager.in_memory(),
)
```

**Project-specific settings:**

Settings load from two locations and merge:
1. Global: `~/.harn/agent/settings.json`
2. Project: `<cwd>/.harn/settings.json`

Project overrides global. Nested objects merge keys. Setters modify global settings by default.

## ResourceLoader

Use `DefaultResourceLoader` to discover extensions, skills, prompts, themes, and context files.

```python
from harn_coding_agent import DefaultResourceLoader, get_agent_dir

loader = DefaultResourceLoader(
    cwd=os.getcwd(),
    agent_dir=get_agent_dir(),
)
await loader.reload()

extensions = loader.get_extensions()
skills = loader.get_skills()
prompts = loader.get_prompts()
themes = loader.get_themes()
context_files = loader.get_agents_files().agents_files
```

## RPC Mode Alternative

For subprocess-based integration without building with the SDK, use the CLI directly:

```bash
harn --mode rpc --no-session
```

See [RPC documentation](rpc.md) for the JSON protocol.

The SDK is preferred when:
- You want type safety
- You're in the same Python process
- You need direct access to agent state
- You want to customize tools/extensions programmatically

RPC mode is preferred when:
- You're integrating from another language
- You want process isolation
- You're building a language-agnostic client

## Exports

The main entry point exports:

```python
# Factory
create_agent_session
create_agent_session_runtime
AgentSessionRuntime

# Auth and Models
AuthStorage
ModelRegistry

# Resource loading
DefaultResourceLoader
ResourceLoader  # Protocol
create_event_bus

# Helpers
define_tool

# Session management
SessionManager
SettingsManager

# Tool factories
create_coding_tools
create_read_only_tools
create_read_tool, create_bash_tool, create_edit_tool, create_write_tool
create_grep_tool, create_find_tool, create_ls_tool

# Types
CreateAgentSessionOptions
CreateAgentSessionResult
ExtensionFactory
ExtensionAPI
ToolDefinition
Skill
PromptTemplate
Tool
```

For extension types, see [extensions.md](extensions.md) for the full API.
