> harn can create extensions. Ask it to build one for your use case.

# Extensions

Extensions are Python modules that extend harn's behavior. They can subscribe to lifecycle events, register custom tools callable by the LLM, add commands, and more.

> **Placement for /reload:** Put extensions in `~/.harn/agent/extensions/` (global) or `.harn/extensions/` (project-local) for auto-discovery. Use `harn -e ./path.py` only for quick tests. Extensions in auto-discovered locations can be hot-reloaded with `/reload`.

**Key capabilities:**
- **Custom tools** - Register tools the LLM can call via `harn.register_tool()`
- **Event interception** - Block or modify tool calls, inject context, customize compaction
- **User interaction** - Prompt users via `ctx.ui` (select, confirm, input, notify)
- **Custom UI components** - Full TUI components with keyboard input via `ctx.ui.custom()` for complex interactions
- **Custom commands** - Register commands like `/mycommand` via `harn.register_command()`
- **Session persistence** - Store state that survives restarts via `harn.append_entry()`
- **Custom rendering** - Control how tool calls/results and messages appear in TUI

**Example use cases:**
- Permission gates (confirm before `rm -rf`, `sudo`, etc.)
- Git checkpointing (stash at each turn, restore on branch)
- Path protection (block writes to `.env`, `node_modules/`)
- Custom compaction (summarize conversation your way)
- Conversation summaries (see `summarize.py` example)
- Interactive tools (questions, wizards, custom dialogs)
- Stateful tools (todo lists, connection pools)
- External integrations (file watchers, webhooks, CI triggers)
- Games while you wait (see `snake.py` example)

See [examples/extensions/](../examples/extensions/) for working implementations.

## Table of Contents

- [Quick Start](#quick-start)
- [Extension Locations](#extension-locations)
- [Available Imports](#available-imports)
- [Writing an Extension](#writing-an-extension)
  - [Extension Styles](#extension-styles)
- [Events](#events)
  - [Lifecycle Overview](#lifecycle-overview)
  - [Resource Events](#resource-events)
  - [Session Events](#session-events)
  - [Agent Events](#agent-events)
  - [Model Events](#model-events)
  - [Tool Events](#tool-events)
- [ExtensionContext](#extensioncontext)
- [ExtensionCommandContext](#extensioncommandcontext)
- [ExtensionAPI Methods](#extensionapi-methods)
- [State Management](#state-management)
- [Custom Tools](#custom-tools)
- [Custom UI](#custom-ui)
- [Error Handling](#error-handling)
- [Mode Behavior](#mode-behavior)
- [Examples Reference](#examples-reference)

## Quick Start

Create `~/.harn/agent/extensions/my_extension.py`:

```python
from harn_coding_agent import ExtensionAPI
from pydantic import BaseModel, Field


def extension_factory(harn: ExtensionAPI):
    # React to events
    @harn.on("session_start")
    async def on_session_start(event, ctx):
        ctx.ui.notify("Extension loaded!", "info")

    @harn.on("tool_call")
    async def on_tool_call(event, ctx):
        if event.tool_name == "bash" and "rm -rf" in (event.input.get("command") or ""):
            ok = await ctx.ui.confirm("Dangerous!", "Allow rm -rf?")
            if not ok:
                return {"block": True, "reason": "Blocked by user"}

    # Register a custom tool
    class GreetParams(BaseModel):
        name: str = Field(description="Name to greet")

    @harn.register_tool(
        name="greet",
        label="Greet",
        description="Greet someone by name",
        parameters=GreetParams,
    )
    async def greet(tool_call_id, params, signal, on_update, ctx):
        return {
            "content": [{"type": "text", "text": f"Hello, {params.name}!"}],
            "details": {},
        }

    # Register a command
    @harn.register_command("hello", description="Say hello")
    async def hello(args, ctx):
        ctx.ui.notify(f"Hello {args or 'world'}!", "info")
```

Test with `--extension` (or `-e`) flag:

```bash
harn -e ./my_extension.py
```

## Extension Locations

> **Security:** Extensions run with your full system permissions and can execute arbitrary code. Only install from sources you trust.

Extensions are auto-discovered from:

| Location | Scope |
|----------|-------|
| `~/.harn/agent/extensions/*.py` | Global (all projects) |
| `~/.harn/agent/extensions/*/index.py` | Global (subdirectory) |
| `.harn/extensions/*.py` | Project-local |
| `.harn/extensions/*/index.py` | Project-local (subdirectory) |

Additional paths via `settings.json`:

```json
{
  "packages": [
    "pip:some-package>=1.0.0",
    "git:github.com/user/repo@v1"
  ],
  "extensions": [
    "/path/to/local/extension.py",
    "/path/to/local/extension/dir"
  ]
}
```

To share extensions via PyPI or git as harn packages, see [packages.md](packages.md).

## Available Imports

| Package | Purpose |
|---------|---------|
| `harn_coding_agent` | Extension types (`ExtensionAPI`, `ExtensionContext`, events) |
| `pydantic` | Schema definitions for tool parameters |
| `harn_ai` | AI utilities (`StringEnum` for Google-compatible enums) |
| `harn_tui` | TUI components for custom rendering |

pip/uv dependencies work too. Add a `pyproject.toml` or `requirements.txt` next to your extension (or in a parent directory), run `pip install` or `uv pip install`, and imports are resolved automatically.

For distributed harn packages installed with `harn install` (PyPI or git), runtime deps must be in `dependencies`. Package installation uses production installs by default, so dev dependencies are not available at runtime.

Python standard library modules (`os`, `pathlib`, `asyncio`, etc.) are also available.

## Writing an Extension

An extension exports a factory function that receives `ExtensionAPI`. The factory can be synchronous or asynchronous:

```python
from harn_coding_agent import ExtensionAPI


def extension_factory(harn: ExtensionAPI):
    # Subscribe to events
    @harn.on("event_name")
    async def on_event(event, ctx):
        # ctx.ui for user interaction
        ok = await ctx.ui.confirm("Title", "Are you sure?")
        ctx.ui.notify("Done!", "info")
        ctx.ui.set_status("my-ext", "Processing...")  # Footer status
        ctx.ui.set_widget("my-ext", ["Line 1", "Line 2"])  # Widget above editor (default)

    # Register tools, commands, shortcuts, flags
    harn.register_tool(...)
    harn.register_command("name", ...)
    harn.register_shortcut("ctrl+x", ...)
    harn.register_flag("my-flag", ...)
```

Extensions are loaded dynamically via importlib, so Python works without compilation.

If the factory returns a coroutine, harn awaits it before continuing startup. That means async initialization completes before `session_start`, before `resources_discover`, and before provider registrations queued via `harn.register_provider()` are flushed.

### Async factory functions

Use an async factory for one-time startup work such as fetching remote configuration or dynamically discovering available models.

```python
import httpx
from harn_coding_agent import ExtensionAPI


async def extension_factory(harn: ExtensionAPI):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models")
        payload = response.json()

    harn.register_provider("local-openai", {
        "base_url": "http://localhost:1234/v1",
        "api_key": "LOCAL_OPENAI_API_KEY",
        "api": "openai-completions",
        "models": [
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "context_window": model.get("context_window", 128000),
                "max_tokens": model.get("max_tokens", 4096),
            }
            for model in payload["data"]
        ],
    })
```

This pattern makes the fetched models available during normal startup and to `harn --list-models`.

### Extension Styles

**Single file** - simplest, for small extensions:

```
~/.harn/agent/extensions/
  my_extension.py
```

**Directory with index.py** - for multi-file extensions:

```
~/.harn/agent/extensions/
  my_extension/
      __init__.py       # or index.py - Entry point (exports extension_factory)
      tools.py          # Helper module
      utils.py          # Helper module
```

**Package with dependencies** - for extensions that need pip packages:

```
~/.harn/agent/extensions/
  my_extension/
      pyproject.toml    # Declares dependencies and entry points
      src/
          __init__.py
          index.py
```

```toml
# pyproject.toml
[project]
name = "my-extension"
dependencies = [
    "pydantic>=2.0.0",
    "rich>=13.0.0",
]

[tool.harn]
extensions = ["./src/index.py"]
```

Run `pip install` or `uv pip install` in the extension directory, then imports work automatically.

## Events

### Lifecycle Overview

```
harn starts
    |
    +-> session_start { reason: "startup" }
    +-> resources_discover { reason: "startup" }
        |
        v
user sends prompt -----------------------------------------------+
    |                                                            |
    +-> (extension commands checked first, bypass if found)      |
    +-> input (can intercept, transform, or handle)              |
    +-> (skill/template expansion if not handled)                |
    +-> before_agent_start (can inject message, modify system prompt)
    +-> agent_start                                              |
    +-> message_start / message_update / message_end             |
    |                                                            |
    |   +--- turn (repeats while LLM calls tools) ---+           |
    |   |                                            |           |
    |   +-> turn_start                               |           |
    |   +-> context (can modify messages)            |           |
    |   +-> before_provider_request (can inspect or replace payload)
    |   +-> after_provider_response (status + headers, before stream consume)
    |   |                                            |           |
    |   |   LLM responds, may call tools:            |           |
    |   |     +-> tool_execution_start               |           |
    |   |     +-> tool_call (can block)              |           |
    |   |     +-> tool_execution_update              |           |
    |   |     +-> tool_result (can modify)           |           |
    |   |     +-> tool_execution_end                 |           |
    |   |                                            |           |
    |   +-> turn_end                                 |           |
    |                                                            |
    +-> agent_end                                                |
                                                                 |
user sends another prompt <--------------------------------------+

/new (new session) or /resume (switch session)
    +-> session_before_switch (can cancel)
    +-> session_shutdown
    +-> session_start { reason: "new" | "resume", previous_session_file? }
    +-> resources_discover { reason: "startup" }

/fork or /clone
    +-> session_before_fork (can cancel)
    +-> session_shutdown
    +-> session_start { reason: "fork", previous_session_file }
    +-> resources_discover { reason: "startup" }

/compact or auto-compaction
    +-> session_before_compact (can cancel or customize)
    +-> session_compact

/tree navigation
    +-> session_before_tree (can cancel or customize)
    +-> session_tree

/model or Ctrl+P (model selection/cycling)
    +-> thinking_level_select (if model change changes/clamps thinking level)
    +-> model_select

thinking level changes (settings, keybinding, harn.set_thinking_level())
    +-> thinking_level_select

exit (Ctrl+C, Ctrl+D, SIGHUP, SIGTERM)
    +-> session_shutdown
```

### Resource Events

#### resources_discover

Fired after `session_start` so extensions can contribute additional skill, prompt, and theme paths.
The startup path uses `reason: "startup"`. Reload uses `reason: "reload"`.

```python
@harn.on("resources_discover")
async def on_resources_discover(event, ctx):
    # event.cwd - current working directory
    # event.reason - "startup" | "reload"
    return {
        "skill_paths": ["/path/to/skills"],
        "prompt_paths": ["/path/to/prompts"],
        "theme_paths": ["/path/to/themes"],
    }
```

### Session Events

See [Session Format](session-format.md) for session storage internals and the SessionManager API.

#### session_start

Fired when a session is started, loaded, or reloaded.

```python
@harn.on("session_start")
async def on_session_start(event, ctx):
    # event.reason - "startup" | "reload" | "new" | "resume" | "fork"
    # event.previous_session_file - present for "new", "resume", and "fork"
    ctx.ui.notify(f"Session: {ctx.session_manager.get_session_file() or 'ephemeral'}", "info")
```

#### session_before_switch

Fired before starting a new session (`/new`) or switching sessions (`/resume`).

```python
@harn.on("session_before_switch")
async def on_session_before_switch(event, ctx):
    # event.reason - "new" or "resume"
    # event.target_session_file - session we're switching to (only for "resume")

    if event.reason == "new":
        ok = await ctx.ui.confirm("Clear?", "Delete all messages?")
        if not ok:
            return {"cancel": True}
```

After a successful switch or new-session action, harn emits `session_shutdown` for the old extension instance, reloads and rebinds extensions for the new session, then emits `session_start` with `reason: "new" | "resume"` and `previous_session_file`.
Do cleanup work in `session_shutdown`, then reestablish any in-memory state in `session_start`.

#### session_before_fork

Fired when forking via `/fork` or cloning via `/clone`.

```python
@harn.on("session_before_fork")
async def on_session_before_fork(event, ctx):
    # event.entry_id - ID of the selected entry
    # event.position - "before" for /fork, "at" for /clone
    return {"cancel": True}  # Cancel fork/clone
    # OR
    return {"skip_conversation_restore": True}  # Reserved for future conversation restore control
```

After a successful fork or clone, harn emits `session_shutdown` for the old extension instance, reloads and rebinds extensions for the new session, then emits `session_start` with `reason: "fork"` and `previous_session_file`.
Do cleanup work in `session_shutdown`, then reestablish any in-memory state in `session_start`.

#### session_before_compact / session_compact

Fired on compaction. See [compaction.md](compaction.md) for details.

```python
@harn.on("session_before_compact")
async def on_session_before_compact(event, ctx):
    preparation, branch_entries, custom_instructions, signal = (
        event.preparation, event.branch_entries, event.custom_instructions, event.signal,
    )

    # Cancel:
    return {"cancel": True}

    # Custom summary:
    return {
        "compaction": {
            "summary": "...",
            "first_kept_entry_id": preparation.first_kept_entry_id,
            "tokens_before": preparation.tokens_before,
        }
    }


@harn.on("session_compact")
async def on_session_compact(event, ctx):
    # event.compaction_entry - the saved compaction
    # event.from_extension - whether extension provided it
    pass
```

#### session_before_tree / session_tree

Fired on `/tree` navigation. See [Sessions](sessions.md) for tree navigation concepts.

```python
@harn.on("session_before_tree")
async def on_session_before_tree(event, ctx):
    preparation, signal = event.preparation, event.signal
    return {"cancel": True}
    # OR provide custom summary:
    return {"summary": {"summary": "...", "details": {}}}


@harn.on("session_tree")
async def on_session_tree(event, ctx):
    # event.new_leaf_id, old_leaf_id, summary_entry, from_extension
    pass
```

#### session_shutdown

Fired before an extension runtime is torn down.

```python
@harn.on("session_shutdown")
async def on_session_shutdown(event, ctx):
    # event.reason - "quit" | "reload" | "new" | "resume" | "fork"
    # event.target_session_file - destination session for session replacement flows
    # Cleanup, save state, etc.
    pass
```

### Agent Events

#### before_agent_start

Fired after user submits prompt, before agent loop. Can inject a message and/or modify the system prompt.

```python
@harn.on("before_agent_start")
async def on_before_agent_start(event, ctx):
    # event.prompt - user's prompt text
    # event.images - attached images (if any)
    # event.system_prompt - current chained system prompt for this handler
    #   (includes changes from earlier before_agent_start handlers)
    # event.system_prompt_options - structured options used to build the system prompt
    #   .custom_prompt - any custom system prompt (from --system-prompt, SYSTEM.md, or custom templates)
    #   .selected_tools - tools currently active in the prompt
    #   .tool_snippets - one-line descriptions for each tool
    #   .prompt_guidelines - custom guideline bullets
    #   .append_system_prompt - text from --append-system-prompt flags
    #   .cwd - working directory
    #   .context_files - AGENTS.md files and other loaded context files
    #   .skills - loaded skills

    return {
        # Inject a persistent message (stored in session, sent to LLM)
        "message": {
            "custom_type": "my-extension",
            "content": "Additional context for the LLM",
            "display": True,
        },
        # Replace the system prompt for this turn (chained across extensions)
        "system_prompt": event.system_prompt + "\n\nExtra instructions for this turn...",
    }
```

The `system_prompt_options` field gives extensions access to the same structured data harn uses to build the system prompt. This lets you inspect what harn has loaded -- custom prompts, guidelines, tool snippets, context files, skills -- without re-discovering resources or re-parsing flags. Use it when your extension needs to make deep, informed changes to the system prompt while respecting user-provided configuration.

Inside `before_agent_start`, `event.system_prompt` and `ctx.get_system_prompt()` both reflect the chained system prompt as of the current handler. Later `before_agent_start` handlers can still modify it again.

#### agent_start / agent_end

Fired once per user prompt.

```python
@harn.on("agent_start")
async def on_agent_start(event, ctx):
    pass


@harn.on("agent_end")
async def on_agent_end(event, ctx):
    # event.messages - messages from this prompt
    pass
```

#### turn_start / turn_end

Fired for each turn (one LLM response + tool calls).

```python
@harn.on("turn_start")
async def on_turn_start(event, ctx):
    # event.turn_index, event.timestamp
    pass


@harn.on("turn_end")
async def on_turn_end(event, ctx):
    # event.turn_index, event.message, event.tool_results
    pass
```

#### message_start / message_update / message_end

Fired for message lifecycle updates.

- `message_start` and `message_end` fire for user, assistant, and toolResult messages.
- `message_update` fires for assistant streaming updates.
- `message_end` handlers can return `{ "message": ... }` to replace the finalized message. The replacement must keep the same `role`.

```python
@harn.on("message_start")
async def on_message_start(event, ctx):
    # event.message
    pass


@harn.on("message_update")
async def on_message_update(event, ctx):
    # event.message
    # event.assistant_message_event (token-by-token stream event)
    pass


@harn.on("message_end")
async def on_message_end(event, ctx):
    if event.message.role != "assistant":
        return

    return {
        "message": {
            **event.message,
            "usage": {
                **event.message.usage,
                "cost": {
                    **event.message.usage.cost,
                    "total": 0.123,
                },
            },
        },
    }
```

#### tool_execution_start / tool_execution_update / tool_execution_end

Fired for tool execution lifecycle updates.

In parallel tool mode:
- `tool_execution_start` is emitted in assistant source order during the preflight phase
- `tool_execution_update` events may interleave across tools
- `tool_execution_end` is emitted in tool completion order after each tool is finalized
- final `toolResult` message events are still emitted later in assistant source order

```python
@harn.on("tool_execution_start")
async def on_tool_execution_start(event, ctx):
    # event.tool_call_id, event.tool_name, event.args
    pass


@harn.on("tool_execution_update")
async def on_tool_execution_update(event, ctx):
    # event.tool_call_id, event.tool_name, event.args, event.partial_result
    pass


@harn.on("tool_execution_end")
async def on_tool_execution_end(event, ctx):
    # event.tool_call_id, event.tool_name, event.result, event.is_error
    pass
```

#### context

Fired before each LLM call. Modify messages non-destructively. See [Session Format](session-format.md) for message types.

```python
@harn.on("context")
async def on_context(event, ctx):
    # event.messages - deep copy, safe to modify
    filtered = [m for m in event.messages if not should_prune(m)]
    return {"messages": filtered}
```

#### before_provider_request

Fired after the provider-specific payload is built, right before the request is sent. Handlers run in extension load order. Returning `None` keeps the payload unchanged. Returning any other value replaces the payload for later handlers and for the actual request.

This hook can rewrite provider-level system instructions or remove them entirely. Those payload-level changes are not reflected by `ctx.get_system_prompt()`, which reports harn's system prompt string rather than the final serialized provider payload.

```python
@harn.on("before_provider_request")
def on_before_provider_request(event, ctx):
    import json
    print(json.dumps(event.payload, indent=2))

    # Optional: replace payload
    # return {**event.payload, "temperature": 0}
```

This is mainly useful for debugging provider serialization and cache behavior.

#### after_provider_response

Fired after an HTTP response is received and before its stream body is consumed. Handlers run in extension load order.

```python
@harn.on("after_provider_response")
def on_after_provider_response(event, ctx):
    # event.status - HTTP status code
    # event.headers - normalized response headers
    if event.status == 429:
        print("rate limited", event.headers.get("retry-after"))
```

Header availability depends on provider and transport. Providers that abstract HTTP responses may not expose headers.

### Model Events

#### model_select

Fired when the model changes via `/model` command, model cycling (`Ctrl+P`), or session restore.

```python
@harn.on("model_select")
async def on_model_select(event, ctx):
    # event.model - newly selected model
    # event.previous_model - previous model (None if first selection)
    # event.source - "set" | "cycle" | "restore"

    prev = (
        f"{event.previous_model.provider}/{event.previous_model.id}"
        if event.previous_model
        else "none"
    )
    next_model = f"{event.model.provider}/{event.model.id}"

    ctx.ui.notify(f"Model changed ({event.source}): {prev} -> {next_model}", "info")
```

Use this to update UI elements (status bars, footers) or perform model-specific initialization when the active model changes.

#### thinking_level_select

Fired when the thinking level changes. This is notification-only; handler return values are ignored.

```python
@harn.on("thinking_level_select")
async def on_thinking_level_select(event, ctx):
    # event.level - newly selected thinking level
    # event.previous_level - previous thinking level

    ctx.ui.set_status("thinking", f"thinking: {event.level}")
```

Use this to update extension UI when `harn.set_thinking_level()`, model changes, or built-in thinking-level controls change the active thinking level.

### Tool Events

#### tool_call

Fired after `tool_execution_start`, before the tool executes. **Can block.** Use `is_tool_call_event_type` to narrow and get typed inputs.

Before `tool_call` runs, harn waits for previously emitted Agent events to finish draining through `AgentSession`. This means `ctx.session_manager` is up to date through the current assistant tool-calling message.

In the default parallel tool execution mode, sibling tool calls from the same assistant message are preflighted sequentially, then executed concurrently. `tool_call` is not guaranteed to see sibling tool results from that same assistant message in `ctx.session_manager`.

`event.input` is mutable. Mutate it in place to patch tool arguments before execution.

Behavior guarantees:
- Mutations to `event.input` affect the actual tool execution
- Later `tool_call` handlers see mutations made by earlier handlers
- No re-validation is performed after your mutation
- Return values from `tool_call` only control blocking via `{"block": True, "reason": "..."}`

```python
from harn_coding_agent import is_tool_call_event_type

@harn.on("tool_call")
async def on_tool_call(event, ctx):
    # event.tool_name - "bash", "read", "write", "edit", etc.
    # event.tool_call_id
    # event.input - tool parameters (mutable dict)

    # Built-in tools: no type params needed
    if is_tool_call_event_type("bash", event):
        # event.input is {"command": str, "timeout": int | None}
        event.input["command"] = f"source ~/.profile\n{event.input['command']}"

        if "rm -rf" in event.input["command"]:
            return {"block": True, "reason": "Dangerous command"}

    if is_tool_call_event_type("read", event):
        # event.input is {"path": str, "offset": int | None, "limit": int | None}
        print(f"Reading: {event.input['path']}")
```

#### Typing custom tool input

Custom tools should define their input type:

```python
from pydantic import BaseModel

class MyToolInput(BaseModel):
    action: str
    text: str | None = None
```

Use `is_tool_call_event_type` with the tool name:

```python
from harn_coding_agent import is_tool_call_event_type

@harn.on("tool_call")
def on_tool_call(event, ctx):
    if is_tool_call_event_type("my_tool", event):
        event.input["action"]  # typed
```

#### tool_result

Fired after tool execution finishes and before `tool_execution_end` plus the final tool result message events are emitted. **Can modify result.**

In parallel tool mode, `tool_result` and `tool_execution_end` may interleave in tool completion order, while final `toolResult` message events are still emitted later in assistant source order.

`tool_result` handlers chain like middleware:
- Handlers run in extension load order
- Each handler sees the latest result after previous handler changes
- Handlers can return partial patches (`content`, `details`, or `is_error`); omitted fields keep their current values

Use `ctx.signal` for nested async work inside the handler. This lets Esc cancel model calls, HTTP requests, and other abort-aware operations started by the extension.

```python
import httpx
from harn_coding_agent import is_bash_tool_result

@harn.on("tool_result")
async def on_tool_result(event, ctx):
    # event.tool_name, event.tool_call_id, event.input
    # event.content, event.details, event.is_error

    if is_bash_tool_result(event):
        # event.details is typed as BashToolDetails
        pass

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://example.com/summarize",
            json={"content": event.content},
        )

    # Modify result:
    return {"content": [...], "details": {...}, "is_error": False}
```

### User Bash Events

#### user_bash

Fired when user executes `!` or `!!` commands. **Can intercept.**

```python
from harn_coding_agent import create_local_bash_operations

@harn.on("user_bash")
def on_user_bash(event, ctx):
    # event.command - the bash command
    # event.exclude_from_context - True if !! prefix
    # event.cwd - working directory

    # Option 1: Provide custom operations (e.g., SSH)
    return {"operations": remote_bash_ops}

    # Option 2: Wrap harn's built-in local bash backend
    local = create_local_bash_operations()
    return {
        "operations": {
            "exec": lambda command, cwd, options: local.exec(
                f"source ~/.profile\n{command}", cwd, options
            ),
        },
    }

    # Option 3: Full replacement - return result directly
    return {"result": {"output": "...", "exit_code": 0, "cancelled": False, "truncated": False}}
```

### Input Events

#### input

Fired when user input is received, after extension commands are checked but before skill and template expansion. The event sees the raw input text, so `/skill:foo` and `/template` are not yet expanded.

**Processing order:**
1. Extension commands (`/cmd`) checked first - if found, handler runs and input event is skipped
2. `input` event fires - can intercept, transform, or handle
3. If not handled: skill commands (`/skill:name`) expanded to skill content
4. If not handled: prompt templates (`/template`) expanded to template content
5. Agent processing begins (`before_agent_start`, etc.)

```python
@harn.on("input")
async def on_input(event, ctx):
    # event.text - raw input (before skill/template expansion)
    # event.images - attached images, if any
    # event.source - "interactive" (typed), "rpc" (API), or "extension" (via send_user_message)

    # Transform: rewrite input before expansion
    if event.text.startswith("?quick "):
        return {"action": "transform", "text": f"Respond briefly: {event.text[7:]}"}

    # Handle: respond without LLM (extension shows its own feedback)
    if event.text == "ping":
        ctx.ui.notify("pong", "info")
        return {"action": "handled"}

    # Route by source: skip processing for extension-injected messages
    if event.source == "extension":
        return {"action": "continue"}

    # Intercept skill commands before expansion
    if event.text.startswith("/skill:"):
        # Could transform, block, or let pass through
        pass

    return {"action": "continue"}  # Default: pass through to expansion
```

**Results:**
- `continue` - pass through unchanged (default if handler returns nothing)
- `transform` - modify text/images, then continue to expansion
- `handled` - skip agent entirely (first handler to return this wins)

Transforms chain across handlers. See [input_transform.py](../examples/extensions/input_transform.py).

## ExtensionContext

All handlers receive `ctx: ExtensionContext`.

### ctx.ui

UI methods for user interaction. See [Custom UI](#custom-ui) for full details.

### ctx.has_ui

`False` in print mode (`-p`) and JSON mode. `True` in interactive and RPC mode. In RPC mode, dialog methods (`select`, `confirm`, `input`, `editor`) work via the extension UI sub-protocol, and fire-and-forget methods (`notify`, `set_status`, `set_widget`, `set_title`) emit requests to the client. Some TUI-specific methods are no-ops or return defaults (see [rpc.md](rpc.md#extension-ui-protocol)).

### ctx.cwd

Current working directory.

### ctx.session_manager

Read-only access to session state. See [Session Format](session-format.md) for the full SessionManager API and entry types.

For `tool_call`, this state is synchronized through the current assistant message before handlers run. In parallel tool execution mode it is still not guaranteed to include sibling tool results from the same assistant message.

```python
ctx.session_manager.get_entries()       # All entries
ctx.session_manager.get_branch()        # Current branch
ctx.session_manager.get_leaf_id()       # Current leaf entry ID
```

### ctx.model_registry / ctx.model

Access to models and API keys.

### ctx.signal

The current agent abort signal, or `None` when no agent turn is active.

Use this for abort-aware nested work started by extension handlers, for example:
- HTTP requests with timeout/cancellation
- model calls that accept signals
- file or process helpers that accept cancellation tokens

`ctx.signal` is typically defined during active turn events such as `tool_call`, `tool_result`, `message_update`, and `turn_end`.
It is usually `None` in idle or non-turn contexts such as session events, extension commands, and shortcuts fired while harn is idle.

```python
@harn.on("tool_result")
async def on_tool_result(event, ctx):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://example.com/api",
            json=event,
        )
        data = response.json()

    return {"details": data}
```

### ctx.is_idle() / ctx.abort() / ctx.has_pending_messages()

Control flow helpers.

### ctx.shutdown()

Request a graceful shutdown of harn.

- **Interactive mode:** Deferred until the agent becomes idle (after processing all queued steering and follow-up messages).
- **RPC mode:** Deferred until the next idle state (after completing the current command response, when waiting for the next command).
- **Print mode:** No-op. The process exits automatically when all prompts are processed.

Emits `session_shutdown` event to all extensions before exiting. Available in all contexts (event handlers, tools, commands, shortcuts).

```python
@harn.on("tool_call")
def on_tool_call(event, ctx):
    if is_fatal(event.input):
        ctx.shutdown()
```

### ctx.get_context_usage()

Returns current context usage for the active model. Uses last assistant usage when available, then estimates tokens for trailing messages.

```python
usage = ctx.get_context_usage()
if usage and usage.tokens > 100_000:
    # ...
    pass
```

### ctx.compact()

Trigger compaction without awaiting completion. Use `on_complete` and `on_error` for follow-up actions.

```python
ctx.compact(
    custom_instructions="Focus on recent changes",
    on_complete=lambda result: ctx.ui.notify("Compaction completed", "info"),
    on_error=lambda error: ctx.ui.notify(f"Compaction failed: {error}", "error"),
)
```

### ctx.get_system_prompt()

Returns harn's current system prompt string.

- During `before_agent_start`, this reflects chained system-prompt changes made so far for the current turn.
- It does not include later `context` message mutations.
- It does not include `before_provider_request` payload rewrites.
- If later-loaded extensions run after yours, they can still change what is ultimately sent.

```python
@harn.on("before_agent_start")
def on_before_agent_start(event, ctx):
    prompt = ctx.get_system_prompt()
    print(f"System prompt length: {len(prompt)}")
```

## ExtensionCommandContext

Command handlers receive `ExtensionCommandContext`, which extends `ExtensionContext` with session control methods. These are only available in commands because they can deadlock if called from event handlers.

### ctx.wait_for_idle()

Wait for the agent to finish streaming:

```python
@harn.register_command("my-cmd")
async def my_cmd(args, ctx):
    await ctx.wait_for_idle()
    # Agent is now idle, safe to modify session
```

### ctx.new_session(options=None)

Create a new session:

```python
parent_session = ctx.session_manager.get_session_file()
kickoff = "Continue in the replacement session"

result = await ctx.new_session(
    parent_session=parent_session,
    setup=lambda sm: sm.append_message({
        "role": "user",
        "content": [{"type": "text", "text": "Context from previous session..."}],
        "timestamp": time.time_ns() // 1_000_000,
    }),
    with_session=lambda ctx: ctx.send_user_message(kickoff),
)

if result.cancelled:
    # An extension cancelled the new session
    pass
```

Options:
- `parent_session`: parent session file to record in the new session header
- `setup`: mutate the new session's `SessionManager` before `with_session` runs
- `with_session`: run post-switch work against a fresh replacement-session context. Do not use captured old `harn` / command `ctx`; see [Session replacement lifecycle and footguns](#session-replacement-lifecycle-and-footguns).

### ctx.fork(entry_id, options=None)

Fork from a specific entry, creating a new session file:

```python
result = await ctx.fork("entry-id-123",
    with_session=lambda ctx: ctx.ui.notify("Now in the forked session", "info"),
)
if result.cancelled:
    # An extension cancelled the fork
    pass

clone_result = await ctx.fork("entry-id-456", position="at")
if clone_result.cancelled:
    # An extension cancelled the clone
    pass
```

Options:
- `position`: `"before"` (default) forks before the selected user message, restoring that prompt into the editor
- `position`: `"at"` duplicates the active path through the selected entry without restoring editor text
- `with_session`: run post-switch work against a fresh replacement-session context. Do not use captured old `harn` / command `ctx`; see [Session replacement lifecycle and footguns](#session-replacement-lifecycle-and-footguns).

### ctx.navigate_tree(target_id, options=None)

Navigate to a different point in the session tree:

```python
result = await ctx.navigate_tree("entry-id-456",
    summarize=True,
    custom_instructions="Focus on error handling changes",
    replace_instructions=False,  # True = replace default prompt entirely
    label="review-checkpoint",
)
```

Options:
- `summarize`: Whether to generate a summary of the abandoned branch
- `custom_instructions`: Custom instructions for the summarizer
- `replace_instructions`: If True, `custom_instructions` replaces the default prompt instead of being appended
- `label`: Label to attach to the branch summary entry (or target entry if not summarizing)

### ctx.switch_session(session_path, options=None)

Switch to a different session file:

```python
result = await ctx.switch_session("/path/to/session.jsonl",
    with_session=lambda ctx: ctx.send_user_message("Resume work in the replacement session"),
)
if result.cancelled:
    # An extension cancelled the switch via session_before_switch
    pass
```

Options:
- `with_session`: run post-switch work against a fresh replacement-session context. Do not use captured old `harn` / command `ctx`; see [Session replacement lifecycle and footguns](#session-replacement-lifecycle-and-footguns).

To discover available sessions, use the static `SessionManager.list()` or `SessionManager.list_all()` methods:

```python
from harn_coding_agent import SessionManager

@harn.register_command("switch", description="Switch to another session")
async def switch(args, ctx):
    sessions = await SessionManager.list(ctx.cwd)
    if not sessions:
        return
    choice = await ctx.ui.select(
        "Pick session:",
        [s.file for s in sessions],
    )
    if choice:
        await ctx.switch_session(choice,
            with_session=lambda ctx: ctx.ui.notify("Switched session", "info"),
        )
```

### Session replacement lifecycle and footguns

`with_session` receives a fresh `ReplacedSessionContext`, which extends `ExtensionCommandContext` with async `send_message()` and `send_user_message()` helpers bound to the replacement session.

Lifecycle and footguns:
- `with_session` runs only after the old session has emitted `session_shutdown`, the old runtime has been torn down, the replacement session has been rebound, and the new extension instance has already received `session_start`.
- The callback still executes in the original closure, not inside the new extension instance. That means your old extension instance may already have run its shutdown cleanup before `with_session` starts.
- Captured old `harn` / old command `ctx` session-bound objects are stale after replacement and will throw if used. Use only the `ctx` passed to `with_session` for session-bound work.
- Previously extracted raw objects are still your responsibility. For example, if you capture `sm = ctx.session_manager` before replacement, `sm` is still the old `SessionManager` object. Do not reuse it after replacement.
- Code in `with_session` should assume any state invalidated by your `session_shutdown` handler is already gone. Only capture plain data that survives shutdown cleanly, such as strings, ids, and serialized config.

Safe pattern:

```python
@harn.register_command("handoff")
async def handoff(args, ctx):
    kickoff = "Continue from the replacement session"
    await ctx.new_session(
        with_session=lambda ctx: ctx.send_user_message(kickoff),
    )
```

Unsafe pattern:

```python
@harn.register_command("handoff")
async def handoff(args, ctx):
    old_session_manager = ctx.session_manager
    await ctx.new_session(
        with_session=lambda _ctx: (
            # stale old objects: do not do this
            old_session_manager.get_session_file(),
            harn.send_user_message("wrong"),
        ),
    )
```

### ctx.reload()

Run the same reload flow as `/reload`.

```python
@harn.register_command("reload-runtime", description="Reload extensions, skills, prompts, and themes")
async def reload_runtime(args, ctx):
    await ctx.reload()
    return
```

Important behavior:
- `await ctx.reload()` emits `session_shutdown` for the current extension runtime
- It then reloads resources and emits `session_start` with `reason: "reload"` and `resources_discover` with reason `"reload"`
- The currently running command handler still continues in the old call frame
- Code after `await ctx.reload()` still runs from the pre-reload version
- Code after `await ctx.reload()` must not assume old in-memory extension state is still valid
- After the handler returns, future commands/events/tool calls use the new extension version

For predictable behavior, treat reload as terminal for that handler (`await ctx.reload(); return`).

Tools run with `ExtensionContext`, so they cannot call `ctx.reload()` directly. Use a command as the reload entrypoint, then expose a tool that queues that command as a follow-up user message.

Example tool the LLM can call to trigger reload:

```python
from harn_coding_agent import ExtensionAPI
from pydantic import BaseModel


def extension_factory(harn: ExtensionAPI):
    @harn.register_command("reload-runtime", description="Reload extensions, skills, prompts, and themes")
    async def reload_runtime(args, ctx):
        await ctx.reload()
        return

    class EmptyParams(BaseModel):
        pass

    @harn.register_tool(
        name="reload_runtime",
        label="Reload Runtime",
        description="Reload extensions, skills, prompts, and themes",
        parameters=EmptyParams,
    )
    async def reload_runtime_tool(tool_call_id, params, signal, on_update, ctx):
        harn.send_user_message("/reload-runtime", deliver_as="follow_up")
        return {
            "content": [{"type": "text", "text": "Queued /reload-runtime as a follow-up command."}],
        }
```

## ExtensionAPI Methods

### harn.on(event, handler)

Subscribe to events. See [Events](#events) for event types and return values.

### harn.register_tool(definition)

Register a custom tool callable by the LLM. See [Custom Tools](#custom-tools) for full details.

`harn.register_tool()` works both during extension load and after startup. You can call it inside `session_start`, command handlers, or other event handlers. New tools are refreshed immediately in the same session, so they appear in `harn.get_all_tools()` and are callable by the LLM without `/reload`.

Use `harn.set_active_tools()` to enable or disable tools (including dynamically added tools) at runtime.

Use `prompt_snippet` to opt a custom tool into a one-line entry in `Available tools`, and `prompt_guidelines` to append tool-specific bullets to the default `Guidelines` section when the tool is active.

**Important:** `prompt_guidelines` bullets are appended flat to the `Guidelines` section with no tool name prefix. Each guideline must name the tool it refers to -- avoid "Use this tool when..." because the LLM cannot tell which tool "this" means. Write "Use my_tool when..." instead.

See [dynamic_tools.py](../examples/extensions/dynamic_tools.py) for a full example.

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from harn_ai import StringEnum

class MyToolParams(BaseModel):
    action: Literal["list", "add"] = Field(description="Action to perform")
    text: Optional[str] = None

@harn.register_tool(
    name="my_tool",
    label="My Tool",
    description="What this tool does",
    prompt_snippet="Summarize or transform text according to action",
    prompt_guidelines=["Use my_tool when the user asks to summarize previously generated text."],
    parameters=MyToolParams,
)
async def my_tool(tool_call_id, params, signal, on_update, ctx):
    # Stream progress
    if on_update:
        on_update({"content": [{"type": "text", "text": "Working..."}]})

    return {
        "content": [{"type": "text", "text": "Done"}],
        "details": {"result": "..."},
    }
```

### harn.send_message(message, options=None)

Inject a custom message into the session.

```python
harn.send_message(
    {
        "custom_type": "my-extension",
        "content": "Message text",
        "display": True,
        "details": {...},
    },
    trigger_turn=True,
    deliver_as="steer",
)
```

**Options:**
- `deliver_as` - Delivery mode:
  - `"steer"` (default) - Queues the message while streaming. Delivered after the current assistant turn finishes executing its tool calls, before the next LLM call.
  - `"follow_up"` - Waits for agent to finish. Delivered only when agent has no more tool calls.
  - `"next_turn"` - Queued for next user prompt. Does not interrupt or trigger anything.
- `trigger_turn=True` - If agent is idle, trigger an LLM response immediately. Only applies to `"steer"` and `"follow_up"` modes (ignored for `"next_turn"`).

### harn.send_user_message(content, options=None)

Send a user message to the agent. Unlike `send_message()` which sends custom messages, this sends an actual user message that appears as if typed by the user. Always triggers a turn.

```python
# Simple text message
harn.send_user_message("What is 2+2?")

# With content list (text + images)
harn.send_user_message([
    {"type": "text", "text": "Describe this image:"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
])

# During streaming - must specify delivery mode
harn.send_user_message("Focus on error handling", deliver_as="steer")
harn.send_user_message("And then summarize", deliver_as="follow_up")
```

**Options:**
- `deliver_as` - Required when agent is streaming:
  - `"steer"` - Queues the message for delivery after the current assistant turn finishes executing its tool calls
  - `"follow_up"` - Waits for agent to finish all tools

When not streaming, the message is sent immediately and triggers a new turn. When streaming without `deliver_as`, raises an error.

See [send_user_message.py](../examples/extensions/send_user_message.py) for a complete example.

### harn.append_entry(custom_type, data=None)

Persist extension state (does NOT participate in LLM context).

```python
harn.append_entry("my-state", {"count": 42})

# Restore on reload
@harn.on("session_start")
async def on_session_start(event, ctx):
    for entry in ctx.session_manager.get_entries():
        if entry.type == "custom" and entry.custom_type == "my-state":
            # Reconstruct from entry.data
            pass
```

### harn.set_session_name(name)

Set the session display name (shown in session selector instead of first message).

```python
harn.set_session_name("Refactor auth module")
```

### harn.get_session_name()

Get the current session name, if set.

```python
name = harn.get_session_name()
if name:
    print(f"Session: {name}")
```

### harn.set_label(entry_id, label)

Set or clear a label on an entry. Labels are user-defined markers for bookmarking and navigation (shown in `/tree` selector).

```python
# Set a label
harn.set_label(entry_id, "checkpoint-before-refactor")

# Clear a label
harn.set_label(entry_id, None)

# Read labels via session_manager
label = ctx.session_manager.get_label(entry_id)
```

Labels persist in the session and survive restarts. Use them to mark important points (turns, checkpoints) in the conversation tree.

### harn.register_command(name, options)

Register a command.

If multiple extensions register the same command name, harn keeps them all and assigns numeric invocation suffixes in load order, for example `/review:1` and `/review:2`.

```python
@harn.register_command("stats", description="Show session statistics")
async def stats(args, ctx):
    count = len(ctx.session_manager.get_entries())
    ctx.ui.notify(f"{count} entries", "info")
```

Optional: add argument auto-completion for `/command ...`:

```python
def get_deploy_completions(prefix: str):
    envs = ["dev", "staging", "prod"]
    items = [{"value": e, "label": e} for e in envs]
    filtered = [i for i in items if i["value"].startswith(prefix)]
    return filtered if filtered else None


@harn.register_command(
    "deploy",
    description="Deploy to an environment",
    get_argument_completions=get_deploy_completions,
)
async def deploy(args, ctx):
    ctx.ui.notify(f"Deploying: {args}", "info")
```

### harn.get_commands()

Get the slash commands available for invocation via `prompt` in the current session. Includes extension commands, prompt templates, and skill commands.
The list matches the RPC `get_commands` ordering: extensions first, then templates, then skills.

```python
commands = harn.get_commands()
by_source = [cmd for cmd in commands if cmd.source == "extension"]
user_scoped = [cmd for cmd in commands if cmd.source_info.scope == "user"]
```

Each entry has this shape:

```python
{
    "name": str,           # Invokable command name without the leading slash. May be suffixed like "review:1"
    "description": str,    # Optional
    "source": str,         # "extension" | "prompt" | "skill"
    "source_info": {
        "path": str,
        "source": str,
        "scope": str,      # "user" | "project" | "temporary"
        "origin": str,     # "package" | "top-level"
        "base_dir": str,   # Optional
    },
}
```

Use `source_info` as the canonical provenance field. Do not infer ownership from command names or from ad hoc path parsing.

Built-in interactive commands (like `/model` and `/settings`) are not included here. They are handled only in interactive mode and would not execute if sent via `prompt`.

### harn.register_message_renderer(custom_type, renderer)

Register a custom TUI renderer for messages with your `custom_type`. See [Custom UI](#custom-ui).

### harn.register_shortcut(shortcut, options)

Register a keyboard shortcut. See [keybindings.md](keybindings.md) for the shortcut format and built-in keybindings.

```python
@harn.register_shortcut("ctrl+shift+p", description="Toggle plan mode")
async def toggle_plan(ctx):
    ctx.ui.notify("Toggled!")
```

### harn.register_flag(name, options)

Register a CLI flag.

```python
harn.register_flag("plan",
    description="Start in plan mode",
    type="boolean",
    default=False,
)

# Check value
if harn.get_flag("plan"):
    # Plan mode enabled
    pass
```

### harn.exec(command, args, options=None)

Execute a shell command.

```python
result = await harn.exec("git", ["status"], signal=signal, timeout=5000)
# result.stdout, result.stderr, result.code, result.killed
```

### harn.get_active_tools() / harn.get_all_tools() / harn.set_active_tools(names)

Manage active tools. This works for both built-in tools and dynamically registered tools.

```python
active = harn.get_active_tools()
all_tools = harn.get_all_tools()
# [{"name": "read", "description": "Read file contents...", "parameters": ...,
#   "source_info": {"path": "<builtin:read>", "source": "builtin", "scope": "temporary", "origin": "top-level"}
# }, ...]
names = [t["name"] for t in all_tools]
builtin_tools = [t for t in all_tools if t["source_info"]["source"] == "builtin"]
extension_tools = [t for t in all_tools if t["source_info"]["source"] not in ("builtin", "sdk")]
harn.set_active_tools(["read", "bash"])  # Switch to read-only
```

`harn.get_all_tools()` returns `name`, `description`, `parameters`, and `source_info`.

Typical `source_info["source"]` values:
- `builtin` for built-in tools
- `sdk` for tools passed via `create_agent_session(custom_tools=...)`
- extension source metadata for tools registered by extensions

### harn.set_model(model)

Set the current model. Returns `False` if no API key is available for the model. See [models.md](models.md) for configuring custom models.

```python
model = ctx.model_registry.find("anthropic", "claude-sonnet-4-5")
if model:
    success = await harn.set_model(model)
    if not success:
        ctx.ui.notify("No API key for this model", "error")
```

### harn.get_thinking_level() / harn.set_thinking_level(level)

Get or set the thinking level. Level is clamped to model capabilities (non-reasoning models always use "off"). Changes emit `thinking_level_select`.

```python
current = harn.get_thinking_level()  # "off" | "minimal" | "low" | "medium" | "high" | "xhigh"
harn.set_thinking_level("high")
```

### harn.events

Shared event bus for communication between extensions:

```python
harn.events.on("my:event", lambda data: ...)
harn.events.emit("my:event", {...})
```

### harn.register_provider(name, config)

Register or override a model provider dynamically. Useful for proxies, custom endpoints, or team-wide model configurations.

Calls made during the extension factory function are queued and applied once the runner initialises. Calls made after that -- for example from a command handler following a user setup flow -- take effect immediately without requiring a `/reload`.

If you need to discover models from a remote endpoint, prefer an async extension factory over deferring the fetch to `session_start`. harn waits for the factory before startup continues, so the registered models are available immediately, including to `harn --list-models`.

```python
# Register a new provider with custom models
harn.register_provider("my-proxy", {
    "name": "My Proxy",
    "base_url": "https://proxy.example.com",
    "api_key": "PROXY_API_KEY",  # env var name or literal
    "api": "anthropic-messages",
    "models": [
        {
            "id": "claude-sonnet-4-20250514",
            "name": "Claude 4 Sonnet (proxy)",
            "reasoning": False,
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            "context_window": 200000,
            "max_tokens": 16384,
        },
    ],
})

# Override base_url for an existing provider (keeps all models)
harn.register_provider("anthropic", {
    "base_url": "https://proxy.example.com",
})

# Register provider with OAuth support for /login
harn.register_provider("corporate-ai", {
    "base_url": "https://ai.corp.com",
    "api": "openai-responses",
    "models": [...],
    "oauth": {
        "name": "Corporate AI (SSO)",
        "login": my_login_handler,
        "refresh_token": my_refresh_handler,
        "get_api_key": lambda credentials: credentials["access"],
    },
})
```

**Config options:**
- `name` - Display name for the provider in UI such as `/login`.
- `base_url` - API endpoint URL. Required when defining models.
- `api_key` - API key or environment variable name. Required when defining models (unless `oauth` provided).
- `api` - API type: `"anthropic-messages"`, `"openai-completions"`, `"openai-responses"`, etc.
- `headers` - Custom headers to include in requests.
- `auth_header` - If True, adds `Authorization: Bearer` header automatically.
- `models` - List of model definitions. If provided, replaces all existing models for this provider. Model definitions can set `base_url` to override the provider endpoint for that model.
- `oauth` - OAuth provider config for `/login` support. When provided, the provider appears in the login menu.
- `stream_simple` - Custom streaming implementation for non-standard APIs.

See [custom-provider.md](custom-provider.md) for advanced topics: custom streaming APIs, OAuth details, model definition reference.

### harn.unregister_provider(name)

Remove a previously registered provider and its models. Built-in models that were overridden by the provider are restored. Has no effect if the provider was not registered.

Like `register_provider`, this takes effect immediately when called after the initial load phase, so a `/reload` is not required.

```python
@harn.register_command("my-setup-teardown", description="Remove the custom proxy provider")
async def my_setup_teardown(args, ctx):
    harn.unregister_provider("my-proxy")
```

## State Management

Extensions with state should store it in tool result `details` for proper branching support:

```python
from harn_coding_agent import ExtensionAPI


def extension_factory(harn: ExtensionAPI):
    items: list[str] = []

    # Reconstruct state from session
    @harn.on("session_start")
    async def on_session_start(event, ctx):
        nonlocal items
        items = []
        for entry in ctx.session_manager.get_branch():
            if entry.type == "message" and entry.message.role == "toolResult":
                if entry.message.tool_name == "my_tool":
                    items = entry.message.details.get("items", [])

    @harn.register_tool(
        name="my_tool",
        # ...
    )
    async def my_tool(tool_call_id, params, signal, on_update, ctx):
        nonlocal items
        items.append("new item")
        return {
            "content": [{"type": "text", "text": "Added"}],
            "details": {"items": list(items)},  # Store for reconstruction
        }
```

## Custom Tools

Register tools the LLM can call via `harn.register_tool()`. Tools appear in the system prompt and can have custom rendering.

Use `prompt_snippet` for a short one-line entry in the `Available tools` section in the default system prompt. If omitted, custom tools are left out of that section.

Use `prompt_guidelines` to add tool-specific bullets to the default system prompt `Guidelines` section. These bullets are included only while the tool is active (for example, after `harn.set_active_tools([...])`).

**Important:** `prompt_guidelines` bullets are appended flat to the `Guidelines` section with no tool name prefix or grouping. Each guideline must name the tool it refers to -- avoid "Use this tool when..." because the LLM cannot tell which tool "this" means. Write "Use my_tool when..." instead.

Note: Some models include the @ prefix in tool path arguments. Built-in tools strip a leading @ before resolving paths. If your custom tool accepts a path, normalize a leading @ as well.

If your custom tool mutates files, use `with_file_mutation_queue()` so it participates in the same per-file queue as built-in `edit` and `write`. This matters because tool calls run in parallel by default. Without the queue, two tools can read the same old file contents, compute different updates, and then whichever write lands last overwrites the other.

Example failure case: your custom tool edits `foo.py` while built-in `edit` also changes `foo.py` in the same assistant turn. If your tool does not participate in the queue, both can read the original `foo.py`, apply separate changes, and one of those changes is lost.

Pass the real target file path to `with_file_mutation_queue()`, not the raw user argument. Resolve it to an absolute path first, relative to `ctx.cwd` or your tool's working directory. For existing files, the helper canonicalizes through `os.path.realpath()`, so symlink aliases for the same file share one queue. For new files, it falls back to the resolved absolute path because there is nothing to resolve yet.

Queue the entire mutation window on that target path. That includes read-modify-write logic, not just the final write.

```python
from harn_coding_agent import with_file_mutation_queue
from pathlib import Path


async def execute(tool_call_id, params, signal, on_update, ctx):
    absolute_path = str(Path(ctx.cwd) / params.path)

    async def mutate():
        path = Path(absolute_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        current = path.read_text(encoding="utf-8")
        next_content = current.replace(params.old_text, params.new_text)
        path.write_text(next_content, encoding="utf-8")

        return {
            "content": [{"type": "text", "text": f"Updated {params.path}"}],
            "details": {},
        }

    return await with_file_mutation_queue(absolute_path, mutate)
```

### Tool Definition

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from harn_ai import StringEnum
from harn_tui import Text

class MyToolParams(BaseModel):
    action: Literal["list", "add"] = Field(description="Action type")  # Use Literal for Google compatibility
    text: Optional[str] = None

@harn.register_tool(
    name="my_tool",
    label="My Tool",
    description="What this tool does (shown to LLM)",
    prompt_snippet="List or add items in the project todo list",
    prompt_guidelines=[
        "Use my_tool for todo planning instead of direct file edits when the user asks for a task list."
    ],
    parameters=MyToolParams,
)
async def my_tool(tool_call_id, params, signal, on_update, ctx):
    # Check for cancellation
    if signal and signal.is_set():
        return {"content": [{"type": "text", "text": "Cancelled"}]}

    # Stream progress updates
    if on_update:
        on_update({
            "content": [{"type": "text", "text": "Working..."}],
            "details": {"progress": 50},
        })

    # Run commands via harn.exec (captured from extension closure)
    result = await harn.exec("some-command", [], signal=signal)

    # Return result
    return {
        "content": [{"type": "text", "text": "Done"}],   # Sent to LLM
        "details": {"data": result},                       # For rendering & state
        # Optional: stop after this tool batch when every finalized tool result
        # in the batch also returns terminate: True.
        "terminate": True,
    }
```

**Signaling errors:** To mark a tool execution as failed (sets `is_error: True` on the result and reports it to the LLM), raise an exception from `execute`. Returning a value never sets the error flag regardless of what properties you include in the return object.

**Early termination:** Return `terminate: True` from `execute()` to hint that the automatic follow-up LLM call should be skipped after the current tool batch. This only takes effect when every finalized tool result in that batch is terminating. See [examples/extensions/structured_output.py](../examples/extensions/structured_output.py) for a minimal example where the agent ends on a final structured-output tool call.

```python
# Correct: raise to signal an error
async def execute(tool_call_id, params, signal, on_update, ctx):
    if not is_valid(params.input):
        raise ValueError(f"Invalid input: {params.input}")
    return {"content": [{"type": "text", "text": "OK"}], "details": {}}
```

**Important:** Use `Literal` types or `StringEnum` from `harn_ai` for string enums. Standard Python `Enum` may not work with all provider APIs.

**Argument preparation:** `prepare_arguments(args)` is optional. If defined, it runs before schema validation and before `execute()`. Use it to mimic an older accepted input shape when harn resumes an older session whose stored tool call arguments no longer match the current schema. Return the dict you want validated against `parameters`. Keep the public schema strict. Do not add deprecated compatibility fields to `parameters` just to keep old resumed sessions working.

### Overriding Built-in Tools

Extensions can override built-in tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) by registering a tool with the same name. Interactive mode displays a warning when this happens.

```bash
# Extension's read tool replaces built-in read
harn -e ./tool_override.py
```

Alternatively, use `--no-builtin-tools` to start without any built-in tools while keeping extension tools enabled:
```bash
# No built-in tools, only extension tools
harn --no-builtin-tools -e ./my_extension.py
```

See [examples/extensions/tool_override.py](../examples/extensions/tool_override.py) for a complete example that overrides `read` with logging and access control.

**Rendering:** Built-in renderer inheritance is resolved per slot. Execution override and rendering override are independent. If your override omits `render_call`, the built-in `render_call` is used. If your override omits `render_result`, the built-in `render_result` is used. If your override omits both, the built-in renderer is used automatically (syntax highlighting, diffs, etc.). This lets you wrap built-in tools for logging or access control without reimplementing the UI.

**Prompt metadata:** `prompt_snippet` and `prompt_guidelines` are not inherited from the built-in tool. If your override should keep those prompt instructions, define them on the override explicitly.

**Your implementation must match the exact result shape**, including the `details` type. The UI and session logic depend on these shapes for rendering and state tracking.

Built-in tool implementations:
- [read.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/read.py) - `ReadToolDetails`
- [bash.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/bash.py) - `BashToolDetails`
- [edit.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/edit.py)
- [write.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/write.py)
- [grep.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/grep.py) - `GrepToolDetails`
- [find.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/find.py) - `FindToolDetails`
- [ls.py](https://github.com/secemp9/harn/blob/main/packages/harn_coding_agent/src/harn_coding_agent/core/tools/ls.py) - `LsToolDetails`

### Remote Execution

Built-in tools support pluggable operations for delegating to remote systems (SSH, containers, etc.):

```python
from harn_coding_agent import create_read_tool, create_bash_tool

# Create tool with custom operations
remote_read = create_read_tool(cwd, operations={
    "read_file": lambda path: ssh_exec(remote, f"cat {path}"),
    "access": lambda path: ssh_exec(remote, f"test -r {path}"),
})

# Register, checking flag at execution time
@harn.register_tool(**{**remote_read, "name": "read"})
async def read_tool(tool_call_id, params, signal, on_update, ctx):
    ssh = get_ssh_config()
    if ssh:
        tool = create_read_tool(cwd, operations=create_remote_ops(ssh))
        return await tool.execute(tool_call_id, params, signal, on_update)
    return await local_read.execute(tool_call_id, params, signal, on_update)
```

**Operations interfaces:** `ReadOperations`, `WriteOperations`, `EditOperations`, `BashOperations`, `LsOperations`, `GrepOperations`, `FindOperations`

For `user_bash`, extensions can reuse harn's local shell backend via `create_local_bash_operations()` instead of reimplementing local process spawning, shell resolution, and process-tree termination.

The bash tool also supports a spawn hook to adjust the command, cwd, or env before execution:

```python
from harn_coding_agent import create_bash_tool

bash_tool = create_bash_tool(cwd, spawn_hook=lambda command, cwd, env: {
    "command": f"source ~/.profile\n{command}",
    "cwd": f"/mnt/sandbox{cwd}",
    "env": {**env, "CI": "1"},
})
```

See [examples/extensions/ssh.py](../examples/extensions/ssh.py) for a complete SSH example with `--ssh` flag.

### Output Truncation

**Tools MUST truncate their output** to avoid overwhelming the LLM context. Large outputs can cause:
- Context overflow errors (prompt too long)
- Compaction failures
- Degraded model performance

The built-in limit is **50KB** (~10k tokens) and **2000 lines**, whichever is hit first. Use the exported truncation utilities:

```python
from harn_coding_agent import (
    truncate_head,       # Keep first N lines/bytes (good for file reads, search results)
    truncate_tail,       # Keep last N lines/bytes (good for logs, command output)
    truncate_line,       # Truncate a single line to max_bytes with ellipsis
    format_size,         # Human-readable size (e.g., "50KB", "1.5MB")
    DEFAULT_MAX_BYTES,   # 50KB
    DEFAULT_MAX_LINES,   # 2000
)


async def execute(tool_call_id, params, signal, on_update, ctx):
    output = await run_command()

    # Apply truncation
    truncation = truncate_head(output,
        max_lines=DEFAULT_MAX_LINES,
        max_bytes=DEFAULT_MAX_BYTES,
    )

    result = truncation.content

    if truncation.truncated:
        # Write full output to temp file
        temp_file = write_temp_file(output)

        # Inform the LLM where to find complete output
        result += f"\n\n[Output truncated: {truncation.output_lines} of {truncation.total_lines} lines"
        result += f" ({format_size(truncation.output_bytes)} of {format_size(truncation.total_bytes)})."
        result += f" Full output saved to: {temp_file}]"

    return {"content": [{"type": "text", "text": result}]}
```

**Key points:**
- Use `truncate_head` for content where the beginning matters (search results, file reads)
- Use `truncate_tail` for content where the end matters (logs, command output)
- Always inform the LLM when output is truncated and where to find the full version
- Document the truncation limits in your tool's description

See [examples/extensions/truncated_tool.py](../examples/extensions/truncated_tool.py) for a complete example wrapping `rg` (ripgrep) with proper truncation.

### Multiple Tools

One extension can register multiple tools with shared state:

```python
def extension_factory(harn: ExtensionAPI):
    connection = None

    harn.register_tool(name="db_connect", ...)
    harn.register_tool(name="db_query", ...)
    harn.register_tool(name="db_close", ...)

    @harn.on("session_shutdown")
    async def on_shutdown(event, ctx):
        nonlocal connection
        if connection:
            connection.close()
```

### Custom Rendering

Tools can provide `render_call` and `render_result` for custom TUI display. See [tui.md](tui.md) for the full component API.

By default, tool output is wrapped in a `Box` that handles padding and background. A defined `render_call` or `render_result` must return a `Component`. If a slot renderer is not defined, fallback rendering is used for that slot.

Set `render_shell="self"` when the tool should render its own shell instead of using the default `Box`. This is useful for tools that need complete control over framing or background behavior.

```python
from harn_tui import Text

@harn.register_tool(
    name="my_tool",
    label="My Tool",
    description="Custom shell example",
    parameters=EmptyParams,
    render_shell="self",
)
async def my_tool(tool_call_id, params, signal, on_update, ctx):
    return {"content": [{"type": "text", "text": "ok"}], "details": None}
```

`render_call` and `render_result` each receive a `context` object with:
- `args` - the current tool call arguments
- `state` - shared row-local state across `render_call` and `render_result`
- `last_component` - the previously returned component for that slot, if any
- `invalidate()` - request a rerender of this tool row
- `tool_call_id`, `cwd`, `execution_started`, `args_complete`, `is_partial`, `expanded`, `show_images`, `is_error`

Use `context.state` for cross-slot shared state. Keep slot-local caches on the returned component instance when you want to reuse and mutate the same component across renders.

#### Best Practices

- Use `Text` with padding `(0, 0)`. The default Box handles padding.
- Use `\n` for multi-line content.
- Handle `is_partial` for streaming progress.
- Support `expanded` for detail on demand.
- Keep default view compact.
- Read `context.args` in `render_result` instead of copying args into `context.state`.
- Use `context.state` only for data that must be shared across call and result slots.
- Reuse `context.last_component` when the same component instance can be updated in place.
- Use `render_shell="self"` only when the default boxed shell gets in the way.

#### Fallback

If a slot renderer is not defined or raises:
- `render_call`: Shows the tool name
- `render_result`: Shows raw text from `content`

## Custom UI

Extensions can interact with users via `ctx.ui` methods and customize how messages/tools render.

**For custom components, see [tui.md](tui.md)** which has copy-paste patterns for:
- Selection dialogs (SelectList)
- Async operations with cancel (BorderedLoader)
- Settings toggles (SettingsList)
- Status indicators (set_status)
- Working message, visibility, and indicator during streaming (`set_working_message`, `set_working_visible`, `set_working_indicator`)
- Widgets above/below editor (set_widget)
- Autocomplete providers layered on top of built-in slash/path completion (add_autocomplete_provider)
- Custom footers (set_footer)

### Dialogs

```python
# Select from options
choice = await ctx.ui.select("Pick one:", ["A", "B", "C"])

# Confirm dialog
ok = await ctx.ui.confirm("Delete?", "This cannot be undone")

# Text input
name = await ctx.ui.input("Name:", "placeholder")

# Multi-line editor
text = await ctx.ui.editor("Edit:", "prefilled text")

# Notification (non-blocking)
ctx.ui.notify("Done!", "info")  # "info" | "warning" | "error"
```

#### Timed Dialogs with Countdown

Dialogs support a `timeout` option that auto-dismisses with a live countdown display:

```python
# Dialog shows "Title (5s)" -> "Title (4s)" -> ... -> auto-dismisses at 0
confirmed = await ctx.ui.confirm(
    "Timed Confirmation",
    "This dialog will auto-cancel in 5 seconds. Confirm?",
    timeout=5000,
)

if confirmed:
    # User confirmed
    pass
else:
    # User cancelled or timed out
    pass
```

**Return values on timeout:**
- `select()` returns `None`
- `confirm()` returns `False`
- `input()` returns `None`

### Widgets, Status, and Footer

```python
# Status in footer (persistent until cleared)
ctx.ui.set_status("my-ext", "Processing...")
ctx.ui.set_status("my-ext", None)  # Clear

# Working loader (shown during streaming)
ctx.ui.set_working_message("Thinking deeply...")
ctx.ui.set_working_message()  # Restore default
ctx.ui.set_working_visible(False)  # Hide the built-in working loader row entirely
ctx.ui.set_working_visible(True)   # Show the built-in working loader row

# Widget above editor (default)
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"])
# Widget below editor
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"], placement="below_editor")
ctx.ui.set_widget("my-widget", None)  # Clear

# Terminal title
ctx.ui.set_title("harn - my-project")

# Editor text
ctx.ui.set_editor_text("Prefill text")
current = ctx.ui.get_editor_text()

# Paste into editor (triggers paste handling, including collapse for large content)
ctx.ui.paste_to_editor("pasted content")

# Tool output expansion
was_expanded = ctx.ui.get_tools_expanded()
ctx.ui.set_tools_expanded(True)
ctx.ui.set_tools_expanded(was_expanded)
```

### Custom Components

For complex UI, use `ctx.ui.custom()`. This temporarily replaces the editor with your component until `done()` is called:

```python
from harn_tui import Text, Component

result = await ctx.ui.custom(lambda tui, theme, keybindings, done: (
    # Return a component; call done(value) to close
    Text("Press Enter to confirm, Escape to cancel", 1, 1)
))

if result:
    # User pressed Enter
    pass
```

The callback receives:
- `tui` - TUI instance (for screen dimensions, focus management)
- `theme` - Current theme for styling
- `keybindings` - App keybinding manager (for checking shortcuts)
- `done(value)` - Call to close component and return value

See [tui.md](tui.md) for the full component API.

### Message Rendering

Register a custom renderer for messages with your `custom_type`:

```python
from harn_tui import Text

@harn.register_message_renderer("my-extension")
def render_my_message(message, options, theme):
    expanded = options.get("expanded", False)
    text = theme.fg("accent", f"[{message.custom_type}] ")
    text += message.content

    if expanded and message.details:
        import json
        text += "\n" + theme.fg("dim", json.dumps(message.details, indent=2))

    return Text(text, 0, 0)
```

Messages are sent via `harn.send_message()`:

```python
harn.send_message({
    "custom_type": "my-extension",  # Matches register_message_renderer
    "content": "Status update",
    "display": True,                # Show in TUI
    "details": {...},               # Available in renderer
})
```

### Theme Colors

All render functions receive a `theme` object. See [themes.md](themes.md) for creating custom themes and the full color palette.

```python
# Foreground colors
theme.fg("toolTitle", text)   # Tool names
theme.fg("accent", text)      # Highlights
theme.fg("success", text)     # Success (green)
theme.fg("error", text)       # Errors (red)
theme.fg("warning", text)     # Warnings (yellow)
theme.fg("muted", text)       # Secondary text
theme.fg("dim", text)         # Tertiary text

# Text styles
theme.bold(text)
theme.italic(text)
theme.strikethrough(text)
```

## Error Handling

- Extension errors are logged, agent continues
- `tool_call` errors block the tool (fail-safe)
- Tool `execute` errors must be signaled by raising; the raised exception is caught, reported to the LLM with `is_error: True`, and execution continues

## Mode Behavior

| Mode | UI Methods | Notes |
|------|-----------|-------|
| Interactive | Full TUI | Normal operation |
| RPC (`--mode rpc`) | JSON protocol | Host handles UI, see [rpc.md](rpc.md) |
| JSON (`--mode json`) | No-op | Event stream to stdout, see [json.md](json.md) |
| Print (`-p`) | No-op | Extensions run but can't prompt |

In non-interactive modes, check `ctx.has_ui` before using UI methods.

## Examples Reference

All examples in [examples/extensions/](../examples/extensions/).

| Example | Description | Key APIs |
|---------|-------------|----------|
| **Tools** |||
| `hello.py` | Minimal tool registration | `register_tool` |
| `question.py` | Tool with user interaction | `register_tool`, `ui.select` |
| `questionnaire.py` | Multi-step wizard tool | `register_tool`, `ui.custom` |
| `todo.py` | Stateful tool with persistence | `register_tool`, `append_entry`, `render_result`, session events |
| `dynamic_tools.py` | Register tools after startup and during commands | `register_tool`, `session_start`, `register_command` |
| `structured_output.py` | Final structured-output tool with `terminate: True` | `register_tool`, terminating tool results |
| `truncated_tool.py` | Output truncation example | `register_tool`, `truncate_head` |
| `tool_override.py` | Override built-in read tool | `register_tool` (same name as built-in) |
| **Commands** |||
| `pirate.py` | Modify system prompt per-turn | `register_command`, `before_agent_start` |
| `summarize.py` | Conversation summary command | `register_command`, `ui.custom` |
| `handoff.py` | Cross-provider model handoff | `register_command`, `ui.editor`, `ui.custom` |
| `qna.py` | Q&A with custom UI | `register_command`, `ui.custom`, `set_editor_text` |
| `send_user_message.py` | Inject user messages | `register_command`, `send_user_message` |
| `reload_runtime.py` | Reload command and LLM tool handoff | `register_command`, `ctx.reload()`, `send_user_message` |
| `shutdown_command.py` | Graceful shutdown command | `register_command`, `shutdown()` |
| **Events & Gates** |||
| `permission_gate.py` | Block dangerous commands | `on("tool_call")`, `ui.confirm` |
| `protected_paths.py` | Block writes to specific paths | `on("tool_call")` |
| `confirm_destructive.py` | Confirm session changes | `on("session_before_switch")`, `on("session_before_fork")` |
| `dirty_repo_guard.py` | Warn on dirty git repo | `on("session_before_*")`, `exec` |
| `input_transform.py` | Transform user input | `on("input")` |
| `model_status.py` | React to model changes | `on("model_select")`, `set_status` |
| `provider_payload.py` | Inspect payloads and provider response headers | `on("before_provider_request")`, `on("after_provider_response")` |
| `system_prompt_header.py` | Display system prompt info | `on("agent_start")`, `get_system_prompt` |
| `claude_rules.py` | Load rules from files | `on("session_start")`, `on("before_agent_start")` |
| `prompt_customizer.py` | Add context-aware tool guidance using `system_prompt_options` | `on("before_agent_start")`, `BuildSystemPromptOptions` |
| `file_trigger.py` | File watcher triggers messages | `send_message` |
| **Compaction & Sessions** |||
| `custom_compaction.py` | Custom compaction summary | `on("session_before_compact")` |
| `trigger_compact.py` | Trigger compaction manually | `compact()` |
| `git_checkpoint.py` | Git stash on turns | `on("turn_start")`, `on("session_before_fork")`, `exec` |
| `auto_commit_on_exit.py` | Commit on shutdown | `on("session_shutdown")`, `exec` |
| **UI Components** |||
| `status_line.py` | Footer status indicator | `set_status`, session events |
| `working_indicator.py` | Customize the streaming working indicator | `set_working_indicator`, `register_command` |
| `github_issue_autocomplete.py` | Add `#1234` issue completions | `add_autocomplete_provider`, `on("session_start")`, `exec` |
| `custom_footer.py` | Replace footer entirely | `register_command`, `set_footer` |
| `custom_header.py` | Replace startup header | `on("session_start")`, `set_header` |
| `modal_editor.py` | Vim-style modal editor | `set_editor_component`, `CustomEditor` |
| `rainbow_editor.py` | Custom editor styling | `set_editor_component` |
| `widget_placement.py` | Widget above/below editor | `set_widget` |
| `notify.py` | Simple notifications | `ui.notify` |
| `timed_confirm.py` | Dialogs with timeout | `ui.confirm` with timeout/signal |
| **Complex Extensions** |||
| `plan_mode/` | Full plan mode implementation | All event types, `register_command`, `register_shortcut`, `register_flag`, `set_status`, `set_widget`, `send_message`, `set_active_tools` |
| `preset.py` | Saveable presets (model, tools, thinking) | `register_command`, `register_shortcut`, `register_flag`, `set_model`, `set_active_tools`, `set_thinking_level`, `append_entry` |
| `tools.py` | Toggle tools on/off UI | `register_command`, `set_active_tools`, `SettingsList`, session events |
| **Remote & Sandbox** |||
| `ssh.py` | SSH remote execution | `register_flag`, `on("user_bash")`, `on("before_agent_start")`, tool operations |
| `interactive_shell.py` | Persistent shell session | `on("user_bash")` |
| `sandbox/` | Sandboxed tool execution | Tool operations |
| `subagent/` | Spawn sub-agents | `register_tool`, `exec` |
| **Providers** |||
| `custom_provider_anthropic/` | Custom Anthropic proxy | `register_provider` |
| `custom_provider_gitlab_duo/` | GitLab Duo integration | `register_provider` with OAuth |
| **Messages & Communication** |||
| `message_renderer.py` | Custom message rendering | `register_message_renderer`, `send_message` |
| `event_bus.py` | Inter-extension events | `harn.events` |
| **Session Metadata** |||
| `session_name.py` | Name sessions for selector | `set_session_name`, `get_session_name` |
| `bookmark.py` | Bookmark entries for /tree | `set_label` |
| **Misc** |||
| `inline_bash.py` | Inline bash in tool calls | `on("tool_call")` |
| `bash_spawn_hook.py` | Adjust bash command, cwd, and env before execution | `create_bash_tool`, `spawn_hook` |
| `with_deps/` | Extension with pip dependencies | Package structure with `pyproject.toml` |
