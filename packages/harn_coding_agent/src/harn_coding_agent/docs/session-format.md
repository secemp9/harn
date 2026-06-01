# Session File Format

Sessions are stored as JSONL (JSON Lines) files. Each line is a JSON object with a `type` field. Session entries form a tree structure via `id`/`parentId` fields, enabling in-place branching without creating new files.

## File Location

```
~/.harn/agent/sessions/--<path>--/<timestamp>_<uuid>.jsonl
```

Where `<path>` is the working directory with `/` replaced by `-`.

## Deleting Sessions

Sessions can be removed by deleting their `.jsonl` files under `~/.harn/agent/sessions/`.

Harn also supports deleting sessions interactively from `/resume` (select a session and press `Ctrl+D`, then confirm). When available, harn uses the `trash` CLI to avoid permanent deletion.

## Session Version

Sessions have a version field in the header:

- **Version 1**: Linear entry sequence (legacy, auto-migrated on load)
- **Version 2**: Tree structure with `id`/`parentId` linking
- **Version 3**: Renamed `hookMessage` role to `custom` (extensions unification)

Existing sessions are automatically migrated to the current version (v3) when loaded.

## Message Types

Session entries contain `AgentMessage` objects. Understanding these types is essential for parsing sessions and writing extensions.

### Content Blocks

Messages contain arrays of typed content blocks:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class TextContent:
    type: str = "text"
    text: str = ""

@dataclass
class ImageContent:
    type: str = "image"
    data: str = ""        # base64 encoded
    mime_type: str = ""   # e.g., "image/jpeg", "image/png"

@dataclass
class ThinkingContent:
    type: str = "thinking"
    thinking: str = ""

@dataclass
class ToolCall:
    type: str = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = None
```

### Base Message Types (from harn-ai)

```python
@dataclass
class UserMessage:
    role: str = "user"
    content: str | list[TextContent | ImageContent] = ""
    timestamp: int = 0  # Unix ms

@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall] = None
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: "Usage" = None
    stop_reason: str = ""  # "stop", "length", "toolUse", "error", "aborted"
    error_message: str | None = None
    timestamp: int = 0

@dataclass
class ToolResultMessage:
    role: str = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[TextContent | ImageContent] = None
    details: Any = None      # Tool-specific metadata
    is_error: bool = False
    timestamp: int = 0

@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: "Cost" = None

@dataclass
class Cost:
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0
    total: float = 0
```

### Extended Message Types (from harn-coding-agent)

```python
@dataclass
class BashExecutionMessage:
    role: str = "bashExecution"
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    exclude_from_context: bool = False  # true for !! prefix commands
    timestamp: int = 0

@dataclass
class CustomMessage:
    role: str = "custom"
    custom_type: str = ""            # Extension identifier
    content: str | list[TextContent | ImageContent] = ""
    display: bool = False            # Show in TUI
    details: Any = None              # Extension-specific metadata
    timestamp: int = 0

@dataclass
class BranchSummaryMessage:
    role: str = "branchSummary"
    summary: str = ""
    from_id: str = ""                # Entry we branched from
    timestamp: int = 0

@dataclass
class CompactionSummaryMessage:
    role: str = "compactionSummary"
    summary: str = ""
    tokens_before: int = 0
    timestamp: int = 0
```

### AgentMessage Union

```python
from typing import Union

AgentMessage = Union[
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    BashExecutionMessage,
    CustomMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
]
```

## Entry Base

All entries (except `SessionHeader`) extend `SessionEntryBase`:

```python
@dataclass
class SessionEntryBase:
    type: str = ""
    id: str = ""               # 8-char hex ID
    parent_id: str | None = None  # Parent entry ID (null for first entry)
    timestamp: str = ""        # ISO timestamp
```

## Entry Types

### SessionHeader

First line of the file. Metadata only, not part of the tree (no `id`/`parentId`).

```json
{"type":"session","version":3,"id":"uuid","timestamp":"2024-12-03T14:00:00.000Z","cwd":"/path/to/project"}
```

For sessions with a parent (created via `/fork`, `/clone`, or `new_session(parent_session=...)`):

```json
{"type":"session","version":3,"id":"uuid","timestamp":"2024-12-03T14:00:00.000Z","cwd":"/path/to/project","parentSession":"/path/to/original/session.jsonl"}
```

### SessionMessageEntry

A message in the conversation. The `message` field contains an `AgentMessage`.

```json
{"type":"message","id":"a1b2c3d4","parentId":"prev1234","timestamp":"2024-12-03T14:00:01.000Z","message":{"role":"user","content":"Hello"}}
{"type":"message","id":"b2c3d4e5","parentId":"a1b2c3d4","timestamp":"2024-12-03T14:00:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}],"provider":"anthropic","model":"claude-sonnet-4-5","usage":{...},"stopReason":"stop"}}
{"type":"message","id":"c3d4e5f6","parentId":"b2c3d4e5","timestamp":"2024-12-03T14:00:03.000Z","message":{"role":"toolResult","toolCallId":"call_123","toolName":"bash","content":[{"type":"text","text":"output"}],"isError":false}}
```

### ModelChangeEntry

Emitted when the user switches models mid-session.

```json
{"type":"model_change","id":"d4e5f6g7","parentId":"c3d4e5f6","timestamp":"2024-12-03T14:05:00.000Z","provider":"openai","modelId":"gpt-4o"}
```

### ThinkingLevelChangeEntry

Emitted when the user changes the thinking/reasoning level.

```json
{"type":"thinking_level_change","id":"e5f6g7h8","parentId":"d4e5f6g7","timestamp":"2024-12-03T14:06:00.000Z","thinkingLevel":"high"}
```

### CompactionEntry

Created when context is compacted. Stores a summary of earlier messages.

```json
{"type":"compaction","id":"f6g7h8i9","parentId":"e5f6g7h8","timestamp":"2024-12-03T14:10:00.000Z","summary":"User discussed X, Y, Z...","firstKeptEntryId":"c3d4e5f6","tokensBefore":50000}
```

Optional fields:
- `details`: Implementation-specific data (e.g., `{ readFiles: string[], modifiedFiles: string[] }` for default, or custom data for extensions)
- `fromHook`: `true` if generated by an extension, `false`/`undefined` if harn-generated (legacy field name)

### BranchSummaryEntry

Created when switching branches via `/tree` with an LLM generated summary of the left branch up to the common ancestor. Captures context from the abandoned path.

```json
{"type":"branch_summary","id":"g7h8i9j0","parentId":"a1b2c3d4","timestamp":"2024-12-03T14:15:00.000Z","fromId":"f6g7h8i9","summary":"Branch explored approach A..."}
```

Optional fields:
- `details`: File tracking data (`{ readFiles: string[], modifiedFiles: string[] }`) for default, or custom data for extensions
- `fromHook`: `true` if generated by an extension, `false`/`undefined` if harn-generated (legacy field name)

### CustomEntry

Extension state persistence. Does NOT participate in LLM context.

```json
{"type":"custom","id":"h8i9j0k1","parentId":"g7h8i9j0","timestamp":"2024-12-03T14:20:00.000Z","customType":"my-extension","data":{"count":42}}
```

Use `customType` to identify your extension's entries on reload.

### CustomMessageEntry

Extension-injected messages that DO participate in LLM context.

```json
{"type":"custom_message","id":"i9j0k1l2","parentId":"h8i9j0k1","timestamp":"2024-12-03T14:25:00.000Z","customType":"my-extension","content":"Injected context...","display":true}
```

Fields:
- `content`: String or `[TextContent | ImageContent]` (same as UserMessage)
- `display`: `true` = show in TUI with distinct styling, `false` = hidden
- `details`: Optional extension-specific metadata (not sent to LLM)

### LabelEntry

User-defined bookmark/marker on an entry.

```json
{"type":"label","id":"j0k1l2m3","parentId":"i9j0k1l2","timestamp":"2024-12-03T14:30:00.000Z","targetId":"a1b2c3d4","label":"checkpoint-1"}
```

Set `label` to `null` to clear a label.

### SessionInfoEntry

Session metadata (e.g., user-defined display name). Set via `/name` command or `harn.set_session_name()` in extensions.

```json
{"type":"session_info","id":"k1l2m3n4","parentId":"j0k1l2m3","timestamp":"2024-12-03T14:35:00.000Z","name":"Refactor auth module"}
```

The session name is displayed in the session selector (`/resume`) instead of the first message when set.

## Tree Structure

Entries form a tree:
- First entry has `parentId: null`
- Each subsequent entry points to its parent via `parentId`
- Branching creates new children from an earlier entry
- The "leaf" is the current position in the tree

```
[user msg] --- [assistant] --- [user msg] --- [assistant] -+- [user msg] <- current leaf
                                                           |
                                                           +- [branch_summary] --- [user msg] <- alternate branch
```

## Context Building

`build_session_context()` walks from the current leaf to the root, producing the message list for the LLM:

1. Collects all entries on the path
2. Extracts current model and thinking level settings
3. If a `CompactionEntry` is on the path:
   - Emits the summary first
   - Then messages from `firstKeptEntryId` to compaction
   - Then messages after compaction
4. Converts `BranchSummaryEntry` and `CustomMessageEntry` to appropriate message formats

## Parsing Example

```python
import json

with open("session.jsonl") as f:
    lines = f.read().strip().split("\n")

for line in lines:
    entry = json.loads(line)

    match entry["type"]:
        case "session":
            print(f"Session v{entry.get('version', 1)}: {entry['id']}")
        case "message":
            print(f"[{entry['id']}] {entry['message']['role']}: {json.dumps(entry['message']['content'])}")
        case "compaction":
            print(f"[{entry['id']}] Compaction: {entry['tokensBefore']} tokens summarized")
        case "branch_summary":
            print(f"[{entry['id']}] Branch from {entry['fromId']}")
        case "custom":
            print(f"[{entry['id']}] Custom ({entry['customType']}): {json.dumps(entry['data'])}")
        case "custom_message":
            print(f"[{entry['id']}] Extension message ({entry['customType']}): {entry['content']}")
        case "label":
            print(f"[{entry['id']}] Label \"{entry['label']}\" on {entry['targetId']}")
        case "model_change":
            print(f"[{entry['id']}] Model: {entry['provider']}/{entry['modelId']}")
        case "thinking_level_change":
            print(f"[{entry['id']}] Thinking: {entry['thinkingLevel']}")
```

## SessionManager API

Key methods for working with sessions programmatically.

### Static Creation Methods
- `SessionManager.create(cwd, session_dir=None)` - New session
- `SessionManager.open(path, session_dir=None)` - Open existing session file
- `SessionManager.continue_recent(cwd, session_dir=None)` - Continue most recent or create new
- `SessionManager.in_memory(cwd=None)` - No file persistence
- `SessionManager.fork_from(source_path, target_cwd, session_dir=None)` - Fork session from another project

### Static Listing Methods
- `SessionManager.list(cwd, session_dir=None, on_progress=None)` - List sessions for a directory
- `SessionManager.list_all(on_progress=None)` - List all sessions across all projects

### Instance Methods - Session Management
- `new_session(options=None)` - Start a new session (options: `parent_session: str`)
- `set_session_file(path)` - Switch to a different session file
- `create_branched_session(leaf_id)` - Extract branch to new session file

### Instance Methods - Appending (all return entry ID)
- `append_message(message)` - Add message
- `append_thinking_level_change(level)` - Record thinking change
- `append_model_change(provider, model_id)` - Record model change
- `append_compaction(summary, first_kept_entry_id, tokens_before, details=None, from_hook=None)` - Add compaction
- `append_custom_entry(custom_type, data=None)` - Extension state (not in context)
- `append_session_info(name)` - Set session display name
- `append_custom_message_entry(custom_type, content, display, details=None)` - Extension message (in context)
- `append_label_change(target_id, label)` - Set/clear label

### Instance Methods - Tree Navigation
- `get_leaf_id()` - Current position
- `get_leaf_entry()` - Get current leaf entry
- `get_entry(id)` - Get entry by ID
- `get_branch(from_id=None)` - Walk from entry to root
- `get_tree()` - Get full tree structure
- `get_children(parent_id)` - Get direct children
- `get_label(id)` - Get label for entry
- `branch(entry_id)` - Move leaf to earlier entry
- `reset_leaf()` - Reset leaf to None (before any entries)
- `branch_with_summary(entry_id, summary, details=None, from_hook=None)` - Branch with context summary

### Instance Methods - Context & Info
- `build_session_context()` - Get messages, thinking_level, and model for LLM
- `get_entries()` - All entries (excluding header)
- `get_header()` - Session header metadata
- `get_session_name()` - Get display name from latest session_info entry
- `get_cwd()` - Working directory
- `get_session_dir()` - Session storage directory
- `get_session_id()` - Session UUID
- `get_session_file()` - Session file path (None for in-memory)
- `is_persisted()` - Whether session is saved to disk
