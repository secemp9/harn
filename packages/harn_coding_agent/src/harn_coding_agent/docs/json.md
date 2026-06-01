# JSON Event Stream Mode

```bash
harn --mode json "Your prompt"
```

Outputs all session events as JSON lines to stdout. Useful for integrating harn into other tools or custom UIs.

## Event Types

```python
from typing import Union, Literal
from dataclasses import dataclass

AgentSessionEvent = Union[
    "AgentEvent",
    "QueueUpdateEvent",
    "CompactionStartEvent",
    "CompactionEndEvent",
    "AutoRetryStartEvent",
    "AutoRetryEndEvent",
]

@dataclass
class QueueUpdateEvent:
    type: Literal["queue_update"] = "queue_update"
    steering: list[str] = None
    follow_up: list[str] = None

@dataclass
class CompactionStartEvent:
    type: Literal["compaction_start"] = "compaction_start"
    reason: str = ""  # "manual", "threshold", "overflow"

@dataclass
class CompactionEndEvent:
    type: Literal["compaction_end"] = "compaction_end"
    reason: str = ""
    result: "CompactionResult | None" = None
    aborted: bool = False
    will_retry: bool = False
    error_message: str | None = None

@dataclass
class AutoRetryStartEvent:
    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""

@dataclass
class AutoRetryEndEvent:
    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool = False
    attempt: int = 0
    final_error: str | None = None
```

`queue_update` emits the full pending steering and follow-up queues whenever they change. `compaction_start` and `compaction_end` cover both manual and automatic compaction.

Base events from `AgentEvent`:

```python
AgentEvent = Union[
    # Agent lifecycle
    "AgentStartEvent",    # {"type": "agent_start"}
    "AgentEndEvent",      # {"type": "agent_end", "messages": [...]}
    # Turn lifecycle
    "TurnStartEvent",     # {"type": "turn_start"}
    "TurnEndEvent",       # {"type": "turn_end", "message": ..., "toolResults": [...]}
    # Message lifecycle
    "MessageStartEvent",  # {"type": "message_start", "message": ...}
    "MessageUpdateEvent", # {"type": "message_update", "message": ..., "assistantMessageEvent": ...}
    "MessageEndEvent",    # {"type": "message_end", "message": ...}
    # Tool execution
    "ToolExecutionStartEvent",   # {"type": "tool_execution_start", ...}
    "ToolExecutionUpdateEvent",  # {"type": "tool_execution_update", ...}
    "ToolExecutionEndEvent",     # {"type": "tool_execution_end", ...}
]
```

## Output Format

Each line is a JSON object. The first line is the session header:

```json
{"type":"session","version":3,"id":"uuid","timestamp":"...","cwd":"/path"}
```

Followed by events as they occur:

```json
{"type":"agent_start"}
{"type":"turn_start"}
{"type":"message_start","message":{"role":"assistant","content":[],...}}
{"type":"message_update","message":{...},"assistantMessageEvent":{"type":"text_delta","delta":"Hello",...}}
{"type":"message_end","message":{...}}
{"type":"turn_end","message":{...},"toolResults":[]}
{"type":"agent_end","messages":[...]}
```

## Example

```bash
harn --mode json "List files" 2>/dev/null | jq -c 'select(.type == "message_end")'
```
