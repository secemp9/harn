# Compaction & Branch Summarization

LLMs have limited context windows. When conversations grow too long, harn uses compaction to summarize older content while preserving recent work. This page covers both auto-compaction and branch summarization.

## Overview

Harn has two summarization mechanisms:

| Mechanism | Trigger | Purpose |
|-----------|---------|---------|
| Compaction | Context exceeds threshold, or `/compact` | Summarize old messages to free up context |
| Branch summarization | `/tree` navigation | Preserve context when switching branches |

Both use the same structured summary format and track file operations cumulatively.

## Compaction

### When It Triggers

Auto-compaction triggers when:

```
contextTokens > contextWindow - reserveTokens
```

By default, `reserveTokens` is 16384 tokens (configurable in `~/.harn/agent/settings.json` or `<project-dir>/.harn/settings.json`). This leaves room for the LLM's response.

You can also trigger manually with `/compact [instructions]`, where optional instructions focus the summary.

### How It Works

1. **Find cut point**: Walk backwards from newest message, accumulating token estimates until `keepRecentTokens` (default 20k, configurable in `~/.harn/agent/settings.json` or `<project-dir>/.harn/settings.json`) is reached
2. **Extract messages**: Collect messages from the previous kept boundary (or session start) up to the cut point
3. **Generate summary**: Call LLM to summarize with structured format, passing the previous summary as iterative context when present
4. **Append entry**: Save `CompactionEntry` with summary and `firstKeptEntryId`
5. **Reload**: Session reloads, using summary + messages from `firstKeptEntryId` onwards

```
Before compaction:

  entry:  0     1     2     3      4     5     6      7      8     9
        +-----+-----+-----+-----+------+-----+-----+------+------+-----+
        | hdr | usr | ass | tool | usr | ass | tool | tool | ass | tool|
        +-----+-----+-----+-----+------+-----+-----+------+------+-----+
                +--------+-------+ +--------------+------------------+
               messagesToSummarize            kept messages
                                   ^
                          firstKeptEntryId (entry 4)

After compaction (new entry appended):

  entry:  0     1     2     3      4     5     6      7      8     9     10
        +-----+-----+-----+-----+------+-----+-----+------+------+-----+-----+
        | hdr | usr | ass | tool | usr | ass | tool | tool | ass | tool| cmp |
        +-----+-----+-----+-----+------+-----+-----+------+------+-----+-----+
               +----------+------+ +----------------------+-------------------+
                 not sent to LLM                    sent to LLM
                                                         ^
                                              starts from firstKeptEntryId

What the LLM sees:

  +--------+---------+-----+-----+------+------+-----+------+
  | system | summary | usr | ass | tool | tool | ass | tool |
  +--------+---------+-----+-----+------+------+-----+------+
       ^         ^      +------------------+----------------+
    prompt   from cmp          messages from firstKeptEntryId
```

On repeated compactions, the summarized span starts at the previous compaction's kept boundary (`firstKeptEntryId`), not at the compaction entry itself. This preserves messages that survived the earlier compaction by including them in the next summarization pass as well. Harn also recalculates `tokensBefore` from the rebuilt session context before writing the new `CompactionEntry`, so the token count reflects the actual pre-compaction context being replaced.

### Split Turns

A "turn" starts with a user message and includes all assistant responses and tool calls until the next user message. Normally, compaction cuts at turn boundaries.

When a single turn exceeds `keepRecentTokens`, the cut point lands mid-turn at an assistant message. This is a "split turn":

```
Split turn (one huge turn exceeds budget):

  entry:  0     1     2      3     4      5      6     7      8
        +-----+-----+-----+------+-----+------+------+-----+------+
        | hdr | usr | ass | tool | ass | tool | tool | ass | tool |
        +-----+-----+-----+------+-----+------+------+-----+------+
                ^                                     ^
         turnStartIndex = 1                  firstKeptEntryId = 7
                |                                     |
                +---- turnPrefixMessages (1-6) -------+
                                                      +-- kept (7-8)

  isSplitTurn = true
  messagesToSummarize = []  (no complete turns before)
  turnPrefixMessages = [usr, ass, tool, ass, tool, tool]
```

For split turns, harn generates two summaries and merges them:
1. **History summary**: Previous context (if any)
2. **Turn prefix summary**: The early part of the split turn

### Cut Point Rules

Valid cut points are:
- User messages
- Assistant messages
- BashExecution messages
- Custom messages (custom_message, branch_summary)

Never cut at tool results (they must stay with their tool call).

### CompactionEntry Structure

```python
@dataclass
class CompactionEntry:
    type: str = "compaction"
    id: str = ""
    parent_id: str = ""
    timestamp: int = 0
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    from_hook: bool = None   # true if provided by extension (legacy field name)
    details: Any = None      # implementation-specific data

@dataclass
class CompactionDetails:
    read_files: list[str] = None
    modified_files: list[str] = None
```

Extensions can store any JSON-serializable data in `details`. The default compaction tracks file operations, but custom extension implementations can use their own structure.

## Branch Summarization

### When It Triggers

When you use `/tree` to navigate to a different branch, harn offers to summarize the work you're leaving. This injects context from the left branch into the new branch.

### How It Works

1. **Find common ancestor**: Deepest node shared by old and new positions
2. **Collect entries**: Walk from old leaf back to common ancestor
3. **Prepare with budget**: Include messages up to token budget (newest first)
4. **Generate summary**: Call LLM with structured format
5. **Append entry**: Save `BranchSummaryEntry` at navigation point

```
Tree before navigation:

         +- B - C - D (old leaf, being abandoned)
    A ---+
         +- E - F (target)

Common ancestor: A
Entries to summarize: B, C, D

After navigation with summary:

         +- B - C - D - [summary of B,C,D]
    A ---+
         +- E - F (new leaf)
```

### Cumulative File Tracking

Both compaction and branch summarization track files cumulatively. When generating a summary, harn extracts file operations from:
- Tool calls in the messages being summarized
- Previous compaction or branch summary `details` (if any)

This means file tracking accumulates across multiple compactions or nested branch summaries, preserving the full history of read and modified files.

### BranchSummaryEntry Structure

```python
@dataclass
class BranchSummaryEntry:
    type: str = "branch_summary"
    id: str = ""
    parent_id: str = ""
    timestamp: int = 0
    summary: str = ""
    from_id: str = ""       # Entry we navigated from
    from_hook: bool = None  # true if provided by extension (legacy field name)
    details: Any = None     # implementation-specific data

@dataclass
class BranchSummaryDetails:
    read_files: list[str] = None
    modified_files: list[str] = None
```

Same as compaction, extensions can store custom data in `details`.

## Summary Format

Both compaction and branch summarization use the same structured format:

```markdown
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements mentioned by user]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

## Critical Context
- [Data needed to continue]

<read-files>
path/to/file1.py
path/to/file2.py
</read-files>

<modified-files>
path/to/changed.py
</modified-files>
```

### Message Serialization

Before summarization, messages are serialized to text via `serialize_conversation()`:

```
[User]: What they said
[Assistant thinking]: Internal reasoning
[Assistant]: Response text
[Assistant tool calls]: read(path="foo.py"); edit(path="bar.py", ...)
[Tool result]: Output from tool
```

This prevents the model from treating it as a conversation to continue.

Tool results are truncated to 2000 characters during serialization. Content beyond that limit is replaced with a marker indicating how many characters were truncated. This keeps summarization requests within reasonable token budgets, since tool results (especially from `read` and `bash`) are typically the largest contributors to context size.

## Custom Summarization via Extensions

Extensions can intercept and customize both compaction and branch summarization.

### session_before_compact

Fired before auto-compaction or `/compact`. Can cancel or provide custom summary.

```python
@harn.on("session_before_compact")
async def on_before_compact(event, ctx):
    preparation = event.preparation
    # preparation.messages_to_summarize - messages to summarize
    # preparation.turn_prefix_messages - split turn prefix (if is_split_turn)
    # preparation.previous_summary - previous compaction summary
    # preparation.file_ops - extracted file operations
    # preparation.tokens_before - context tokens before compaction
    # preparation.first_kept_entry_id - where kept messages start
    # preparation.settings - compaction settings

    # Cancel:
    return {"cancel": True}

    # Custom summary:
    return {
        "compaction": {
            "summary": "Your summary...",
            "first_kept_entry_id": preparation.first_kept_entry_id,
            "tokens_before": preparation.tokens_before,
            "details": { ... },  # custom data
        }
    }
```

#### Converting Messages to Text

To generate a summary with your own model, convert messages to text using `serialize_conversation`:

```python
from harn_coding_agent import convert_to_llm, serialize_conversation

@harn.on("session_before_compact")
async def on_before_compact(event, ctx):
    preparation = event.preparation

    # Convert AgentMessage[] to Message[], then serialize to text
    conversation_text = serialize_conversation(
        convert_to_llm(preparation.messages_to_summarize)
    )
    # Returns:
    # [User]: message text
    # [Assistant thinking]: thinking content
    # [Assistant]: response text
    # [Assistant tool calls]: read(path="..."); bash(command="...")
    # [Tool result]: output text

    # Now send to your model for summarization
    summary = await my_model.summarize(conversation_text)

    return {
        "compaction": {
            "summary": summary,
            "first_kept_entry_id": preparation.first_kept_entry_id,
            "tokens_before": preparation.tokens_before,
        }
    }
```

### session_before_tree

Fired before `/tree` navigation. Always fires regardless of whether user chose to summarize. Can cancel navigation or provide custom summary.

```python
@harn.on("session_before_tree")
async def on_before_tree(event, ctx):
    preparation = event.preparation
    # preparation.target_id - where we're navigating to
    # preparation.old_leaf_id - current position (being abandoned)
    # preparation.common_ancestor_id - shared ancestor
    # preparation.entries_to_summarize - entries that would be summarized
    # preparation.user_wants_summary - whether user chose to summarize

    # Cancel navigation entirely:
    return {"cancel": True}

    # Provide custom summary (only used if user_wants_summary is True):
    if preparation.user_wants_summary:
        return {
            "summary": {
                "summary": "Your summary...",
                "details": { ... },  # custom data
            }
        }
```

## Settings

Configure compaction in `~/.harn/agent/settings.json` or `<project-dir>/.harn/settings.json`:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable auto-compaction |
| `reserveTokens` | `16384` | Tokens to reserve for LLM response |
| `keepRecentTokens` | `20000` | Recent tokens to keep (not summarized) |

Disable auto-compaction with `"enabled": false`. You can still compact manually with `/compact`.
