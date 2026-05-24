from __future__ import annotations

import time
from typing import Any

import pytest
from harnify_agent.harness.compaction.branch_summarization import (
    BRANCH_SUMMARY_PREAMBLE,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    prepare_branch_entries,
)
from harnify_agent.harness.compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_summary,
    get_last_assistant_usage,
    prepare_compaction,
    serialize_conversation,
    should_compact,
)
from harnify_agent.harness.session.memory_storage import InMemorySessionStorage
from harnify_agent.harness.session.session import Session, build_session_context
from harnify_agent.harness.types import (
    CompactionPreparation,
    CompactionSettings,
    FileOperations,
    GenerateBranchSummaryOptions,
    get_or_throw,
)
from harnify_ai.providers.faux import faux_assistant_message, register_faux_provider
from harnify_ai.types import AssistantMessage, Model, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage


def create_id() -> str:
    create_id.counter += 1
    return f"entry-{create_id.counter}"


create_id.counter = 0


def create_mock_usage(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cacheRead=cache_read,
        cacheWrite=cache_write,
        totalTokens=input_tokens + output_tokens + cache_read + cache_write,
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=int(time.time() * 1000))


def create_assistant_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=usage or create_mock_usage(100, 50),
        stopReason="stop",
        timestamp=int(time.time() * 1000),
    )


def create_message_entry(message: Any, parent_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "message",
        "id": create_id(),
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": message,
    }


def create_compaction_entry(summary: str, first_kept_entry_id: str, parent_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "compaction",
        "id": create_id(),
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "summary": summary,
        "firstKeptEntryId": first_kept_entry_id,
        "tokensBefore": 1234,
    }


def create_thinking_level_entry(level: str, parent_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "thinking_level_change",
        "id": create_id(),
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "thinkingLevel": level,
    }


def create_model_change_entry(provider: str, model_id: str, parent_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "model_change",
        "id": create_id(),
        "parentId": parent_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "provider": provider,
        "modelId": model_id,
    }


def create_faux_model(
    registrations: list[Any],
    *,
    reasoning: bool,
    max_tokens: int = 8192,
) -> Model:
    registration = register_faux_provider(
        {
            "models": [
                {
                    "id": "reasoning-model" if reasoning else "non-reasoning-model",
                    "reasoning": reasoning,
                    "contextWindow": 200000,
                    "maxTokens": max_tokens,
                }
            ]
        }
    )
    registrations.append(registration)
    model = registration.get_model()
    assert model is not None
    return model


def capture_options_response(sink: list[dict[str, Any]], text: str):
    def responder(_context, options, _state, _model):
        sink.append(options.model_dump())
        return faux_assistant_message(text)

    return responder


def capture_max_tokens_response(sink: list[int], text: str):
    def responder(_context, options, _state, _model):
        sink.append(options.maxTokens)
        return faux_assistant_message(text)

    return responder


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    create_id.counter = 0


@pytest.fixture
def registrations() -> list[Any]:
    items: list[Any] = []
    yield items
    while items:
        items.pop().unregister()


def test_compaction_threshold_and_usage_helpers() -> None:
    settings = CompactionSettings(enabled=True, reserveTokens=10000, keepRecentTokens=20000)

    assert calculate_context_tokens(create_mock_usage(1000, 500, 200, 100)) == 1800
    assert calculate_context_tokens(create_mock_usage(0, 0, 0, 0)) == 0
    assert should_compact(95000, 100000, settings) is True
    assert should_compact(89000, 100000, settings) is False
    assert should_compact(
        95000,
        100000,
        CompactionSettings(enabled=False, reserveTokens=1, keepRecentTokens=1),
    ) is False


def test_cut_point_and_turn_start_edge_cases() -> None:
    thinking = create_thinking_level_entry("high")
    model_change = create_model_change_entry("openai", "gpt-4", thinking["id"])
    result = find_cut_point([thinking, model_change], 0, 2, 1)
    assert result.firstKeptEntryIndex == 0
    assert result.turnStartIndex == -1
    assert result.isSplitTurn is False

    branch_summary = {
        "type": "branch_summary",
        "id": create_id(),
        "parentId": model_change["id"],
        "timestamp": "2026-01-01T00:00:00.000Z",
        "fromId": "branch",
        "summary": "branch summary",
    }
    custom_message = {
        "type": "custom_message",
        "id": create_id(),
        "parentId": branch_summary["id"],
        "timestamp": "2026-01-01T00:00:00.000Z",
        "customType": "note",
        "content": "custom content",
        "display": True,
    }
    assert find_turn_start_index([thinking, branch_summary], 1, 0) == 1
    assert find_turn_start_index([thinking, custom_message], 1, 0) == 1
    assert find_turn_start_index([thinking, model_change], 1, 0) == -1

    result = find_cut_point([thinking, branch_summary, custom_message], 0, 3, 1)
    assert result.firstKeptEntryIndex == 0

    tool_result = create_message_entry(
        ToolResultMessage(
            toolCallId="call-1",
            toolName="read",
            content=[TextContent(text="tool output")],
            isError=False,
            timestamp=int(time.time() * 1000),
        )
    )
    result = find_cut_point([tool_result], 0, 1, 1)
    assert result.firstKeptEntryIndex == 0
    assert result.turnStartIndex == -1
    assert result.isSplitTurn is False

    user = create_message_entry(create_user_message("user"))
    compaction = create_compaction_entry("summary", user["id"], user["id"])
    assistant = create_message_entry(create_assistant_message("assistant"), compaction["id"])
    assert find_cut_point([user, compaction, assistant], 0, 3, 1).firstKeptEntryIndex == 2


def test_estimate_tokens_and_context_usage_across_roles() -> None:
    usage = create_mock_usage(10, 5, 3, 2)
    assistant = create_assistant_message("assistant", usage)
    assistant_with_thinking_and_tool = assistant.model_copy(
        update={
            "content": [
                {"type": "thinking", "thinking": "thinking"},
                {"type": "toolCall", "id": "call-1", "name": "read", "arguments": {"path": "file.ts"}},
            ]
        }
    )
    custom_string = {
        "role": "custom",
        "customType": "note",
        "content": "custom text",
        "display": True,
        "timestamp": int(time.time() * 1000),
    }
    tool_result_with_image = ToolResultMessage(
        toolCallId="call-1",
        toolName="read",
        content=[TextContent(text="tool text"), {"type": "image", "mimeType": "image/png", "data": "abc"}],
        isError=False,
        timestamp=int(time.time() * 1000),
    )
    bash_execution = {
        "role": "bashExecution",
        "command": "npm run check",
        "output": "ok",
        "exitCode": 0,
        "cancelled": False,
        "truncated": False,
        "timestamp": int(time.time() * 1000),
    }
    branch_summary_message = {
        "role": "branchSummary",
        "summary": "branch",
        "fromId": "x",
        "timestamp": int(time.time() * 1000),
    }
    compaction_summary_message = {
        "role": "compactionSummary",
        "summary": "compact",
        "tokensBefore": 123,
        "timestamp": int(time.time() * 1000),
    }

    assert estimate_tokens({"role": "user", "content": "plain user", "timestamp": int(time.time() * 1000)}) > 0
    assert estimate_tokens(assistant_with_thinking_and_tool) > 0
    assert estimate_tokens(custom_string) > 0
    assert estimate_tokens(tool_result_with_image) > 1000
    assert estimate_tokens(bash_execution) > 0
    assert estimate_tokens(branch_summary_message) > 0
    assert estimate_tokens(compaction_summary_message) > 0
    assert estimate_tokens({"role": "unknown", "timestamp": int(time.time() * 1000)}) == 0

    assert get_last_assistant_usage(
        [create_message_entry(create_user_message("user")), create_message_entry(assistant)]
    ) == usage
    assert get_last_assistant_usage(
        [
            create_message_entry(assistant.model_copy(update={"stopReason": "aborted"})),
            create_message_entry(assistant.model_copy(update={"stopReason": "error"})),
        ]
    ) is None
    assert estimate_context_tokens([create_user_message("no usage")]).lastUsageIndex is None
    assert estimate_context_tokens([assistant, create_user_message("tail")]).usageTokens == 20


def test_prepare_compaction_and_serialization_behaviour() -> None:
    u1 = create_message_entry(create_user_message("user msg 1"))
    a1 = create_message_entry(create_assistant_message("assistant msg 1"), u1["id"])
    u2 = create_message_entry(create_user_message("user msg 2"), a1["id"])
    a2 = create_message_entry(create_assistant_message("assistant msg 2", create_mock_usage(5000, 1000)), u2["id"])
    compaction1 = create_compaction_entry("First summary", u2["id"], a2["id"])
    u3 = create_message_entry(create_user_message("user msg 3"), compaction1["id"])
    a3 = create_message_entry(create_assistant_message("assistant msg 3", create_mock_usage(8000, 2000)), u3["id"])
    path_entries = [u1, a1, u2, a2, compaction1, u3, a3]
    preparation = get_or_throw(prepare_compaction(path_entries, DEFAULT_COMPACTION_SETTINGS))
    assert preparation is not None
    assert preparation.previousSummary == "First summary"
    assert preparation.firstKeptEntryId
    assert preparation.tokensBefore == estimate_context_tokens(build_session_context(path_entries).messages).tokens

    assistant_with_tool = create_assistant_message("assistant msg 1").model_copy(
        update={"content": [ToolCall(id="tool-1", name="write", arguments={"path": "written.ts"})]}
    )
    u1 = create_message_entry(create_user_message("user msg 1"))
    a1 = create_message_entry(assistant_with_tool, u1["id"])
    compaction1 = create_compaction_entry("First summary", u1["id"], a1["id"])
    compaction1["details"] = {"readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts"]}
    u2 = create_message_entry(create_user_message("large turn"), compaction1["id"])
    a2 = create_message_entry(create_assistant_message("large assistant message"), u2["id"])
    preparation = get_or_throw(
        prepare_compaction(
            [u1, a1, compaction1, u2, a2],
            CompactionSettings(enabled=True, reserveTokens=100, keepRecentTokens=1),
        )
    )
    assert preparation is not None
    assert preparation.previousSummary == "First summary"
    assert preparation.isSplitTurn is True
    assert [message.role for message in preparation.turnPrefixMessages] == ["user"]
    assert "old-read.ts" in preparation.fileOps.read
    assert "old-edit.ts" in preparation.fileOps.edited
    assert "written.ts" in preparation.fileOps.written

    branch_summary = {
        "type": "branch_summary",
        "id": create_id(),
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "fromId": "branch",
        "summary": "branch summary",
    }
    custom_message = {
        "type": "custom_message",
        "id": create_id(),
        "parentId": branch_summary["id"],
        "timestamp": "2026-01-01T00:00:00.000Z",
        "customType": "note",
        "content": "custom content",
        "display": True,
    }
    user = create_message_entry(create_user_message("keep"), custom_message["id"])
    assistant = create_message_entry(create_assistant_message("assistant"), user["id"])
    preparation = get_or_throw(
        prepare_compaction(
            [branch_summary, custom_message, user, assistant],
            CompactionSettings(enabled=True, reserveTokens=100, keepRecentTokens=1),
        )
    )
    assert preparation is not None
    assert [message.role for message in preparation.messagesToSummarize] == ["branchSummary", "custom"]

    assert get_or_throw(
        prepare_compaction(
            [create_compaction_entry("already compacted", "entry-keep")],
            DEFAULT_COMPACTION_SETTINGS,
        )
    ) is None
    assert get_or_throw(prepare_compaction([], DEFAULT_COMPACTION_SETTINGS)) is None

    long_content = "x" * 5000
    result = serialize_conversation(
        [
            ToolResultMessage(
                toolCallId="tc1",
                toolName="read",
                content=[TextContent(text=long_content)],
                isError=False,
                timestamp=int(time.time() * 1000),
            )
        ]
    )
    assert "[Tool result]:" in result
    assert "[... 3000 more characters truncated]" in result


@pytest.mark.asyncio
async def test_generate_summary_and_compact_behaviour(registrations: list[Any]) -> None:
    messages = [create_user_message("Summarize this.")]
    seen_options: list[dict[str, Any] | None] = []

    reasoning_model = create_faux_model(registrations, reasoning=True)
    registrations[-1].set_responses([capture_options_response(seen_options, "## Goal\nTest summary")])
    get_or_throw(await generate_summary(messages, reasoning_model, 2000, "test-key", thinking_level="medium"))
    assert seen_options[0]["reasoning"] == "medium"

    off_model = create_faux_model(registrations, reasoning=True)
    registrations[-1].set_responses([capture_options_response(seen_options, "## Goal\nTest summary")])
    get_or_throw(await generate_summary(messages, off_model, 2000, "test-key", thinking_level="off"))
    assert "reasoning" not in seen_options[1] or seen_options[1]["reasoning"] is None

    non_reasoning_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([capture_options_response(seen_options, "## Goal\nTest summary")])
    get_or_throw(await generate_summary(messages, non_reasoning_model, 2000, "test-key", thinking_level="medium"))
    assert "reasoning" not in seen_options[2] or seen_options[2]["reasoning"] is None

    prompt_text = ""
    prompt_model = create_faux_model(registrations, reasoning=False)

    def capture_prompt(context, _options, _state, _model):
        nonlocal prompt_text
        prompt_text = context.messages[0].content[0].text
        return faux_assistant_message("## Goal\nTest summary")

    registrations[-1].set_responses([capture_prompt])
    summary = get_or_throw(
        await generate_summary(
            messages,
            prompt_model,
            2000,
            "test-key",
            headers={"x-test": "yes"},
            custom_instructions="focus",
            previous_summary="old summary",
        )
    )
    assert "Test summary" in summary
    assert "<previous-summary>\nold summary\n</previous-summary>" in prompt_text
    assert "Additional focus: focus" in prompt_text

    error_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="error", error_message="boom")])
    error_result = await generate_summary(messages, error_model, 2000, "test-key")
    assert error_result.ok is False
    assert error_result.error.code == "summarization_failed"

    aborted_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="aborted", error_message="stopped")])
    aborted_result = await generate_summary(messages, aborted_model, 2000, "test-key")
    assert aborted_result.ok is False
    assert aborted_result.error.code == "aborted"

    clamp_model = create_faux_model(registrations, reasoning=False, max_tokens=128000)
    seen_max_tokens: list[int] = []
    registrations[-1].set_responses(
        [
            capture_max_tokens_response(seen_max_tokens, "## Goal\nTest summary"),
            capture_max_tokens_response(seen_max_tokens, "## Goal\nTest summary"),
        ]
    )
    preparation = CompactionPreparation(
        firstKeptEntryId="entry-keep",
        messagesToSummarize=messages,
        turnPrefixMessages=messages,
        isSplitTurn=True,
        tokensBefore=600000,
        previousSummary=None,
        fileOps=FileOperations(),
        settings=CompactionSettings(enabled=True, reserveTokens=500000, keepRecentTokens=20000),
    )
    get_or_throw(await compact(preparation, clamp_model, "test-key"))
    assert seen_max_tokens == [128000, 128000]

    history_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="error", error_message="history failed")])
    invalid_prep = CompactionPreparation(
        firstKeptEntryId="entry-keep",
        messagesToSummarize=messages,
        turnPrefixMessages=[],
        isSplitTurn=False,
        tokensBefore=100,
        previousSummary=None,
        fileOps=FileOperations(),
        settings=CompactionSettings(enabled=True, reserveTokens=2000, keepRecentTokens=20),
    )
    failed = await compact(invalid_prep, history_model, "test-key")
    assert failed.ok is False
    assert failed.error.message == "Summarization failed: history failed"

    invalid_result = await compact(
        CompactionPreparation(
            firstKeptEntryId="",
            messagesToSummarize=[],
            turnPrefixMessages=[],
            isSplitTurn=False,
            tokensBefore=100,
            previousSummary=None,
            fileOps=FileOperations(),
            settings=CompactionSettings(enabled=True, reserveTokens=2000, keepRecentTokens=20),
        ),
        history_model,
        "test-key",
    )
    assert invalid_result.ok is False
    assert invalid_result.error.code == "invalid_session"

    prefix_model = create_faux_model(registrations, reasoning=True)
    prefix_options: list[dict[str, Any] | None] = []
    registrations[-1].set_responses(
        [capture_options_response(prefix_options, "## Original Request\nTest summary")]
    )
    get_or_throw(
        await compact(
            CompactionPreparation(
                firstKeptEntryId="entry-keep",
                messagesToSummarize=[],
                turnPrefixMessages=messages,
                isSplitTurn=True,
                tokensBefore=100,
                previousSummary=None,
                fileOps=FileOperations(),
                settings=CompactionSettings(enabled=True, reserveTokens=2000, keepRecentTokens=20),
            ),
            prefix_model,
            "test-key",
            thinking_level="high",
        )
    )
    assert prefix_options[0]["reasoning"] == "high"

    prefix_error_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="error", error_message="prefix failed")])
    prefix_failed = await compact(
        CompactionPreparation(
            firstKeptEntryId="entry-keep",
            messagesToSummarize=[],
            turnPrefixMessages=messages,
            isSplitTurn=True,
            tokensBefore=100,
            previousSummary=None,
            fileOps=FileOperations(),
            settings=CompactionSettings(enabled=True, reserveTokens=2000, keepRecentTokens=20),
        ),
        prefix_error_model,
        "test-key",
    )
    assert prefix_failed.ok is False
    assert prefix_failed.error.message == "Turn prefix summarization failed: prefix failed"

    prefix_aborted_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="aborted", error_message="prefix stopped")])
    prefix_aborted = await compact(
        CompactionPreparation(
            firstKeptEntryId="entry-keep",
            messagesToSummarize=[],
            turnPrefixMessages=messages,
            isSplitTurn=True,
            tokensBefore=100,
            previousSummary=None,
            fileOps=FileOperations(),
            settings=CompactionSettings(enabled=True, reserveTokens=2000, keepRecentTokens=20),
        ),
        prefix_aborted_model,
        "test-key",
    )
    assert prefix_aborted.ok is False
    assert prefix_aborted.error.code == "aborted"

    u1 = create_message_entry(create_user_message("read a file"))
    assistant_message = create_assistant_message("calling tool", create_mock_usage(1000, 200)).model_copy(
        update={"content": [ToolCall(id="tool-1", name="read", arguments={"path": "src/index.ts"})]}
    )
    a1 = create_message_entry(assistant_message, u1["id"])
    u2 = create_message_entry(create_user_message("continue"), a1["id"])
    a2 = create_message_entry(create_assistant_message("done", create_mock_usage(4000, 500)), u2["id"])
    preparation_obj = get_or_throw(
        prepare_compaction(
            [u1, a1, u2, a2],
            CompactionSettings(enabled=True, reserveTokens=100, keepRecentTokens=1),
        )
    )
    assert preparation_obj is not None
    file_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses(
        [
            faux_assistant_message("## Goal\nHistory summary"),
            faux_assistant_message("## Original Request\nTurn prefix summary"),
        ]
    )
    result = get_or_throw(await compact(preparation_obj, file_model, "test-key"))
    assert len(result.summary) > 0
    assert result.details == {"readFiles": ["src/index.ts"], "modifiedFiles": []}


@pytest.mark.asyncio
async def test_branch_summary_collection_and_generation(registrations: list[Any]) -> None:
    storage = InMemorySessionStorage()
    session = Session(storage)
    root = await session.appendMessage(create_user_message("root"))
    branch_user = await session.appendMessage(create_user_message("branch user"))
    branch_assistant = await session.appendMessage(
        create_assistant_message("branch assistant").model_copy(
            update={"content": [ToolCall(id="tool-1", name="read", arguments={"path": "src/file.ts"})]}
        )
    )
    await session.moveTo(root)
    target = await session.appendMessage(create_user_message("target"))

    collected = await collect_entries_for_branch_summary(session, branch_assistant, target)
    assert collected.commonAncestorId == root
    assert [entry["id"] for entry in collected.entries] == [branch_user, branch_assistant]

    summary_entry = {
        "type": "branch_summary",
        "id": create_id(),
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "fromId": "branch",
        "summary": "earlier summary",
        "details": {"readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts"]},
    }
    branch_preparation = prepare_branch_entries([summary_entry, *collected.entries], token_budget=10_000)
    assert [message.role for message in branch_preparation.messages][0] == "branchSummary"
    assert "old-read.ts" in branch_preparation.fileOps.read
    assert "old-edit.ts" in branch_preparation.fileOps.edited
    assert "src/file.ts" in branch_preparation.fileOps.read

    no_content_model = create_faux_model(registrations, reasoning=False)
    no_content = await generate_branch_summary(
        [],
        GenerateBranchSummaryOptions(model=no_content_model, apiKey="test", signal=None),
    )
    assert no_content.ok is True
    assert no_content.value.summary == "No content to summarize"

    prompt_text = ""
    branch_model = create_faux_model(registrations, reasoning=False)

    def capture_branch_prompt(context, _options, _state, _model):
        nonlocal prompt_text
        prompt_text = context.messages[0].content[0].text
        return faux_assistant_message("## Goal\nBranch summary")

    registrations[-1].set_responses([capture_branch_prompt])
    branch_result = get_or_throw(
        await generate_branch_summary(
            [summary_entry, *collected.entries],
            GenerateBranchSummaryOptions(
                model=branch_model,
                apiKey="test-key",
                signal=None,
                customInstructions="focus on files",
                reserveTokens=1000,
            ),
        )
    )
    assert branch_result.summary.startswith(BRANCH_SUMMARY_PREAMBLE)
    assert "<read-files>\nold-read.ts\nsrc/file.ts\n</read-files>" in branch_result.summary
    assert "<modified-files>\nold-edit.ts\n</modified-files>" in branch_result.summary
    assert "Additional focus: focus on files" in prompt_text

    replace_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([capture_branch_prompt])
    get_or_throw(
        await generate_branch_summary(
            [summary_entry, *collected.entries],
            GenerateBranchSummaryOptions(
                model=replace_model,
                apiKey="test-key",
                signal=None,
                customInstructions="ONLY THIS",
                replaceInstructions=True,
            ),
        )
    )
    assert prompt_text.endswith("ONLY THIS")

    error_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="error", error_message="branch boom")])
    error_result = await generate_branch_summary(
        [summary_entry, *collected.entries],
        GenerateBranchSummaryOptions(model=error_model, apiKey="test-key", signal=None),
    )
    assert error_result.ok is False
    assert error_result.error.message == "Branch summary failed: branch boom"

    aborted_model = create_faux_model(registrations, reasoning=False)
    registrations[-1].set_responses([faux_assistant_message("", stop_reason="aborted", error_message="branch stopped")])
    aborted_result = await generate_branch_summary(
        [summary_entry, *collected.entries],
        GenerateBranchSummaryOptions(model=aborted_model, apiKey="test-key", signal=None),
    )
    assert aborted_result.ok is False
    assert aborted_result.error.code == "aborted"
