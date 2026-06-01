"""
End-to-end streaming tests against live OpenRouter API.

Ported from the TypeScript E2E tests in the upstream earendil-pi repository:
  - stream.test.ts   (basic text, streaming events, tool calling, thinking, multi-turn)
  - abort.test.ts    (abort mid-stream, immediate abort, abort-then-continue)
  - tokens.test.ts   (token usage stats on abort)
  - empty.test.ts    (empty content, empty string, whitespace-only, empty assistant)
  - responseid.test.ts (responseId populated after completion)
  - context-overflow.test.ts (overflow detection via isContextOverflow)

These tests hit the real OpenRouter API with cheap/fast models.
They are skipped when OPENROUTER_API_KEY is not set.

Models used:
  - google/gemma-3-27b-it  (non-reasoning, cheap, supports images)
  - deepseek/deepseek-r1   (reasoning, emits thinking blocks)
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest

from harn_ai.models import get_model
from harn_ai.stream import complete, stream
from harn_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ProviderStreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from harn_ai.utils.overflow import is_context_overflow

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
skip_no_key = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NON_REASONING_MODEL_ID = "google/gemma-3-27b-it"
REASONING_MODEL_ID = "deepseek/deepseek-r1"


def _get_model(model_id: str) -> Model:
    m = get_model("openrouter", model_id)
    assert m is not None, f"Model {model_id!r} not found in openrouter registry"
    return m


def _opts(**extra: Any) -> ProviderStreamOptions:
    return ProviderStreamOptions(apiKey=OPENROUTER_API_KEY, **extra)


def _now() -> int:
    return time.time_ns() // 1_000_000


def _text_of(msg: AssistantMessage) -> str:
    return "".join(
        block.text for block in msg.content if isinstance(block, TextContent)
    )


def _has_thinking(msg: AssistantMessage) -> bool:
    return any(isinstance(block, ThinkingContent) for block in msg.content)


def _has_tool_call(msg: AssistantMessage) -> bool:
    return any(isinstance(block, ToolCall) for block in msg.content)


CALCULATOR_TOOL = Tool(
    name="math_operation",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The operation to perform.",
            },
        },
        "required": ["a", "b", "operation"],
    },
)

LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum. "
)


# =========================================================================
# 1. Basic text generation (from stream.test.ts :: basicTextGeneration)
# =========================================================================


@skip_no_key
async def test_basic_text_generation():
    """Model should complete basic text generation and multi-turn."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant. Be concise.",
        messages=[
            UserMessage(
                role="user",
                content="Reply with exactly: 'Hello test successful'",
                timestamp=_now(),
            )
        ],
    )
    response = await complete(model, context, _opts())

    assert response.role == "assistant"
    assert response.content, "Expected non-empty content"
    assert response.stopReason != "error", f"Error: {response.errorMessage}"
    text = _text_of(response)
    assert "Hello test successful" in text or "hello test successful" in text.lower()

    # Multi-turn follow-up
    context.messages.append(response)
    context.messages.append(
        UserMessage(
            role="user",
            content="Now say 'Goodbye test successful'",
            timestamp=_now(),
        )
    )
    second = await complete(model, context, _opts())
    assert second.stopReason != "error", f"Error: {second.errorMessage}"
    text2 = _text_of(second)
    assert "goodbye test successful" in text2.lower()


# =========================================================================
# 2. Streaming events (from stream.test.ts :: handleStreaming)
# =========================================================================


@skip_no_key
async def test_streaming_events():
    """Stream should emit text_start, text_delta, text_end events."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(role="user", content="Count from 1 to 3", timestamp=_now())
        ],
        systemPrompt="You are a helpful assistant.",
    )

    s = stream(model, context, _opts())
    text_started = False
    text_chunks = ""
    text_completed = False

    async for event in s:
        if event.type == "text_start":
            text_started = True
        elif event.type == "text_delta":
            text_chunks += event.delta
        elif event.type == "text_end":
            text_completed = True

    result = await s.result()

    assert text_started, "Never received text_start event"
    assert len(text_chunks) > 0, "No text_delta events received"
    assert text_completed, "Never received text_end event"
    assert any(
        b.type == "text" for b in result.content
    ), "Final result has no text content"


# =========================================================================
# 3. Tool calling (from stream.test.ts :: handleToolCall)
# =========================================================================


@skip_no_key
async def test_tool_calling():
    """Model should call the math_operation tool when asked to calculate."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant that uses tools when asked.",
        messages=[
            UserMessage(
                role="user",
                content="Calculate 15 + 27 using the math_operation tool.",
                timestamp=_now(),
            )
        ],
        tools=[CALCULATOR_TOOL],
    )

    s = stream(model, context, _opts())
    has_tool_start = False
    has_tool_delta = False
    has_tool_end = False

    async for event in s:
        if event.type == "toolcall_start":
            has_tool_start = True
        elif event.type == "toolcall_delta":
            has_tool_delta = True
        elif event.type == "toolcall_end":
            has_tool_end = True
            assert event.toolCall.name == "math_operation"

    result = await s.result()

    assert has_tool_start, "Never received toolcall_start"
    assert has_tool_delta, "Never received toolcall_delta"
    assert has_tool_end, "Never received toolcall_end"
    assert result.stopReason == "toolUse", f"Expected toolUse, got {result.stopReason}"
    tool_calls = [b for b in result.content if isinstance(b, ToolCall)]
    assert len(tool_calls) >= 1
    assert tool_calls[0].name == "math_operation"


# =========================================================================
# 4. Thinking / reasoning (from stream.test.ts :: handleThinking)
# =========================================================================


@skip_no_key
async def test_thinking_reasoning():
    """Reasoning model should emit thinking_start/delta/end events."""
    model = _get_model(REASONING_MODEL_ID)
    import random

    rand_a = random.randint(0, 255)
    context = Context(
        messages=[
            UserMessage(
                role="user",
                content=f"Think about {rand_a} + 27. Think step by step. Then output the result.",
                timestamp=_now(),
            )
        ],
        systemPrompt="You are a helpful assistant.",
    )

    s = stream(model, context, _opts(reasoningEffort="high"))
    thinking_started = False
    thinking_chunks = ""
    thinking_completed = False

    async for event in s:
        if event.type == "thinking_start":
            thinking_started = True
        elif event.type == "thinking_delta":
            thinking_chunks += event.delta
        elif event.type == "thinking_end":
            thinking_completed = True

    result = await s.result()

    assert result.stopReason == "stop", f"Error: {result.errorMessage}"
    assert thinking_started, "Never received thinking_start"
    assert len(thinking_chunks) > 0, "No thinking content received"
    assert thinking_completed, "Never received thinking_end"
    assert _has_thinking(result), "Final result has no thinking blocks"


# =========================================================================
# 5. Multi-turn with tools (from stream.test.ts :: multiTurn)
# =========================================================================


@skip_no_key
async def test_multi_turn_tool_use():
    """Model should handle multi-turn conversation with tool calls and results."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant that can use tools to answer questions.",
        messages=[
            UserMessage(
                role="user",
                content="Calculate 42 * 17 using the math_operation tool, then tell me the answer.",
                timestamp=_now(),
            )
        ],
        tools=[CALCULATOR_TOOL],
    )

    max_turns = 5
    all_text = ""
    has_tool_calls = False

    for _turn in range(max_turns):
        response = await complete(model, context, _opts())
        context.messages.append(response)

        results = []
        for block in response.content:
            if isinstance(block, TextContent):
                all_text += block.text
            elif isinstance(block, ToolCall):
                has_tool_calls = True
                args = block.arguments
                a, b, op = args.get("a", 0), args.get("b", 0), args.get("operation", "add")
                if op == "multiply":
                    result_val = a * b
                elif op == "add":
                    result_val = a + b
                elif op == "subtract":
                    result_val = a - b
                elif op == "divide":
                    result_val = a / b if b != 0 else 0
                else:
                    result_val = 0

                results.append(
                    ToolResultMessage(
                        role="toolResult",
                        toolCallId=block.id,
                        toolName=block.name,
                        content=[TextContent(text=str(result_val))],
                        isError=False,
                        timestamp=_now(),
                    )
                )
        context.messages.extend(results)

        assert response.stopReason != "error", f"Error: {response.errorMessage}"
        if response.stopReason == "stop":
            break

    assert has_tool_calls, "Model never called a tool"
    assert "714" in all_text, f"Expected 714 in output, got: {all_text[:200]}"


# =========================================================================
# 6. Abort mid-stream (from abort.test.ts :: testAbortSignal)
# =========================================================================


@skip_no_key
async def test_abort_mid_stream():
    """Aborting mid-stream should yield stopReason='aborted'."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(
                role="user",
                content="What is 15 + 27? Think step by step. Then list 50 first names.",
                timestamp=_now(),
            )
        ],
        systemPrompt="You are a helpful assistant.",
    )

    abort_signal = asyncio.Event()
    text = ""

    s = stream(model, context, _opts(signal=abort_signal))
    async for event in s:
        if event.type in ("text_delta", "thinking_delta"):
            text += event.delta
        if len(text) >= 50:
            abort_signal.set()
            break

    msg = await s.result()
    assert msg.stopReason == "aborted", f"Expected aborted, got {msg.stopReason}"
    assert len(msg.content) > 0 or msg.stopReason == "aborted"


# =========================================================================
# 7. Immediate abort (from abort.test.ts :: testImmediateAbort)
# =========================================================================


@skip_no_key
async def test_immediate_abort():
    """Pre-set abort signal should produce stopReason='aborted' immediately."""
    model = _get_model(NON_REASONING_MODEL_ID)
    abort_signal = asyncio.Event()
    abort_signal.set()  # Pre-abort

    context = Context(
        messages=[
            UserMessage(role="user", content="Hello", timestamp=_now())
        ],
    )

    response = await complete(model, context, _opts(signal=abort_signal))
    assert response.stopReason == "aborted"


# =========================================================================
# 8. Abort then new message (from abort.test.ts :: testAbortThenNewMessage)
# =========================================================================


@skip_no_key
async def test_abort_then_new_message():
    """After aborting, a follow-up request with the aborted context should work."""
    model = _get_model(NON_REASONING_MODEL_ID)
    abort_signal = asyncio.Event()
    abort_signal.set()

    context = Context(
        messages=[
            UserMessage(
                role="user",
                content="Hello, how are you?",
                timestamp=_now(),
            )
        ],
    )

    aborted_response = await complete(model, context, _opts(signal=abort_signal))
    assert aborted_response.stopReason == "aborted"

    # Add aborted message and new user turn
    context.messages.append(aborted_response)
    context.messages.append(
        UserMessage(
            role="user",
            content="What is 2 + 2?",
            timestamp=_now(),
        )
    )

    follow_up = await complete(model, context, _opts())
    assert follow_up.stopReason == "stop", f"Error: {follow_up.errorMessage}"
    assert len(follow_up.content) > 0


# =========================================================================
# 9. Token stats on abort (from tokens.test.ts)
# =========================================================================


@skip_no_key
async def test_token_stats_on_abort():
    """OpenAI-completions providers lose usage data on abort (usage comes in final chunk)."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(
                role="user",
                content="Write a long poem with 20 stanzas about the beauty of nature.",
                timestamp=_now(),
            )
        ],
        systemPrompt="You are a helpful assistant.",
    )

    abort_signal = asyncio.Event()
    text = ""

    s = stream(model, context, _opts(signal=abort_signal))
    async for event in s:
        if event.type in ("text_delta", "thinking_delta"):
            text += event.delta
            if len(text) >= 500:
                abort_signal.set()
                break

    msg = await s.result()
    assert msg.stopReason == "aborted"
    # OpenRouter uses openai-completions which sends usage only in the final
    # chunk. When aborted, usage data is lost. This matches the upstream
    # TS test behavior for openai-completions providers.
    assert msg.usage.input == 0
    assert msg.usage.output == 0


# =========================================================================
# 10. Empty content handling (from empty.test.ts)
# =========================================================================


@skip_no_key
async def test_empty_content_array():
    """Sending an empty content array should be handled gracefully."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(role="user", content=[], timestamp=_now())
        ],
    )
    response = await complete(model, context, _opts())
    assert response is not None
    assert response.role == "assistant"
    # May error or succeed -- either is valid graceful handling
    if response.stopReason == "error":
        assert response.errorMessage is not None


@skip_no_key
async def test_empty_string_content():
    """Sending empty string content should be handled gracefully."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(role="user", content="", timestamp=_now())
        ],
    )
    response = await complete(model, context, _opts())
    assert response is not None
    assert response.role == "assistant"
    if response.stopReason == "error":
        assert response.errorMessage is not None


@skip_no_key
async def test_whitespace_only_content():
    """Whitespace-only content should be handled gracefully."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        messages=[
            UserMessage(role="user", content="   \n\t  ", timestamp=_now())
        ],
    )
    response = await complete(model, context, _opts())
    assert response is not None
    assert response.role == "assistant"
    if response.stopReason == "error":
        assert response.errorMessage is not None


@skip_no_key
async def test_empty_assistant_in_conversation():
    """Empty assistant message in conversation should be handled gracefully."""
    model = _get_model(NON_REASONING_MODEL_ID)
    empty_assistant = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage={
            "input": 10,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 10,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        stopReason="stop",
        timestamp=_now(),
    )
    context = Context(
        messages=[
            UserMessage(
                role="user",
                content="Hello, how are you?",
                timestamp=_now(),
            ),
            empty_assistant,
            UserMessage(
                role="user",
                content="Please respond this time.",
                timestamp=_now(),
            ),
        ],
    )
    response = await complete(model, context, _opts())
    assert response is not None
    assert response.role == "assistant"
    if response.stopReason != "error":
        assert len(response.content) > 0


# =========================================================================
# 11. responseId (from responseid.test.ts)
# =========================================================================


@skip_no_key
async def test_response_id_populated():
    """Completed response should have a non-empty responseId."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant. Be concise.",
        messages=[
            UserMessage(
                role="user",
                content="Reply with exactly: response id test",
                timestamp=_now(),
            )
        ],
    )
    response = await complete(model, context, _opts())
    assert response.stopReason != "error", f"Error: {response.errorMessage}"
    assert response.responseId, "responseId should be a non-empty string"
    assert isinstance(response.responseId, str)


# =========================================================================
# 12. Usage stats on successful completion (from stream.test.ts)
# =========================================================================


@skip_no_key
async def test_usage_stats_on_success():
    """Successful completion should report input and output token counts."""
    model = _get_model(NON_REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant. Be concise.",
        messages=[
            UserMessage(
                role="user",
                content="Reply with exactly: 'Hello test successful'",
                timestamp=_now(),
            )
        ],
    )
    response = await complete(model, context, _opts())
    assert response.stopReason != "error", f"Error: {response.errorMessage}"
    assert (response.usage.input + response.usage.cacheRead) > 0, "Expected input tokens > 0"
    assert response.usage.output > 0, "Expected output tokens > 0"


# =========================================================================
# 13. Context overflow detection (from context-overflow.test.ts)
# =========================================================================


@skip_no_key
async def test_context_overflow_detection():
    """Sending input exceeding context window should be detected as overflow."""
    model = _get_model(NON_REASONING_MODEL_ID)

    # Generate content that exceeds the context window
    target_tokens = model.contextWindow + 10_000
    target_chars = int(target_tokens * 4 * 1.5)
    repetitions = (target_chars // len(LOREM_IPSUM)) + 1
    overflow_content = LOREM_IPSUM * repetitions

    context = Context(
        systemPrompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                role="user",
                content=overflow_content,
                timestamp=_now(),
            )
        ],
    )

    response = await complete(model, context, _opts())

    # OpenRouter normalizes overflow errors to a standard pattern
    assert response.stopReason == "error", (
        f"Expected error stopReason for overflow, got {response.stopReason}"
    )
    assert response.errorMessage is not None
    assert is_context_overflow(response, model.contextWindow), (
        f"isContextOverflow should return True. errorMessage: {response.errorMessage}"
    )


# =========================================================================
# 14. Reasoning model basic completion (from stream.test.ts)
# =========================================================================


@skip_no_key
async def test_reasoning_model_basic_completion():
    """Reasoning model should complete successfully with thinking content."""
    model = _get_model(REASONING_MODEL_ID)
    context = Context(
        systemPrompt="You are a helpful assistant. Be concise.",
        messages=[
            UserMessage(
                role="user",
                content="What is 7 * 8? Answer briefly.",
                timestamp=_now(),
            )
        ],
    )
    response = await complete(model, context, _opts(reasoningEffort="high"))
    assert response.stopReason == "stop", f"Error: {response.errorMessage}"
    text = _text_of(response)
    assert "56" in text, f"Expected 56 in output, got: {text[:200]}"
    assert _has_thinking(response), "Reasoning model should produce thinking blocks"
