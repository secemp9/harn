from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from harnify_agent import AbortController, Agent, AgentTool, AgentToolResult
from harnify_agent.agent_loop import run_agent_loop, run_agent_loop_continue
from harnify_agent.types import AgentContext, AgentEvent, AgentLoopConfig, AgentMessage
from harnify_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    ToolCall,
    UserMessage,
)
from harnify_ai.utils.event_stream import AssistantMessageEventStream

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def create_model() -> Model:
    return Model(
        id="mock",
        name="Mock",
        api="openai-responses",
        provider="openai",
        baseUrl="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=8192,
        maxTokens=2048,
    )


def create_usage() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def create_user_message(text: str) -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=int(time.time() * 1000))


def create_assistant_message(
    content: list[TextContent | ToolCall],
    stop_reason: str = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stopReason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def identity_converter(messages: list[AgentMessage]):
    return [message for message in messages if getattr(message, "role", None) in {"user", "assistant", "toolResult"}]


def create_context(*, tools: list[AgentTool] | None = None, messages: list[AgentMessage] | None = None) -> AgentContext:
    return AgentContext(systemPrompt="", messages=messages or [], tools=tools)


async def collect_events(event: AgentEvent, sink: list[AgentEvent]) -> None:
    sink.append(event)


@pytest.mark.asyncio
async def test_run_agent_loop_executes_tool_call_and_continues() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
        )

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=TOOL_SCHEMA,
        execute=execute,
    )
    context = create_context(tools=[tool])
    config = AgentLoopConfig(model=create_model(), convertToLlm=identity_converter)

    call_index = 0

    def stream_fn(_model, _context, _options=None):
        nonlocal call_index
        stream = AssistantMessageEventStream()

        async def run() -> None:
            nonlocal call_index
            if call_index == 0:
                message = create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})],
                    "toolUse",
                )
                stream.push(DoneEvent(reason="toolUse", message=message))
            else:
                stream.push(DoneEvent(reason="stop", message=create_assistant_message([TextContent(text="done")])))
            call_index += 1

        asyncio.create_task(run())
        return stream

    events: list[AgentEvent] = []
    messages = await run_agent_loop(
        [create_user_message("echo something")],
        context,
        config,
        lambda event: collect_events(event, events),
        None,
        stream_fn,
    )

    assert executed == ["hello"]
    assert [message.role for message in messages] == ["user", "assistant", "toolResult", "assistant"]
    assert any(event.type == "tool_execution_start" for event in events)
    assert any(event.type == "tool_execution_end" for event in events)


@pytest.mark.asyncio
async def test_parallel_tool_execution_emits_end_in_completion_order() -> None:
    first_done = asyncio.Event()
    first_finished = False
    parallel_observed = False

    async def execute(_tool_call_id: str, params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        nonlocal first_finished, parallel_observed
        if params["value"] == "first":
            await first_done.wait()
            first_finished = True
        if params["value"] == "second" and not first_finished:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
        )

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=TOOL_SCHEMA,
        execute=execute,
        executionMode="parallel",
    )
    context = create_context(tools=[tool])
    config = AgentLoopConfig(
        model=create_model(),
        convertToLlm=identity_converter,
        toolExecution="parallel",
    )

    call_index = 0

    def stream_fn(_model, _context, _options=None):
        nonlocal call_index
        stream = AssistantMessageEventStream()

        async def run() -> None:
            nonlocal call_index
            if call_index == 0:
                message = create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    "toolUse",
                )
                stream.push(DoneEvent(reason="toolUse", message=message))
                asyncio.get_running_loop().call_later(0.02, first_done.set)
            else:
                stream.push(DoneEvent(reason="stop", message=create_assistant_message([TextContent(text="done")])))
            call_index += 1

        asyncio.create_task(run())
        return stream

    events: list[AgentEvent] = []
    await run_agent_loop(
        [create_user_message("echo both")],
        context,
        config,
        lambda event: collect_events(event, events),
        None,
        stream_fn,
    )

    tool_execution_end_ids = [
        event.toolCallId
        for event in events
        if event.type == "tool_execution_end"
    ]
    tool_result_ids = [
        event.message.toolCallId
        for event in events
        if event.type == "message_end" and event.message.role == "toolResult"
    ]

    assert parallel_observed is True
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    assert tool_result_ids == ["tool-1", "tool-2"]


@pytest.mark.asyncio
async def test_steering_messages_are_injected_after_tool_results() -> None:
    executed: list[str] = []

    async def execute(_tool_call_id: str, params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"ok:{params['value']}")],
            details={"value": params["value"]},
        )

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters=TOOL_SCHEMA,
        execute=execute,
    )
    context = create_context(tools=[tool])
    queued_message = create_user_message("interrupt")
    delivered = False
    call_index = 0
    saw_interrupt_in_context = False

    async def get_steering_messages() -> list[AgentMessage]:
        nonlocal delivered
        if executed and not delivered:
            delivered = True
            return [queued_message]
        return []

    config = AgentLoopConfig(
        model=create_model(),
        convertToLlm=identity_converter,
        toolExecution="sequential",
        getSteeringMessages=get_steering_messages,
    )

    def stream_fn(_model, llm_context, _options=None):
        nonlocal call_index, saw_interrupt_in_context
        stream = AssistantMessageEventStream()

        async def run() -> None:
            nonlocal call_index, saw_interrupt_in_context
            if call_index == 1:
                saw_interrupt_in_context = any(
                    message.role == "user" and message.content == "interrupt"
                    for message in llm_context.messages
                )
            if call_index == 0:
                message = create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    "toolUse",
                )
                stream.push(DoneEvent(reason="toolUse", message=message))
            else:
                stream.push(DoneEvent(reason="stop", message=create_assistant_message([TextContent(text="done")])))
            call_index += 1

        asyncio.create_task(run())
        return stream

    events: list[AgentEvent] = []
    await run_agent_loop(
        [create_user_message("start")],
        context,
        config,
        lambda event: collect_events(event, events),
        None,
        stream_fn,
    )

    event_sequence = []
    for event in events:
        if event.type != "message_start":
            continue
        if event.message.role == "toolResult":
            event_sequence.append(f"tool:{event.message.toolCallId}")
        if event.message.role == "user" and event.message.content == "interrupt":
            event_sequence.append("interrupt")

    assert executed == ["first", "second"]
    assert saw_interrupt_in_context is True
    assert event_sequence.index("tool:tool-1") < event_sequence.index("interrupt")
    assert event_sequence.index("tool:tool-2") < event_sequence.index("interrupt")


@pytest.mark.asyncio
async def test_follow_up_messages_continue_when_agent_would_otherwise_stop() -> None:
    follow_up_delivered = False
    llm_calls = 0

    async def get_follow_up_messages() -> list[AgentMessage]:
        nonlocal follow_up_delivered
        if not follow_up_delivered:
            follow_up_delivered = True
            return [create_user_message("follow-up")]
        return []

    config = AgentLoopConfig(
        model=create_model(),
        convertToLlm=identity_converter,
        getFollowUpMessages=get_follow_up_messages,
    )
    context = create_context()

    def stream_fn(_model, llm_context, _options=None):
        nonlocal llm_calls
        stream = AssistantMessageEventStream()

        async def run() -> None:
            nonlocal llm_calls
            if llm_calls == 1:
                assert any(
                    message.role == "user" and message.content == "follow-up"
                    for message in llm_context.messages
                )
            stream.push(
                DoneEvent(
                    reason="stop",
                    message=create_assistant_message([TextContent(text=f"answer {llm_calls + 1}")]),
                )
            )
            llm_calls += 1

        asyncio.create_task(run())
        return stream

    messages = await run_agent_loop(
        [create_user_message("start")],
        context,
        config,
        lambda _event: None,
        None,
        stream_fn,
    )

    assert llm_calls == 2
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_run_agent_loop_continue_skips_prompt_message_events() -> None:
    config = AgentLoopConfig(model=create_model(), convertToLlm=identity_converter)
    context = create_context(messages=[create_user_message("hello")])

    def stream_fn(_model, _context, _options=None):
        stream = AssistantMessageEventStream()

        async def run() -> None:
            stream.push(DoneEvent(reason="stop", message=create_assistant_message([TextContent(text="response")])))

        asyncio.create_task(run())
        return stream

    events: list[AgentEvent] = []
    messages = await run_agent_loop_continue(
        context,
        config,
        lambda event: collect_events(event, events),
        None,
        stream_fn,
    )

    message_end_events = [event for event in events if event.type == "message_end"]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"
    assert [message.role for message in messages] == ["assistant"]


@pytest.mark.asyncio
async def test_abort_terminates_loop_cleanly() -> None:
    controller = AbortController()
    config = AgentLoopConfig(model=create_model(), convertToLlm=identity_converter)
    context = create_context()

    def stream_fn(model: Model, _context, options=None):
        signal = None if options is None else options.signal
        stream = AssistantMessageEventStream()

        async def run() -> None:
            partial = AssistantMessage(
                role="assistant",
                content=[TextContent(text="")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=create_usage(),
                stopReason="stop",
                timestamp=int(time.time() * 1000),
            )
            stream.push(StartEvent(partial=partial))
            while signal is not None and not signal.aborted:
                await asyncio.sleep(0.01)
            stream.push(
                ErrorEvent(
                    reason="aborted",
                    error=partial.model_copy(
                        update={
                            "stopReason": "aborted",
                            "errorMessage": "Request was aborted",
                            "timestamp": int(time.time() * 1000),
                        }
                    ),
                )
            )

        asyncio.create_task(run())
        return stream

    events: list[AgentEvent] = []

    async def run_loop() -> list[AgentMessage]:
        return await run_agent_loop(
            [create_user_message("hello")],
            context,
            config,
            lambda event: collect_events(event, events),
            controller.signal,
            stream_fn,
        )

    task = asyncio.create_task(run_loop())
    await asyncio.sleep(0.02)
    controller.abort()
    messages = await task

    assert messages[-1].role == "assistant"
    assert messages[-1].stopReason == "aborted"
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_agent_continue_consumes_follow_up_queue_from_assistant_tail() -> None:
    responses = [
        create_assistant_message([TextContent(text="initial")]),
        create_assistant_message([TextContent(text="follow-up answer")]),
    ]
    call_index = 0

    def stream_fn(_model, _context, _options=None):
        nonlocal call_index
        stream = AssistantMessageEventStream()

        async def run() -> None:
            nonlocal call_index
            stream.push(DoneEvent(reason="stop", message=responses[call_index]))
            call_index += 1

        asyncio.create_task(run())
        return stream

    agent = Agent(initialState={"model": create_model()}, streamFn=stream_fn)
    await agent.prompt("hello")
    agent.followUp(create_user_message("queued follow-up"))
    await agent.continue_()

    assert [message.role for message in agent.state.messages] == ["user", "assistant", "user", "assistant"]
