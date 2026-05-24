from __future__ import annotations

import asyncio
import time

import pytest
from harnify_agent import Agent
from harnify_ai.providers.faux import faux_assistant_message, faux_text, faux_thinking, register_faux_provider
from harnify_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
)
from harnify_ai.utils.event_stream import AssistantMessageEventStream

ZERO_USAGE = {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 0,
    "cost": {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "total": 0,
    },
}


def _assistant_text(message: AssistantMessage) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


def _make_model(*, model_id: str, name: str, api: str, provider: str) -> Model:
    return Model(
        id=model_id,
        name=name,
        api=api,
        provider=provider,
        baseUrl="",
        reasoning=False,
        input=[],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=0,
        maxTokens=0,
    )


def _make_user_message(text: str, *, timestamp: int | None = None) -> dict[str, object]:
    return {
        "role": "user",
        "content": [TextContent(text=text)],
        "timestamp": timestamp or int(time.time() * 1000),
    }


def _make_partial_assistant(model: Model, text: str = "") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=ZERO_USAGE,
        stopReason="stop",
        timestamp=int(time.time() * 1000),
    )


@pytest.mark.asyncio
async def test_agent_default_state_and_custom_initial_state() -> None:
    default_agent = Agent()

    assert default_agent.state.systemPrompt == ""
    assert default_agent.state.model.id == "unknown"
    assert default_agent.state.thinkingLevel == "off"
    assert default_agent.state.tools == []
    assert default_agent.state.messages == []
    assert default_agent.state.isStreaming is False
    assert default_agent.state.streamingMessage is None
    assert default_agent.state.pendingToolCalls == set()
    assert default_agent.state.errorMessage is None

    registration = register_faux_provider({"api": "faux-agent-initial-state"})
    model = registration.get_model()
    assert model is not None

    custom_agent = Agent(
        initialState={
            "systemPrompt": "You are helpful.",
            "model": model,
            "thinkingLevel": "low",
            "messages": [_make_user_message("history")],
        }
    )

    assert custom_agent.state.systemPrompt == "You are helpful."
    assert custom_agent.state.model == model
    assert custom_agent.state.thinkingLevel == "low"
    assert len(custom_agent.state.messages) == 1

    registration.unregister()


@pytest.mark.asyncio
async def test_agent_subscribers_are_ordered_and_unsubscribe_works() -> None:
    registration = register_faux_provider({"api": "faux-agent-subscribe", "tokenSize": {"min": 32, "max": 32}})
    registration.set_responses([faux_assistant_message([faux_thinking("plan"), faux_text("answer")])])

    model = registration.get_model()
    assert model is not None
    agent = Agent(initialState={"model": model})

    calls: list[tuple[str, str]] = []
    unsubscribe_first = agent.subscribe(lambda event, _signal: calls.append(("first", event.type)))
    agent.subscribe(lambda event, _signal: calls.append(("second", event.type)))

    await agent.prompt("hello")

    assert calls[:6] == [
        ("first", "agent_start"),
        ("second", "agent_start"),
        ("first", "turn_start"),
        ("second", "turn_start"),
        ("first", "message_start"),
        ("second", "message_start"),
    ]
    assert calls[-2:] == [("first", "agent_end"), ("second", "agent_end")]

    unsubscribe_first()
    calls.clear()
    registration.set_responses([faux_assistant_message("next")])

    await agent.prompt("again")

    assert all(listener == "second" for listener, _ in calls)

    registration.unregister()


@pytest.mark.asyncio
async def test_agent_prompt_streams_faux_provider_through_public_api() -> None:
    registration = register_faux_provider({"api": "faux-agent-prompt", "tokenSize": {"min": 32, "max": 32}})
    registration.set_responses([faux_assistant_message([faux_thinking("reason"), faux_text("final answer")])])

    model = registration.get_model()
    assert model is not None
    agent = Agent(initialState={"model": model})

    event_types: list[str] = []
    agent_end_messages: list[str] = []

    def listener(event, _signal) -> None:
        event_types.append(event.type)
        if event.type == "agent_end":
            agent_end_messages.extend(getattr(message, "role", "unknown") for message in event.messages)

    agent.subscribe(listener)

    await agent.prompt("hello")

    assert event_types[:4] == ["agent_start", "turn_start", "message_start", "message_end"]
    assert "message_update" in event_types
    assert event_types[-3:] == ["message_end", "turn_end", "agent_end"]
    assert [message.role for message in agent.state.messages] == ["user", "assistant"]
    assistant_message = agent.state.messages[-1]
    assert isinstance(assistant_message, AssistantMessage)
    assert assistant_message.stopReason == "stop"
    assert [block.type for block in assistant_message.content] == ["thinking", "text"]
    assert _assistant_text(assistant_message) == "final answer"
    assert agent_end_messages == ["user", "assistant"]

    registration.unregister()


@pytest.mark.asyncio
async def test_agent_wait_for_idle_awaits_async_subscribers() -> None:
    registration = register_faux_provider({"api": "faux-agent-idle", "tokenSize": {"min": 32, "max": 32}})
    registration.set_responses([faux_assistant_message("done")])

    model = registration.get_model()
    assert model is not None
    agent = Agent(initialState={"model": model})
    barrier = asyncio.Event()
    listener_finished = False

    async def listener(event, _signal) -> None:
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier.wait()
            listener_finished = True

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.sleep(0)
    idle_task = asyncio.create_task(agent.waitForIdle())
    await asyncio.sleep(0)

    assert prompt_task.done() is False
    assert idle_task.done() is False
    assert agent.state.isStreaming is True
    assert listener_finished is False

    barrier.set()
    await asyncio.gather(prompt_task, idle_task)

    assert listener_finished is True
    assert agent.state.isStreaming is False

    registration.unregister()


@pytest.mark.asyncio
async def test_agent_abort_forwards_signal_to_stream_and_subscribers() -> None:
    received_stream_signal = None
    received_listener_signal = None

    def stream_fn(model: Model, _context, options=None) -> AssistantMessageEventStream:
        nonlocal received_stream_signal
        received_stream_signal = None if options is None else options.signal
        stream = AssistantMessageEventStream()

        async def run() -> None:
            partial = _make_partial_assistant(model)
            stream.push(StartEvent(partial=partial))
            while received_stream_signal is not None and not received_stream_signal.aborted:
                await asyncio.sleep(0.01)
            aborted = partial.model_copy(
                update={
                    "stopReason": "aborted",
                    "errorMessage": "Request was aborted",
                    "timestamp": int(time.time() * 1000),
                }
            )
            stream.push(ErrorEvent(reason="aborted", error=aborted))

        asyncio.create_task(run())
        return stream

    abort_model = {
        "id": "abort-model",
        "name": "Abort Model",
        "api": "abort-api",
        "provider": "abort-provider",
        "baseUrl": "",
        "reasoning": False,
        "input": [],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 0,
        "maxTokens": 0,
    }
    agent = Agent(initialState={"model": abort_model}, streamFn=stream_fn)

    def listener(event, signal) -> None:
        nonlocal received_listener_signal
        if event.type == "agent_start":
            received_listener_signal = signal

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.sleep(0.02)

    assert received_stream_signal is not None
    assert received_listener_signal is not None
    assert received_listener_signal.aborted is False

    agent.abort()
    await prompt_task

    assert received_stream_signal.aborted is True
    assert received_listener_signal.aborted is True
    assistant_message = agent.state.messages[-1]
    assert isinstance(assistant_message, AssistantMessage)
    assert assistant_message.stopReason == "aborted"
    assert agent.state.errorMessage == "Request was aborted"


@pytest.mark.asyncio
async def test_agent_failure_path_and_session_id_forwarding() -> None:
    seen_session_ids: list[str | None] = []

    def stream_fn(model: Model, _context, options=None) -> AssistantMessageEventStream:
        seen_session_ids.append(None if options is None else options.sessionId)
        raise RuntimeError("provider exploded")

    model = _make_model(
        model_id="failure-model",
        name="Failure Model",
        api="failure-api",
        provider="failure-provider",
    )
    agent = Agent(initialState={"model": model}, sessionId="session-abc", streamFn=stream_fn)

    event_types: list[str] = []
    agent.subscribe(lambda event, _signal: event_types.append(event.type))

    await agent.prompt("hello")

    assert seen_session_ids == ["session-abc"]
    assert event_types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assistant_message = agent.state.messages[-1]
    assert isinstance(assistant_message, AssistantMessage)
    assert assistant_message.stopReason == "error"
    assert assistant_message.errorMessage == "provider exploded"
    assert agent.state.errorMessage == "provider exploded"


@pytest.mark.asyncio
async def test_agent_rejects_parallel_prompt_and_preserves_queued_messages() -> None:
    completion = asyncio.Event()

    def stream_fn(model: Model, _context, options=None) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def run() -> None:
            partial = _make_partial_assistant(model)
            stream.push(StartEvent(partial=partial))
            await completion.wait()
            done = partial.model_copy(update={"content": [TextContent(text="ok")]})
            stream.push(DoneEvent(reason="stop", message=done))

        asyncio.create_task(run())
        return stream

    model = _make_model(
        model_id="busy-model",
        name="Busy Model",
        api="busy-api",
        provider="busy-provider",
    )
    agent = Agent(initialState={"model": model}, streamFn=stream_fn)

    prompt_task = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.02)

    steering = _make_user_message("steer", timestamp=123)
    follow_up = _make_user_message("follow-up", timestamp=124)
    agent.steer(steering)
    agent.followUp(follow_up)

    with pytest.raises(RuntimeError, match="Agent is already processing a prompt"):
        await agent.prompt("second")

    assert agent.hasQueuedMessages() is True
    assert steering not in agent.state.messages
    assert follow_up not in agent.state.messages

    completion.set()
    await prompt_task
