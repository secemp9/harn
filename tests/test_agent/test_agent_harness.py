from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from harnify_agent import AgentTool, AgentToolResult
from harnify_agent.harness.agent_harness import AgentHarness
from harnify_agent.harness.env.local import NodeExecutionEnv
from harnify_agent.harness.session.jsonl_storage import JsonlSessionStorage
from harnify_agent.harness.session.memory_storage import InMemorySessionStorage
from harnify_agent.harness.session.session import Session
from harnify_agent.harness.types import PromptTemplate, Skill
from harnify_ai.providers.faux import faux_assistant_message, faux_tool_call, register_faux_provider
from harnify_ai.types import TextContent

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
    "additionalProperties": False,
}

registrations: list[Any] = []


class AppSkill(Skill):
    source: str


class AppPromptTemplate(PromptTemplate):
    source: str | None = None


def text_from_user_messages(messages: list[Any]) -> list[str]:
    results: list[str] = []
    for message in messages:
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            results.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, "type", None) == "text" and isinstance(getattr(part, "text", None), str):
                results.append(part.text)
    return results


def deferred() -> tuple[asyncio.Future[None], callable]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def resolve() -> None:
        if not future.done():
            future.set_result(None)

    return future, resolve


def get_reasoning(options: Any) -> Any:
    return getattr(options, "reasoning", None) if options is not None else None


def count_user_response(sink: list[int], text: str):
    def responder(context, _options, _state, _model):
        sink.append(len([message for message in context.messages if message.role == "user"]))
        return faux_assistant_message(text)

    return responder


def extend_user_text_response(sink: list[str], text: str):
    def responder(context, _options, _state, _model):
        sink.extend(text_from_user_messages(context.messages))
        return faux_assistant_message(text)

    return responder


def calculate_tool() -> AgentTool:
    async def execute(_tool_call_id: str, params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        expression = params["expression"]
        result = str(eval(expression, {"__builtins__": {}}, {}))
        return AgentToolResult(content=[TextContent(text=result)], details={"result": result})

    return AgentTool(
        name="calculate",
        label="Calculate",
        description="Evaluate simple arithmetic expressions",
        parameters=TOOL_SCHEMA,
        execute=execute,
    )


def get_current_time_tool() -> AgentTool:
    async def execute(_tool_call_id: str, _params: Any, _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="12:00:00")], details={"ok": True})

    return AgentTool(
        name="get_current_time",
        label="Current Time",
        description="Return a deterministic test time",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=execute,
    )


def create_harness(**kwargs: Any) -> AgentHarness:
    return AgentHarness(kwargs)


def create_env(cwd: Path) -> NodeExecutionEnv:
    return NodeExecutionEnv({"cwd": str(cwd)})


def resource_system_prompt(payload: dict[str, Any]) -> str:
    skills = payload["resources"].skills or []
    return skills[0].content if skills else "missing"


@pytest.fixture(autouse=True)
def cleanup_registrations() -> None:
    yield
    while registrations:
        registrations.pop().unregister()


@pytest.mark.asyncio
async def test_agent_harness_getters_and_queue_modes(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    model = registration.get_model()
    assert model is not None
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=model,
        thinkingLevel="high",
        systemPrompt="You are helpful.",
        steeringMode="all",
        followUpMode="all",
    )

    assert harness.env.cwd == str(tmp_path)
    assert harness.getModel() == model
    assert harness.getThinkingLevel() == "high"
    assert harness.getSteeringMode() == "all"
    assert harness.getFollowUpMode() == "all"

    await harness.setSteeringMode("one-at-a-time")
    await harness.setFollowUpMode("one-at-a-time")
    assert harness.getSteeringMode() == "one-at-a-time"
    assert harness.getFollowUpMode() == "one-at-a-time"


@pytest.mark.asyncio
async def test_agent_harness_drains_steering_queue_one_at_a_time(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    user_counts: list[int] = []
    registration.set_responses(
        [
            count_user_response(user_counts, "first"),
            count_user_response(user_counts, "second"),
            count_user_response(user_counts, "third"),
        ]
    )
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
        steeringMode="one-at-a-time",
    )
    steer_queue_lengths: list[int] = []
    queued = False

    async def listener(event, _signal) -> None:
        nonlocal queued
        if event.type == "queue_update":
            steer_queue_lengths.append(len(event.steer))
        if event.type == "message_start" and getattr(event.message, "role", None) == "assistant" and not queued:
            queued = True
            await harness.steer("one")
            await harness.steer("two")

    harness.subscribe(listener)
    await harness.prompt("hello")

    assert user_counts == [1, 2, 3]
    assert steer_queue_lengths == [1, 2, 1, 0]


@pytest.mark.asyncio
async def test_before_agent_start_messages_are_persisted(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    request_text: list[str] = []
    registration.set_responses([extend_user_text_response(request_text, "ok")])
    session = Session(InMemorySessionStorage())
    harness = create_harness(env=create_env(tmp_path), session=session, model=registration.get_model())
    harness.on(
        "before_agent_start",
        lambda _event: {
            "messages": [{"role": "user", "content": [TextContent(text="hook")], "timestamp": int(time.time() * 1000)}]
        },
    )

    await harness.prompt("hello")

    persisted_text: list[str] = []
    for entry in await session.getEntries():
        if entry["type"] != "message" or getattr(entry["message"], "role", None) != "user":
            continue
        persisted_text.extend(text_from_user_messages([entry["message"]]))
    assert request_text == ["hello", "hook"]
    assert persisted_text == ["hello", "hook"]


@pytest.mark.asyncio
async def test_abort_clears_steer_and_follow_up_but_preserves_next_turn(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    release = asyncio.Event()
    observed_signal = None
    second_request_text: list[str] = []

    async def first_response(_context, options, _state, _model):
        nonlocal observed_signal
        observed_signal = options.signal
        await release.wait()
        return faux_assistant_message("aborted-ish")

    registration.set_responses(
        [
            first_response,
            extend_user_text_response(second_request_text, "second"),
        ]
    )
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
    )
    queue_updates: list[tuple[int, int, int]] = []

    def listener(event, _signal) -> None:
        if event.type == "queue_update":
            queue_updates.append((len(event.steer), len(event.followUp), len(event.nextTurn)))

    harness.subscribe(listener)

    first_prompt = asyncio.create_task(harness.prompt("first"))
    await asyncio.sleep(0)
    await harness.steer("steer")
    await harness.followUp("follow")
    await harness.nextTurn("next")
    abort_result_task = asyncio.create_task(harness.abort())
    await asyncio.sleep(0)
    assert observed_signal.aborted is True
    release.set()
    abort_result = await abort_result_task
    await first_prompt
    await harness.prompt("second")

    assert len(abort_result.clearedSteer) == 1
    assert len(abort_result.clearedFollowUp) == 1
    assert (0, 0, 1) in queue_updates
    assert second_request_text == ["first", "next", "second"]


@pytest.mark.asyncio
async def test_follow_up_queue_drains_one_at_a_time(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    user_counts: list[int] = []
    registration.set_responses(
        [
            count_user_response(user_counts, "first"),
            count_user_response(user_counts, "second"),
            count_user_response(user_counts, "third"),
        ]
    )
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
        followUpMode="one-at-a-time",
    )
    follow_up_queue_lengths: list[int] = []
    queued = False

    async def listener(event, _signal) -> None:
        nonlocal queued
        if event.type == "queue_update":
            follow_up_queue_lengths.append(len(event.followUp))
        if event.type == "message_start" and getattr(event.message, "role", None) == "assistant" and not queued:
            queued = True
            await harness.followUp("one")
            await harness.followUp("two")

    harness.subscribe(listener)
    await harness.prompt("hello")

    assert user_counts == [1, 2, 3]
    assert follow_up_queue_lengths == [1, 2, 1, 0]


@pytest.mark.asyncio
async def test_hook_failures_settle_with_persisted_error_message(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    registration.set_responses([faux_assistant_message("unused")])
    session = Session(InMemorySessionStorage())
    harness = create_harness(env=create_env(tmp_path), session=session, model=registration.get_model())
    events: list[str] = []
    harness.subscribe(lambda event, _signal: events.append(event.type))
    harness.on("context", lambda _event: (_ for _ in ()).throw(RuntimeError("context exploded")))

    response = await harness.prompt("hello")
    second = await harness.prompt("after failure")

    entries = await session.getEntries()
    messages = [entry["message"] for entry in entries if entry["type"] == "message"]
    assert response.stopReason == "error"
    assert response.errorMessage == "context exploded"
    assert second.role == "assistant"
    assert getattr(messages[0], "role", None) == "user"
    assert getattr(messages[1], "stopReason", None) == "error"
    assert getattr(messages[1], "errorMessage", None) == "context exploded"
    assert "agent_end" in events
    assert "settled" in events


@pytest.mark.asyncio
async def test_save_points_refresh_model_thinking_resources_and_tools(tmp_path: Path) -> None:
    registration = register_faux_provider(
        {"models": [{"id": "first", "reasoning": True}, {"id": "second", "reasoning": True}]}
    )
    registrations.append(registration)
    second_model = registration.get_model("second")
    assert second_model is not None
    calculate = calculate_tool()
    current_time = get_current_time_tool()
    captured: list[dict[str, Any]] = []
    registration.set_responses(
        [
            lambda context, options, _state, model: (
                captured.append(
                    {
                        "modelId": model.id,
                        "reasoning": get_reasoning(options),
                        "systemPrompt": context.systemPrompt or "",
                        "tools": [tool.name for tool in context.tools or []],
                    }
                ),
                faux_assistant_message(
                    faux_tool_call("calculate", {"expression": "1 + 1"}, options={"id": "call-1"}),
                    stop_reason="toolUse",
                ),
            )[-1],
            lambda context, options, _state, model: (
                captured.append(
                    {
                        "modelId": model.id,
                        "reasoning": get_reasoning(options),
                        "systemPrompt": context.systemPrompt or "",
                        "tools": [tool.name for tool in context.tools or []],
                    }
                ),
                faux_assistant_message("done"),
            )[-1],
        ]
    )
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
        thinkingLevel="off",
        resources={
            "skills": [
                Skill(
                    name="prompt",
                    description="prompt",
                    content="first prompt",
                    filePath="/skills/prompt",
                )
            ]
        },
        systemPrompt=resource_system_prompt,
        tools=[calculate],
    )

    async def listener(event, _signal) -> None:
        if event.type == "tool_execution_start":
            await harness.setModel(second_model)
            await harness.setThinkingLevel("high")
            await harness.setResources(
                {
                    "skills": [
                        Skill(
                            name="prompt",
                            description="prompt",
                            content="second prompt",
                            filePath="/skills/prompt",
                        )
                    ]
                }
            )
            await harness.setTools([calculate, current_time], [current_time.name])

    harness.subscribe(listener)
    await harness.prompt("hello")

    assert captured == [
        {"modelId": "first", "reasoning": None, "systemPrompt": "first prompt", "tools": ["calculate"]},
        {
            "modelId": "second",
            "reasoning": "high",
            "systemPrompt": "second prompt",
            "tools": ["get_current_time"],
        },
    ]


@pytest.mark.asyncio
async def test_listener_pending_session_writes_flush_after_agent_messages(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    registration.set_responses([faux_assistant_message("ok")])
    session = Session(InMemorySessionStorage())
    harness = create_harness(env=create_env(tmp_path), session=session, model=registration.get_model())
    wrote_pending = False

    async def listener(event, _signal) -> None:
        nonlocal wrote_pending
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant" and not wrote_pending:
            wrote_pending = True
            await harness.appendMessage(
                {
                    "role": "custom",
                    "customType": "listener",
                    "content": "listener write",
                    "display": True,
                    "timestamp": int(time.time() * 1000),
                }
            )

    harness.subscribe(listener)
    await harness.prompt("hello")

    roles = [
        getattr(entry["message"], "role", None)
        for entry in await session.getEntries()
        if entry["type"] == "message"
    ]
    assert roles == ["user", "assistant", "custom"]


@pytest.mark.asyncio
async def test_wait_for_idle_waits_for_async_subscribers(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    registration.set_responses([faux_assistant_message("ok")])
    barrier, resolve = deferred()
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
    )
    listener_finished = False

    async def listener(event, _signal) -> None:
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier
            listener_finished = True

    harness.subscribe(listener)
    prompt_task = asyncio.create_task(harness.prompt("hello"))
    idle_task = asyncio.create_task(harness.waitForIdle())
    await asyncio.sleep(0.01)
    assert prompt_task.done() is False
    assert idle_task.done() is False
    assert listener_finished is False
    resolve()
    await asyncio.gather(prompt_task, idle_task)
    assert listener_finished is True


@pytest.mark.asyncio
async def test_tool_call_and_tool_result_hooks_and_stream_configuration(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    calculate = calculate_tool()
    captured_options: list[Any] = []
    final_payload = None
    seen_tool_calls: list[dict[str, Any]] = []
    async def first_response(_context, options, _state, model):
        nonlocal final_payload
        captured_options.append(options.model_dump())
        if options.onPayload is not None:
            final_payload = await options.onPayload({"steps": ["provider"]}, model)
        return faux_assistant_message(
            faux_tool_call("calculate", {"expression": "2 + 2"}, options={"id": "call-1"}),
            stop_reason="toolUse",
        )

    registration.set_responses(
        [
            first_response,
            lambda _context, options, _state, model: (
                captured_options.append(options.model_dump()),
                faux_assistant_message("done"),
            )[-1],
        ]
    )
    harness = create_harness(
        env=create_env(tmp_path),
        session=Session(InMemorySessionStorage()),
        model=registration.get_model(),
        tools=[calculate],
        streamOptions={
            "timeoutMs": 1000,
            "maxRetries": 2,
            "maxRetryDelayMs": 3000,
            "headers": {"x-base": "base"},
            "metadata": {"base": True},
            "cacheRetention": "none",
        },
        getApiKeyAndHeaders=lambda _model: {"apiKey": "secret", "headers": {"x-auth": "auth"}},
    )
    harness.on(
        "before_provider_request",
        lambda event: {
            "streamOptions": {
                "headers": {"x-hook": "hook", "remove": None},
                "metadata": {"hook": True},
            }
        },
    )
    harness.on(
        "before_provider_payload",
        lambda event: {"payload": {"steps": [*(event.payload.get("steps", [])), "hook"]}},
    )
    harness.on(
        "tool_call",
        lambda event: seen_tool_calls.append(
            {"id": event.toolCallId, "name": event.toolName, "expression": event.input["expression"]}
        ),
    )
    harness.on(
        "tool_result",
        lambda _event: {
            "content": [TextContent(text="patched result")],
            "details": {"patched": True},
        },
    )

    async def subscriber(event, _signal) -> None:
        if event.type == "tool_execution_start":
            await harness.setStreamOptions({"timeoutMs": 2000, "headers": {"x-base": "second"}})

    harness.subscribe(subscriber)
    await harness.prompt("hello")

    entries = await harness.session.getEntries()
    tool_result = next(
        entry
        for entry in entries
        if entry["type"] == "message" and getattr(entry["message"], "role", None) == "toolResult"
    )
    assert captured_options[0]["apiKey"] == "secret"
    assert captured_options[0]["headers"] == {"x-base": "base", "x-auth": "auth", "x-hook": "hook"}
    assert captured_options[0]["timeoutMs"] == 1000
    assert captured_options[1]["timeoutMs"] == 2000
    assert final_payload == {"steps": ["provider", "hook"]}
    assert seen_tool_calls == [{"id": "call-1", "name": "calculate", "expression": "2 + 2"}]
    assert tool_result["message"].content[0].text == "patched result"
    assert tool_result["message"].details == {"patched": True}


@pytest.mark.asyncio
async def test_resources_update_copies_lists_and_store_parity(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    registration.set_responses([faux_assistant_message("ok"), faux_assistant_message("ok")])
    resources = {
        "skills": [
            Skill(
                name="inspect",
                description="Inspect",
                content="Inspect",
                filePath="/skills/inspect",
                disableModelInvocation=False,
            )
        ],
        "promptTemplates": [PromptTemplate(name="review", description="Review", content="Review $1")],
    }
    updates: list[tuple[int, int]] = []

    async def run_suite(session: Session, env: NodeExecutionEnv) -> None:
        harness = create_harness(env=env, session=session, model=registration.get_model())
        harness.subscribe(
            lambda event, _signal: updates.append(
                (
                    len(event.resources.skills or []),
                    len(event.previousResources.skills or []),
                )
            )
            if event.type == "resources_update"
            else None
        )
        await harness.setResources(resources)
        resolved = harness.getResources()
        assert resolved.skills is not resources["skills"]
        assert resolved.promptTemplates is not resources["promptTemplates"]
        await harness.prompt("hello")
        roles = [
            getattr(entry["message"], "role", None)
            for entry in await session.getEntries()
            if entry["type"] == "message"
        ]
        assert roles == ["user", "assistant"]

    await run_suite(Session(InMemorySessionStorage()), create_env(tmp_path))
    jsonl_path = tmp_path / "session.jsonl"
    jsonl_env = create_env(tmp_path)
    jsonl_storage = await JsonlSessionStorage.create(
        jsonl_env,
        str(jsonl_path),
        {"cwd": str(tmp_path), "sessionId": "session-jsonl"},
    )
    await run_suite(Session(jsonl_storage), jsonl_env)
    assert updates[0] == (1, 0)


@pytest.mark.asyncio
async def test_compact_and_navigate_tree_entry_points(tmp_path: Path) -> None:
    registration = register_faux_provider()
    registrations.append(registration)
    registration.set_responses(
        [
            faux_assistant_message("## Goal\nCompacted"),
            faux_assistant_message("## Goal\nBranch summary"),
        ]
    )
    session = Session(InMemorySessionStorage())
    root = await session.appendMessage(
        {"role": "user", "content": [TextContent(text="root")], "timestamp": int(time.time() * 1000)}
    )
    await session.appendMessage(faux_assistant_message("root assistant"))
    await session.appendMessage(
        {"role": "user", "content": [TextContent(text="branch")], "timestamp": int(time.time() * 1000)}
    )
    await session.appendMessage(faux_assistant_message("branch assistant"))
    await session.moveTo(root)
    target_user = await session.appendMessage(
        {
            "role": "user",
            "content": [TextContent(text="target editor")],
            "timestamp": int(time.time() * 1000),
        }
    )

    harness = create_harness(
        env=create_env(tmp_path),
        session=session,
        model=registration.get_model(),
        getApiKeyAndHeaders=lambda _model: {"apiKey": "secret"},
    )
    compact_events: list[tuple[str, bool]] = []
    tree_events: list[tuple[str | None, str | None]] = []

    harness.subscribe(
        lambda event, _signal: compact_events.append((event.compactionEntry["type"], event.fromHook))
        if event.type == "session_compact"
        else tree_events.append((event.newLeafId, event.oldLeafId))
        if event.type == "session_tree"
        else None
    )

    compact_result = await harness.compact()
    leaf_after_compact = await session.getLeafId()
    navigate_result = await harness.navigateTree(target_user, {"summarize": True})

    assert compact_result.summary
    assert compact_events == [("compaction", False)]
    assert navigate_result.cancelled is False
    assert navigate_result.editorText == "target editor"
    assert navigate_result.summaryEntry is not None
    assert navigate_result.summaryEntry["type"] == "branch_summary"
    assert tree_events[-1][1] == leaf_after_compact
