from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

import harnify_ai.providers.google_vertex as google_vertex
from harnify_ai.types import Context, Model, ModelCost, SimpleStreamOptions


@dataclass(slots=True)
class _FakeFinishReason:
    name: str


@dataclass(slots=True)
class _FakeCandidate:
    finish_reason: _FakeFinishReason | None = None
    content: Any = None


@dataclass(slots=True)
class _FakeChunk:
    candidates: list[_FakeCandidate]
    response_id: str | None = None
    usage_metadata: Any = None


class _FakeModels:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any):
        self.calls.append(kwargs)

        async def _iterate():
            for chunk in self._chunks:
                yield chunk

        return _iterate()


class _BlockingStream:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False
        self.cancelled = False

    def __aiter__(self) -> _BlockingStream:
        return self

    async def __anext__(self):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _FakeAsyncClient:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.models = _FakeModels(chunks)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.aio = _FakeAsyncClient(chunks)


class _ConstructorCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return kwargs


def _make_model(base_url: str = "https://{location}-aiplatform.googleapis.com") -> Model:
    return Model(
        id="gemini-3-flash-preview",
        name="gemini-3-flash-preview",
        api="google-vertex",
        provider="google-vertex",
        baseUrl=base_url,
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=1_000_000,
        maxTokens=65_536,
    )


def _make_context() -> Context:
    return Context(messages=[{"role": "user", "content": "hello", "timestamp": 1}])


def test_create_client_uses_adc_and_keeps_placeholder_base_url_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _ConstructorCapture()
    monkeypatch.setattr(google_vertex, "GoogleGenAI", capture)

    google_vertex.create_client(_make_model(), "test-project", "us-central1", {"x-test": "1"})

    assert capture.calls == [
        {
            "vertexai": True,
            "project": "test-project",
            "location": "us-central1",
            "http_options": {"apiVersion": "v1", "headers": {"x-test": "1"}},
        }
    ]


def test_create_client_with_api_key_forwards_custom_base_url_and_disables_extra_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _ConstructorCapture()
    monkeypatch.setattr(google_vertex, "GoogleGenAI", capture)

    google_vertex.create_client_with_api_key(
        _make_model("https://proxy.example.com/v1/projects/test-project/locations/global"),
        "AIzaSyExampleRealisticLookingApiKey123456",
    )

    assert capture.calls == [
        {
            "vertexai": True,
            "api_key": "AIzaSyExampleRealisticLookingApiKey123456",
            "http_options": {
                "baseUrl": "https://proxy.example.com/v1/projects/test-project/locations/global",
                "baseUrlResourceScope": "COLLECTION",
                "apiVersion": "",
            },
        }
    ]


@pytest.mark.parametrize(
    ("api_key", "env_api_key", "expected"),
    [
        ("<authenticated>", None, None),
        ("gcp-vertex-credentials", None, None),
        (None, "<authenticated>", None),
        (None, "AIzaSyExampleRealisticLookingApiKey123456", "AIzaSyExampleRealisticLookingApiKey123456"),
        ("AIzaSyInlineKey123", None, "AIzaSyInlineKey123"),
    ],
)
def test_resolve_api_key_filters_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    env_api_key: str | None,
    expected: str | None,
) -> None:
    if env_api_key is None:
        monkeypatch.delenv("GOOGLE_CLOUD_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", env_api_key)

    options = {} if api_key is None else {"apiKey": api_key}
    assert google_vertex.resolve_api_key(options) == expected


def test_build_params_preserves_abort_signal_for_callback_surface() -> None:
    signal = object()

    params = google_vertex.build_params(
        _make_model(),
        _make_context(),
        {"signal": signal},
    )

    assert params["config"]["abortSignal"] is signal


def test_build_params_uses_request_aborted_message_for_preaborted_signal() -> None:
    class _Signal:
        aborted = True

    with pytest.raises(RuntimeError, match="^Request aborted$"):
        google_vertex.build_params(
            _make_model(),
            _make_context(),
            {"signal": _Signal()},
        )


@pytest.mark.asyncio
async def test_stream_google_vertex_uses_adc_branch_for_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient([_FakeChunk(candidates=[_FakeCandidate(finish_reason=_FakeFinishReason("STOP"))])])
    calls = {"adc": 0, "api_key": 0}

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        calls["adc"] += 1
        return fake_client

    def fake_create_client_with_api_key(*_args: Any, **_kwargs: Any) -> _FakeClient:
        calls["api_key"] += 1
        return fake_client

    monkeypatch.setattr(google_vertex, "create_client", fake_create_client)
    monkeypatch.setattr(google_vertex, "create_client_with_api_key", fake_create_client_with_api_key)

    result = await google_vertex.stream_google_vertex(
        _make_model(),
        _make_context(),
        {
            "apiKey": "<authenticated>",
            "project": "test-project",
            "location": "us-central1",
        },
    ).result()

    assert result.api == "google-vertex"
    assert result.stopReason == "stop"
    assert calls == {"adc": 1, "api_key": 0}
    assert fake_client.aio.closed is True


@pytest.mark.asyncio
async def test_stream_google_vertex_strips_abort_signal_before_sdk_call() -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient([_FakeChunk(candidates=[_FakeCandidate(finish_reason=_FakeFinishReason("STOP"))])])
    signal = object()

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    result = await google_vertex.stream_google_vertex(
        _make_model(),
        _make_context(),
        {
            "client": fake_client,
            "signal": signal,
            "onPayload": on_payload,
        },
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["config"]["abortSignal"] is signal
    assert captured_payload["config"]["toolConfig"] is None
    assert "abortSignal" not in fake_client.aio.models.calls[0]["config"]
    assert "toolConfig" not in fake_client.aio.models.calls[0]["config"]


@pytest.mark.asyncio
async def test_stream_google_vertex_aborts_while_waiting_for_stream_chunk() -> None:
    blocking_stream = _BlockingStream()

    class _BlockingModels:
        def generate_content_stream(self, **kwargs: Any):
            return blocking_stream

    class _BlockingAsyncClient:
        def __init__(self) -> None:
            self.models = _BlockingModels()
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class _BlockingClient:
        def __init__(self) -> None:
            self.aio = _BlockingAsyncClient()

    signal = asyncio.Event()
    stream = google_vertex.stream_google_vertex(
        _make_model(),
        _make_context(),
        {"client": _BlockingClient(), "signal": signal},
    )

    await asyncio.wait_for(blocking_stream.entered.wait(), timeout=1)
    signal.set()
    result = await asyncio.wait_for(stream.result(), timeout=1)

    assert result.stopReason == "aborted"
    assert result.errorMessage == "Request was aborted"
    assert blocking_stream.cancelled is True
    assert blocking_stream.closed is True


@pytest.mark.asyncio
async def test_stream_simple_google_vertex_uses_gemini3_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, Any] | None = None
    fake_client = _FakeClient(
        [
            _FakeChunk(
                candidates=[_FakeCandidate(finish_reason=_FakeFinishReason("STOP"))],
                response_id="resp_vertex_simple",
            )
        ]
    )

    def fake_create_client(*_args: Any, **_kwargs: Any) -> _FakeClient:
        return fake_client

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured_payload
        captured_payload = payload

    monkeypatch.setattr(google_vertex, "create_client", fake_create_client)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    result = await google_vertex.stream_simple_google_vertex(
        _make_model(),
        _make_context(),
        SimpleStreamOptions(reasoning="low", onPayload=on_payload),
    ).result()

    assert result.stopReason == "stop"
    assert captured_payload is not None
    assert captured_payload["config"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "LOW",
    }


def test_google_vertex_module_exports_expected_names() -> None:
    assert google_vertex.__all__ == [
        "GoogleVertexOptions",
        "streamGoogleVertex",
        "streamSimpleGoogleVertex",
    ]
