from __future__ import annotations

from types import SimpleNamespace

import pytest

import harnify_ai.providers.faux as faux_provider
import harnify_ai.utils.oauth.pkce as pkce_module
from harnify_ai.image_models import get_image_model, get_image_providers
from harnify_ai.images import generate_images
from harnify_ai.providers.faux import faux_assistant_message, faux_text, faux_thinking, register_faux_provider
from harnify_ai.providers.images import register_builtins as image_register_builtins
from harnify_ai.stream import complete_simple
from harnify_ai.types import AssistantImages, Context, Tool
from harnify_ai.utils.oauth.pkce import generate_pkce


@pytest.mark.asyncio
async def test_faux_provider_streams_configured_response_end_to_end() -> None:
    registration = register_faux_provider({"api": "faux-test-suite"})
    registration.set_responses([faux_assistant_message([faux_thinking("plan"), faux_text("answer")])])

    model = registration.get_model()
    assert model is not None

    message = await complete_simple(model, {"messages": []})  # type: ignore[arg-type]

    assert message.stopReason == "stop"
    assert [block.type for block in message.content] == ["thinking", "text"]
    assert message.content[1].text == "answer"
    assert registration.state["callCount"] == 1
    assert registration.get_pending_response_count() == 0

    registration.unregister()


@pytest.mark.asyncio
async def test_faux_provider_returns_error_message_when_no_responses_are_queued() -> None:
    registration = register_faux_provider({"api": "faux-empty-suite"})

    model = registration.get_model()
    assert model is not None

    message = await complete_simple(model, {"messages": []})  # type: ignore[arg-type]

    assert message.stopReason == "error"
    assert message.errorMessage == "No more faux responses queued"

    registration.unregister()


def test_faux_helpers_match_ts_contract_for_timestamp_and_json_serialization() -> None:
    message = faux_assistant_message("hello", {"timestamp": 0, "stopReason": "stop"})
    tool_call = faux_provider.faux_tool_call("edit", {"path": "a.txt", "text": "hi"})
    context = Context(
        messages=[],
        tools=[
            Tool(
                name="edit",
                description="Edit a file.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
    )

    assert message.timestamp == 0
    assert faux_provider._assistant_content_to_text([tool_call]) == 'edit:{"path":"a.txt","text":"hi"}'
    assert faux_provider._serialize_context(context) == (
        'tools:[{"name":"edit","description":"Edit a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]'
    )


def test_faux_module_exports_and_registration_method_names_match_ts_surface() -> None:
    registration = register_faux_provider({"api": "faux-contract-suite"})
    try:
        assert registration.getModel() is not None
        registration.setResponses([])
        registration.appendResponses([])
        assert registration.getPendingResponseCount() == 0
        assert faux_provider.__all__ == [
            "FauxModelDefinition",
            "FauxContentBlock",
            "fauxText",
            "fauxThinking",
            "fauxToolCall",
            "fauxAssistantMessage",
            "FauxResponseFactory",
            "FauxResponseStep",
            "RegisterFauxProviderOptions",
            "FauxProviderRegistration",
            "registerFauxProvider",
        ]
    finally:
        registration.unregister()


@pytest.mark.asyncio
async def test_image_registry_and_generate_images_facade_use_lazy_provider_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
    assert model is not None
    assert get_image_providers() == ["openrouter"]

    async def fake_generate_images_openrouter(model, context, options=None):
        return AssistantImages(
            api=model.api,
            provider=model.provider,
            model=model.id,
            output=[{"type": "text", "text": "caption"}],
            stopReason="stop",
            timestamp=1_715_000_000_500,
        )

    image_register_builtins._openrouter_images_provider_module = None
    monkeypatch.setattr(
        image_register_builtins.importlib,
        "import_module",
        lambda name: SimpleNamespace(generateImagesOpenRouter=fake_generate_images_openrouter),
    )

    output = await generate_images(model, {"input": [{"type": "text", "text": "draw a cat"}]})  # type: ignore[arg-type]

    assert output.stopReason == "stop"
    assert output.output[0].text == "caption"


@pytest.mark.asyncio
async def test_image_register_builtins_caches_lazy_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
    assert model is not None

    calls = {"count": 0}

    def fail_import(name: str) -> None:
        calls["count"] += 1
        raise RuntimeError("boom")

    image_register_builtins._openrouter_images_provider_module = None
    monkeypatch.setattr(image_register_builtins.importlib, "import_module", fail_import)

    first = await image_register_builtins.generate_images_openrouter(
        model,
        {"input": [{"type": "text", "text": "draw a cat"}]},  # type: ignore[arg-type]
    )
    second = await image_register_builtins.generate_images_openrouter(
        model,
        {"input": [{"type": "text", "text": "draw a cat"}]},  # type: ignore[arg-type]
    )

    assert first.stopReason == "error"
    assert first.errorMessage == "boom"
    assert second.stopReason == "error"
    assert second.errorMessage == "boom"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_generate_pkce_returns_urlsafe_values() -> None:
    pkce = await generate_pkce()

    assert pkce.verifier
    assert pkce.challenge
    assert "=" not in pkce.verifier
    assert "=" not in pkce.challenge
    assert "+" not in pkce.verifier
    assert "/" not in pkce.verifier


def test_pkce_module_exports_expected_names() -> None:
    assert pkce_module.__all__ == ["generatePKCE", "generate_pkce"]
