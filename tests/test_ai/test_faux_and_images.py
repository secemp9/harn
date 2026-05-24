from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnify_ai.image_models import get_image_model, get_image_providers
from harnify_ai.images import generate_images
from harnify_ai.providers.faux import faux_assistant_message, faux_text, faux_thinking, register_faux_provider
from harnify_ai.providers.images import register_builtins as image_register_builtins
from harnify_ai.stream import complete_simple
from harnify_ai.types import AssistantImages
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
        lambda name: SimpleNamespace(generate_images_openrouter=fake_generate_images_openrouter),
    )

    output = await generate_images(model, {"input": [{"type": "text", "text": "draw a cat"}]})  # type: ignore[arg-type]

    assert output.stopReason == "stop"
    assert output.output[0].text == "caption"


@pytest.mark.asyncio
async def test_generate_pkce_returns_urlsafe_values() -> None:
    pkce = await generate_pkce()

    assert pkce.verifier
    assert pkce.challenge
    assert "=" not in pkce.verifier
    assert "=" not in pkce.challenge
    assert "+" not in pkce.verifier
    assert "/" not in pkce.verifier
