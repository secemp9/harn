"""Shared facade for image generation providers."""

from __future__ import annotations

from harnify_ai.images_api_registry import get_images_api_provider
from harnify_ai.providers.images import register_builtins as _register_builtins  # noqa: F401
from harnify_ai.types import AssistantImages, ImagesContext, ImagesModel, ProviderImagesOptions


def _resolve_images_api_provider(api: str):
    provider = get_images_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: ProviderImagesOptions | None = None,
) -> AssistantImages:
    provider = _resolve_images_api_provider(model.api)
    resolved_context = context if isinstance(context, ImagesContext) else ImagesContext.model_validate(context)
    return await provider.generateImages(model, resolved_context, options)


generateImages = generate_images
