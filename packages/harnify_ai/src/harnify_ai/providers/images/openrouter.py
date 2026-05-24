"""OpenRouter image-generation provider."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from harnify_ai.env_api_keys import get_env_api_key
from harnify_ai.types import AssistantImages, ImageContent, ImagesContext, ImagesModel, ImagesOptions, TextContent, Usage
from harnify_ai.utils.headers import headers_to_record
from harnify_ai.utils.sanitize_unicode import sanitize_surrogates


async def generate_images_openrouter(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions | None = None,
) -> AssistantImages:
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stopReason="stop",
        timestamp=0,
    )
    output.timestamp = __import__("time").time_ns() // 1_000_000

    try:
        api_key = options.apiKey if options and options.apiKey else get_env_api_key(model.provider)
        if not api_key:
            raise ValueError(f"No API key available for provider: {model.provider}")

        client = _create_client(model, api_key, options.headers if options else None)
        params = _build_params(model, context)
        if options and options.onPayload is not None:
            next_params = await options.onPayload(params, model)
            if next_params is not None:
                params = next_params

        raw_response = await client.chat.completions.with_raw_response.create(
            **params,
            timeout=options.timeoutMs if options and options.timeoutMs is not None else None,
        )
        response = raw_response.parse()
        if options and options.onResponse is not None:
            await options.onResponse(
                {"status": raw_response.http_response.status_code, "headers": headers_to_record(raw_response.http_response.headers)},
                model,
            )

        output.responseId = getattr(response, "id", None)
        if getattr(response, "usage", None) is not None:
            output.usage = _parse_usage(response.usage, model)

        choices = getattr(response, "choices", []) or []
        if choices:
            choice = choices[0]
            content = getattr(choice.message, "content", None)
            if isinstance(content, str) and content:
                output.output.append(TextContent(text=content))

            for image in getattr(choice.message, "images", []) or []:
                image_url = image.image_url if isinstance(getattr(image, "image_url", None), str) else getattr(image.image_url, "url", None)
                if not image_url or not image_url.startswith("data:"):
                    continue
                prefix, data = image_url.split(",", 1)
                mime_type = prefix.removeprefix("data:").split(";")[0]
                output.output.append(ImageContent(data=data, mimeType=mime_type))

        return output
    except BaseException as error:  # noqa: BLE001
        output.stopReason = "aborted" if _signal_aborted(options.signal if options else None) else "error"
        output.errorMessage = str(error)
        return output


def _create_client(model: ImagesModel, api_key: str, option_headers: dict[str, str] | None = None) -> AsyncOpenAI:
    headers = {}
    if model.headers:
        headers.update(model.headers)
    if option_headers:
        headers.update(option_headers)
    return AsyncOpenAI(api_key=api_key, base_url=model.baseUrl, default_headers=headers)


def _build_params(model: ImagesModel, context: ImagesContext) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in context.input:
        if item.type == "text":
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}})

    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if "text" in model.output else ["image"],
    }


def _parse_usage(raw_usage: Any, model: ImagesModel) -> Usage:
    prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
    details = getattr(raw_usage, "prompt_tokens_details", None)
    reported_cached_tokens = getattr(details, "cached_tokens", 0) if details is not None else 0
    cache_write_tokens = getattr(details, "cache_write_tokens", 0) if details is not None else 0
    cache_read_tokens = max(0, reported_cached_tokens - cache_write_tokens) if cache_write_tokens > 0 else reported_cached_tokens
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)

    usage = Usage(
        input=input_tokens,
        output=completion_tokens,
        cacheRead=cache_read_tokens,
        cacheWrite=cache_write_tokens,
        totalTokens=input_tokens + completion_tokens + cache_read_tokens + cache_write_tokens,
        cost={
            "input": (model.cost.input / 1_000_000) * input_tokens,
            "output": (model.cost.output / 1_000_000) * completion_tokens,
            "cacheRead": (model.cost.cacheRead / 1_000_000) * cache_read_tokens,
            "cacheWrite": (model.cost.cacheWrite / 1_000_000) * cache_write_tokens,
            "total": 0,
        },
    )
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite
    return usage


def _signal_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    if hasattr(signal, "aborted"):
        return bool(signal.aborted)
    if hasattr(signal, "is_set"):
        return bool(signal.is_set())
    return False


generateImagesOpenRouter = generate_images_openrouter
