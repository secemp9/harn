"""OpenAI Responses provider adapter."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterable
from typing import Any, Literal, TypedDict

from openai import AsyncOpenAI

from harnify_ai.env_api_keys import get_env_api_key
from harnify_ai.models import clamp_thinking_level
from harnify_ai.types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    Usage,
)
from harnify_ai.utils.event_stream import AssistantMessageEventStream

import harnify_ai.providers.cloudflare as _cloudflare
import harnify_ai.providers.github_copilot_headers as _copilot_headers
from harnify_ai.providers.openai_prompt_cache import clamp_openai_prompt_cache_key
from harnify_ai.providers.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from harnify_ai.providers.simple_options import build_base_options

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
is_cloudflare_provider = getattr(_cloudflare, "is_cloudflare_provider", lambda _provider: False)
resolve_cloudflare_base_url = getattr(_cloudflare, "resolve_cloudflare_base_url", lambda model: model.baseUrl)
build_copilot_dynamic_headers = getattr(_copilot_headers, "build_copilot_dynamic_headers", lambda **_: {})
has_copilot_vision_input = getattr(_copilot_headers, "has_copilot_vision_input", lambda _messages: False)


class OpenAIResponsesOptions(TypedDict, total=False):
    apiKey: str
    headers: dict[str, str]
    signal: Any
    sessionId: str
    cacheRetention: str
    onPayload: Any
    onResponse: Any
    timeoutMs: int
    maxRetries: int
    reasoningEffort: Literal["minimal", "low", "medium", "high", "xhigh"]
    reasoningSummary: Literal["auto", "detailed", "concise"] | None
    serviceTier: str


def _option(options: Any, name: str, default: Any = None) -> Any:
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def resolve_cache_retention(cache_retention: CacheRetention | None = None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    return "long" if os.environ.get("PI_CACHE_RETENTION") == "long" else "short"


def get_compat(model: Model) -> dict[str, bool]:
    compat = model.compat if getattr(model, "compat", None) is not None else None
    return {
        "sendSessionIdHeader": getattr(compat, "sendSessionIdHeader", None)
        if getattr(compat, "sendSessionIdHeader", None) is not None
        else True,
        "supportsLongCacheRetention": getattr(compat, "supportsLongCacheRetention", None)
        if getattr(compat, "supportsLongCacheRetention", None) is not None
        else True,
    }


def get_prompt_cache_retention(compat: dict[str, bool], cache_retention: CacheRetention) -> str | None:
    return "24h" if cache_retention == "long" and compat["supportsLongCacheRetention"] else None


def format_openai_responses_error(error: Any) -> str:
    if isinstance(error, Exception):
        status = getattr(error, "status", None)
        if isinstance(status, int):
            return f"OpenAI API error ({status}): {error}"
        return str(error)
    try:
        return json.dumps(error)
    except Exception:
        return str(error)


def _empty_usage() -> Usage:
    return AssistantMessage(
        content=[],
        api="openai-responses",
        provider="openai",
        model="",
        usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        stopReason="stop",
        timestamp=0,
    ).usage


def create_client(
    model: Model,
    context: Context,
    api_key: str | None = None,
    options_headers: dict[str, str] | None = None,
    session_id: str | None = None,
    *,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
) -> AsyncOpenAI:
    if not api_key:
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise RuntimeError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
            )
        api_key = env_key

    compat = get_compat(model)
    headers = dict(model.headers or {})
    if model.provider == "github-copilot":
        copilot_headers = build_copilot_dynamic_headers(
            messages=context.messages,
            hasImages=has_copilot_vision_input(context.messages),
        )
        headers.update(copilot_headers)

    if session_id:
        if compat["sendSessionIdHeader"]:
            headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id

    if options_headers:
        headers.update(options_headers)

    default_headers = (
        {
            **headers,
            "Authorization": headers.get("Authorization"),
            "cf-aig-authorization": f"Bearer {api_key}",
        }
        if model.provider == "cloudflare-ai-gateway"
        else headers
    )

    return AsyncOpenAI(
        api_key=api_key,
        base_url=resolve_cloudflare_base_url(model) if is_cloudflare_provider(model.provider) else model.baseUrl,
        default_headers=default_headers,
        timeout=(timeout_ms / 1000) if timeout_ms is not None else None,
        max_retries=max_retries if max_retries is not None else 2,
    )


def build_params(model: Model, context: Context, options: Any = None) -> dict[str, Any]:
    messages = convert_responses_messages(model, context, OPENAI_TOOL_CALL_PROVIDERS)

    cache_retention = resolve_cache_retention(_option(options, "cacheRetention"))
    compat = get_compat(model)
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "prompt_cache_key": (
            None if cache_retention == "none" else clamp_openai_prompt_cache_key(_option(options, "sessionId"))
        ),
        "prompt_cache_retention": get_prompt_cache_retention(compat, cache_retention),
        "store": False,
    }

    if _option(options, "maxTokens") is not None:
        params["max_output_tokens"] = _option(options, "maxTokens")
    if _option(options, "temperature") is not None:
        params["temperature"] = _option(options, "temperature")
    if _option(options, "serviceTier") is not None:
        params["service_tier"] = _option(options, "serviceTier")
    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    reasoning_effort = _option(options, "reasoningEffort")
    reasoning_summary = _option(options, "reasoningSummary")
    if model.reasoning:
        if reasoning_effort or reasoning_summary:
            effort = model.thinkingLevelMap.get(reasoning_effort, reasoning_effort) if reasoning_effort else "medium"
            params["reasoning"] = {"effort": effort, "summary": reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and (model.thinkingLevelMap or {}).get("off") is not None:
            params["reasoning"] = {"effort": (model.thinkingLevelMap or {}).get("off") or "none"}

    return params


def get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2.0
    return 1.0


def apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cacheRead *= multiplier
    usage.cost.cacheWrite *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite


async def _iterate_stream(stream_obj: Any) -> AsyncIterable[dict[str, Any]]:
    async for event in stream_obj:
        if hasattr(event, "model_dump"):
            yield event.model_dump()
        elif isinstance(event, dict):
            yield event
        else:
            yield json.loads(json.dumps(event, default=lambda value: value.__dict__))


def stream_openai_responses(
    model: Model,
    context: Context,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()

    async def run() -> None:
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stopReason="stop",
            timestamp=time.time_ns() // 1_000_000,
        )

        try:
            api_key = _option(options, "apiKey") or get_env_api_key(model.provider) or ""
            cache_retention = resolve_cache_retention(_option(options, "cacheRetention"))
            cache_session_id = None if cache_retention == "none" else _option(options, "sessionId")
            client = create_client(
                model,
                context,
                api_key,
                _option(options, "headers"),
                cache_session_id,
                timeout_ms=_option(options, "timeoutMs"),
                max_retries=_option(options, "maxRetries"),
            )
            params = build_params(model, context, options)
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_params = await _maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params
            openai_stream = await client.responses.create(**params)
            stream.push(StartEvent(partial=output))
            await process_responses_stream(
                _iterate_stream(openai_stream),
                output,
                stream,
                model,
                {
                    "serviceTier": _option(options, "serviceTier"),
                    "applyServiceTierPricing": lambda usage, tier: apply_service_tier_pricing(usage, tier, model),
                },
            )

            signal = _option(options, "signal")
            if signal is not None and getattr(signal, "is_set", lambda: False)():
                raise RuntimeError("Request was aborted")
            if getattr(signal, "aborted", False):
                raise RuntimeError("Request was aborted")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError("An unknown error occurred")

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            signal = _option(options, "signal")
            aborted = getattr(signal, "aborted", False) or (
                signal is not None and getattr(signal, "is_set", lambda: False)()
            )
            output.stopReason = "aborted" if aborted else "error"
            output.errorMessage = format_openai_responses_error(error)
            stream.push(ErrorEvent(reason=output.stopReason, error=output))
        finally:
            stream.end()

    asyncio.create_task(run())
    return stream


def stream_simple_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    api_key = _option(options, "apiKey") or get_env_api_key(model.provider)
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    return stream_openai_responses(
        model,
        context,
        {
            **base.model_dump(),
            "reasoningEffort": reasoning_effort,
        },
    )


streamOpenAIResponses = stream_openai_responses
streamSimpleOpenAIResponses = stream_simple_openai_responses
resolveCacheRetention = resolve_cache_retention
getCompat = get_compat
getPromptCacheRetention = get_prompt_cache_retention
formatOpenAIResponsesError = format_openai_responses_error
createClient = create_client
buildParams = build_params
getServiceTierCostMultiplier = get_service_tier_cost_multiplier
applyServiceTierPricing = apply_service_tier_pricing

__all__ = [
    "OpenAIResponsesOptions",
    "applyServiceTierPricing",
    "apply_service_tier_pricing",
    "buildParams",
    "build_params",
    "createClient",
    "create_client",
    "formatOpenAIResponsesError",
    "format_openai_responses_error",
    "getCompat",
    "getPromptCacheRetention",
    "getServiceTierCostMultiplier",
    "get_compat",
    "get_prompt_cache_retention",
    "get_service_tier_cost_multiplier",
    "resolveCacheRetention",
    "resolve_cache_retention",
    "streamOpenAIResponses",
    "streamSimpleOpenAIResponses",
    "stream_openai_responses",
    "stream_simple_openai_responses",
]
