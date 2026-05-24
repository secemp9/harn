"""Model registry and custom-model loading for coding-agent providers."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from harnify_ai.api_registry import ApiProvider, register_api_provider
from harnify_ai.models import get_models, get_providers
from harnify_ai.providers.register_builtins import resetApiProviders
from harnify_ai.types import (
    AnthropicMessagesCompat,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
)
from harnify_ai.utils.event_stream import AssistantMessageEventStream
from harnify_ai.utils.oauth import OAuthCredentials, registerOAuthProvider, resetOAuthProviders

from harnify_coding_agent.config import get_models_path
from harnify_coding_agent.core.provider_display_names import BUILT_IN_PROVIDER_DISPLAY_NAMES
from harnify_coding_agent.core.resolve_config_value import (
    clearConfigValueCache,
    resolveConfigValueOrThrow,
    resolveConfigValueUncached,
    resolveHeadersOrThrow,
)
from harnify_coding_agent.utils.paths import normalize_path

from .auth_storage import AuthStatus, AuthStorage

type ProviderCompat = dict[str, Any] | OpenAICompletionsCompat | OpenAIResponsesCompat | AnthropicMessagesCompat


@dataclass(slots=True)
class ProviderOverride:
    baseUrl: str | None = None
    compat: ProviderCompat | None = None


@dataclass(slots=True)
class ProviderRequestConfig:
    apiKey: str | None = None
    headers: dict[str, str] | None = None
    authHeader: bool | None = None


@dataclass(slots=True)
class CustomModelsResult:
    models: list[Model]
    overrides: dict[str, ProviderOverride]
    modelOverrides: dict[str, dict[str, dict[str, Any]]]
    error: str | None = None


def _dump_compat(compat: ProviderCompat | None) -> dict[str, Any]:
    if compat is None:
        return {}
    if hasattr(compat, "model_dump"):
        return compat.model_dump(exclude_none=True)
    return dict(compat)


def _coerce_compat(compat: ProviderCompat | None) -> ProviderCompat | None:
    if compat is None:
        return None
    if hasattr(compat, "model_dump"):
        return compat
    if not isinstance(compat, dict):
        return compat
    compat_dict = dict(compat)
    if any(key in compat_dict for key in ("sendSessionIdHeader",)):
        return OpenAIResponsesCompat.model_validate(compat_dict)
    if any(
        key in compat_dict
        for key in (
            "supportsEagerToolInputStreaming",
            "supportsCacheControlOnTools",
            "forceAdaptiveThinking",
        )
    ):
        return AnthropicMessagesCompat.model_validate(compat_dict)
    return OpenAICompletionsCompat.model_validate(compat_dict)


def _merge_compat(baseCompat: ProviderCompat | None, overrideCompat: ProviderCompat | None) -> ProviderCompat | None:
    if overrideCompat is None:
        return baseCompat
    base = _dump_compat(baseCompat)
    override = _dump_compat(overrideCompat)
    merged = {**base, **override}
    if base.get("openRouterRouting") or override.get("openRouterRouting"):
        merged["openRouterRouting"] = {
            **(base.get("openRouterRouting") or {}),
            **(override.get("openRouterRouting") or {}),
        }
    if base.get("vercelGatewayRouting") or override.get("vercelGatewayRouting"):
        merged["vercelGatewayRouting"] = {
            **(base.get("vercelGatewayRouting") or {}),
            **(override.get("vercelGatewayRouting") or {}),
        }
    return _coerce_compat(merged)


def _apply_model_override(model: Model, override: dict[str, Any]) -> Model:
    update: dict[str, Any] = {}
    for key in ("name", "reasoning", "input", "contextWindow", "maxTokens"):
        if key in override:
            update[key] = override[key]
    if "thinkingLevelMap" in override:
        update["thinkingLevelMap"] = {**(model.thinkingLevelMap or {}), **(override["thinkingLevelMap"] or {})}
    if "cost" in override and isinstance(override["cost"], dict):
        update["cost"] = ModelCost.model_validate(
            {
                "input": override["cost"].get("input", model.cost.input),
                "output": override["cost"].get("output", model.cost.output),
                "cacheRead": override["cost"].get("cacheRead", model.cost.cacheRead),
                "cacheWrite": override["cost"].get("cacheWrite", model.cost.cacheWrite),
            }
        )
    update["compat"] = _merge_compat(model.compat, override.get("compat"))
    return model.model_copy(update=update)


def stripJsonComments(value: str) -> str:
    without_comments = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*',
        lambda match: match.group(0) if match.group(0).startswith('"') else "",
        value,
    )
    return re.sub(
        r'"(?:\\.|[^"\\])*"|,(\s*[}\]])',
        lambda match: match.group(0) if match.group(0).startswith('"') else match.group(1),
        without_comments,
    )


def emptyCustomModelsResult(error: str | None = None) -> CustomModelsResult:
    return CustomModelsResult(models=[], overrides={}, modelOverrides={}, error=error)


clearApiKeyCache = clearConfigValueCache


class ModelRegistry:
    def __init__(self, authStorage: AuthStorage, modelsJsonPath: str | None):
        self.models: list[Model] = []
        self.providerRequestConfigs: dict[str, ProviderRequestConfig] = {}
        self.modelRequestHeaders: dict[str, dict[str, str]] = {}
        self.registeredProviders: dict[str, dict[str, Any]] = {}
        self.loadError: str | None = None
        self.authStorage = authStorage
        self.modelsJsonPath = normalize_path(modelsJsonPath) if modelsJsonPath else None
        self.loadModels()

    @classmethod
    def create(cls, authStorage: AuthStorage, modelsJsonPath: str | None = None) -> ModelRegistry:
        return cls(authStorage, modelsJsonPath or get_models_path())

    @classmethod
    def inMemory(cls, authStorage: AuthStorage) -> ModelRegistry:
        return cls(authStorage, None)

    def refresh(self) -> None:
        self.providerRequestConfigs.clear()
        self.modelRequestHeaders.clear()
        self.loadError = None
        resetApiProviders()
        resetOAuthProviders()
        self.loadModels()
        for provider_name, config in list(self.registeredProviders.items()):
            self.applyProviderConfig(provider_name, config)

    def getError(self) -> str | None:
        return self.loadError

    def loadModels(self) -> None:
        if self.modelsJsonPath:
            custom = self.loadCustomModels(self.modelsJsonPath)
        else:
            custom = emptyCustomModelsResult()

        if custom.error:
            self.loadError = custom.error

        built_in_models = self.loadBuiltInModels(custom.overrides, custom.modelOverrides)
        combined = self.mergeCustomModels(built_in_models, custom.models)

        for oauth_provider in self.authStorage.getOAuthProviders():
            credential = self.authStorage.get(oauth_provider.id)
            if (
                isinstance(credential, dict)
                and credential.get("type") == "oauth"
                and getattr(oauth_provider, "modifyModels", None)
            ):
                combined = oauth_provider.modifyModels(
                    combined,
                    OAuthCredentials.model_validate({key: value for key, value in credential.items() if key != "type"}),
                )

        self.models = combined

    def loadBuiltInModels(
        self,
        overrides: dict[str, ProviderOverride],
        modelOverrides: dict[str, dict[str, dict[str, Any]]],
    ) -> list[Model]:
        result: list[Model] = []
        for provider in get_providers():
            models = list(get_models(provider))
            provider_override = overrides.get(provider)
            per_model_overrides = modelOverrides.get(provider, {})
            for model in models:
                updated = model
                if provider_override is not None:
                    updated = updated.model_copy(
                        update={
                            "baseUrl": provider_override.baseUrl or updated.baseUrl,
                            "compat": _merge_compat(updated.compat, provider_override.compat),
                        }
                    )
                model_override = per_model_overrides.get(model.id)
                if model_override is not None:
                    updated = _apply_model_override(updated, model_override)
                result.append(updated)
        return result

    @staticmethod
    def mergeCustomModels(builtInModels: list[Model], customModels: list[Model]) -> list[Model]:
        merged = list(builtInModels)
        for custom_model in customModels:
            existing_index = next(
                (
                    index
                    for index, model in enumerate(merged)
                    if model.provider == custom_model.provider and model.id == custom_model.id
                ),
                None,
            )
            if existing_index is None:
                merged.append(custom_model)
            else:
                merged[existing_index] = custom_model
        return merged

    def loadCustomModels(self, modelsJsonPath: str) -> CustomModelsResult:
        if not os.path.exists(modelsJsonPath):
            return emptyCustomModelsResult()

        try:
            with open(modelsJsonPath, encoding="utf-8") as handle:
                parsed = json.loads(stripJsonComments(handle.read()))

            providers = parsed.get("providers")
            if not isinstance(providers, dict):
                return emptyCustomModelsResult(
                    "Invalid models.json schema:\n"
                    "  - providers: must be an object\n\n"
                    f"File: {modelsJsonPath}"
                )

            self.validateConfig(providers)
            overrides: dict[str, ProviderOverride] = {}
            model_overrides: dict[str, dict[str, dict[str, Any]]] = {}

            for provider_name, provider_config in providers.items():
                if not isinstance(provider_config, dict):
                    raise ValueError(f"Provider {provider_name}: config must be an object.")

                if provider_config.get("baseUrl") or provider_config.get("compat"):
                    overrides[provider_name] = ProviderOverride(
                        baseUrl=provider_config.get("baseUrl"),
                        compat=_coerce_compat(provider_config.get("compat")),
                    )

                self.storeProviderRequestConfig(provider_name, provider_config)

                raw_model_overrides = provider_config.get("modelOverrides")
                if isinstance(raw_model_overrides, dict):
                    model_overrides[provider_name] = copy.deepcopy(raw_model_overrides)
                    for model_id, model_override in raw_model_overrides.items():
                        if isinstance(model_override, dict):
                            self.storeModelHeaders(provider_name, model_id, model_override.get("headers"))

            return CustomModelsResult(
                models=self.parseModels(providers),
                overrides=overrides,
                modelOverrides=model_overrides,
                error=None,
            )
        except json.JSONDecodeError as error:
            return emptyCustomModelsResult(f"Failed to parse models.json: {error}\n\nFile: {modelsJsonPath}")
        except Exception as error:  # noqa: BLE001
            return emptyCustomModelsResult(f"Failed to load models.json: {error}\n\nFile: {modelsJsonPath}")

    def validateConfig(self, providers: dict[str, Any]) -> None:
        built_in_providers = set(get_providers())
        for provider_name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                raise ValueError(f"Provider {provider_name}: config must be an object.")
            is_built_in = provider_name in built_in_providers
            models = provider_config.get("models") or []
            has_model_overrides = bool(provider_config.get("modelOverrides"))

            if not isinstance(models, list):
                raise ValueError(f"Provider {provider_name}: models must be an array.")

            if not models:
                has_override_fields = any(
                    provider_config.get(key) for key in ("baseUrl", "headers", "compat")
                )
                if not has_override_fields and not has_model_overrides:
                    raise ValueError(
                        f'Provider {provider_name}: must specify "baseUrl", "headers", "compat", '
                        '"modelOverrides", or "models".'
                    )
            elif not is_built_in:
                if not provider_config.get("baseUrl"):
                    raise ValueError(f'Provider {provider_name}: "baseUrl" is required when defining custom models.')
                if not provider_config.get("apiKey"):
                    raise ValueError(f'Provider {provider_name}: "apiKey" is required when defining custom models.')

            for model_def in models:
                if not isinstance(model_def, dict):
                    raise ValueError(f"Provider {provider_name}: model entries must be objects.")
                has_provider_api = bool(provider_config.get("api"))
                has_model_api = bool(model_def.get("api"))
                if not has_provider_api and not has_model_api and not is_built_in:
                    raise ValueError(
                        f'Provider {provider_name}, model {model_def.get("id")}: no "api" specified. '
                        "Set at provider or model level."
                    )
                if not model_def.get("id"):
                    raise ValueError(f'Provider {provider_name}: model missing "id"')
                if model_def.get("contextWindow") is not None and model_def["contextWindow"] <= 0:
                    raise ValueError(f'Provider {provider_name}, model {model_def["id"]}: invalid contextWindow')
                if model_def.get("maxTokens") is not None and model_def["maxTokens"] <= 0:
                    raise ValueError(f'Provider {provider_name}, model {model_def["id"]}: invalid maxTokens')

    def parseModels(self, providers: dict[str, Any]) -> list[Model]:
        models: list[Model] = []
        built_in_providers = set(get_providers())
        defaults_cache: dict[str, dict[str, str]] = {}

        def get_built_in_defaults(provider_name: str) -> dict[str, str] | None:
            if provider_name not in built_in_providers:
                return None
            if provider_name in defaults_cache:
                return defaults_cache[provider_name]
            built_in_models = list(get_models(provider_name))
            if not built_in_models:
                return None
            defaults = {"api": built_in_models[0].api, "baseUrl": built_in_models[0].baseUrl}
            defaults_cache[provider_name] = defaults
            return defaults

        for provider_name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                continue
            model_defs = provider_config.get("models") or []
            if not isinstance(model_defs, list) or not model_defs:
                continue

            built_in_defaults = get_built_in_defaults(provider_name)
            for model_def in model_defs:
                if not isinstance(model_def, dict):
                    continue
                api = model_def.get("api") or provider_config.get("api") or (built_in_defaults or {}).get("api")
                if not api:
                    continue
                base_url = (
                    model_def.get("baseUrl")
                    or provider_config.get("baseUrl")
                    or (built_in_defaults or {}).get("baseUrl")
                )
                if not base_url:
                    continue
                compat = _merge_compat(
                    _coerce_compat(provider_config.get("compat")),
                    _coerce_compat(model_def.get("compat")),
                )
                self.storeModelHeaders(provider_name, model_def["id"], model_def.get("headers"))
                cost = model_def.get("cost") or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                models.append(
                    Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,
                        provider=provider_name,
                        baseUrl=base_url,
                        reasoning=bool(model_def.get("reasoning", False)),
                        thinkingLevelMap=model_def.get("thinkingLevelMap"),
                        input=model_def.get("input") or ["text"],
                        cost=ModelCost.model_validate(cost),
                        contextWindow=model_def.get("contextWindow", 128000),
                        maxTokens=model_def.get("maxTokens", 16384),
                        headers=None,
                        compat=compat,
                    )
                )
        return models

    def getAll(self) -> list[Model]:
        return list(self.models)

    def getAvailable(self) -> list[Model]:
        return [model for model in self.models if self.hasConfiguredAuth(model)]

    def find(self, provider: str, modelId: str) -> Model | None:
        return next((model for model in self.models if model.provider == provider and model.id == modelId), None)

    def hasConfiguredAuth(self, model: Model) -> bool:
        return self.authStorage.hasAuth(model.provider) or (
            self.providerRequestConfigs.get(model.provider, ProviderRequestConfig()).apiKey is not None
        )

    @staticmethod
    def getModelRequestKey(provider: str, modelId: str) -> str:
        return f"{provider}:{modelId}"

    def storeProviderRequestConfig(self, providerName: str, config: dict[str, Any]) -> None:
        if not any(config.get(key) is not None for key in ("apiKey", "headers", "authHeader")):
            return
        self.providerRequestConfigs[providerName] = ProviderRequestConfig(
            apiKey=config.get("apiKey"),
            headers=copy.deepcopy(config.get("headers")),
            authHeader=config.get("authHeader"),
        )

    def storeModelHeaders(self, providerName: str, modelId: str, headers: dict[str, str] | None) -> None:
        key = self.getModelRequestKey(providerName, modelId)
        if not headers:
            self.modelRequestHeaders.pop(key, None)
            return
        self.modelRequestHeaders[key] = copy.deepcopy(headers)

    async def getApiKeyAndHeaders(self, model: Model) -> dict[str, Any]:
        try:
            provider_config = self.providerRequestConfigs.get(model.provider)
            api_key_from_auth_storage = await self.authStorage.getApiKey(model.provider, {"includeFallback": False})
            api_key = api_key_from_auth_storage
            if api_key is None and provider_config and provider_config.apiKey:
                api_key = resolveConfigValueOrThrow(provider_config.apiKey, f'API key for provider "{model.provider}"')

            provider_headers = resolveHeadersOrThrow(
                provider_config.headers if provider_config else None,
                f'provider "{model.provider}"',
            )
            model_headers = resolveHeadersOrThrow(
                self.modelRequestHeaders.get(self.getModelRequestKey(model.provider, model.id)),
                f'model "{model.provider}/{model.id}"',
            )
            headers = None
            if model.headers or provider_headers or model_headers:
                headers = {**(model.headers or {}), **(provider_headers or {}), **(model_headers or {})}

            if provider_config and provider_config.authHeader:
                if not api_key:
                    return {"ok": False, "error": f'No API key found for "{model.provider}"'}
                headers = {**(headers or {}), "Authorization": f"Bearer {api_key}"}

            return {
                "ok": True,
                "apiKey": api_key,
                "headers": headers if headers else None,
            }
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": str(error)}

    def getProviderAuthStatus(self, provider: str) -> AuthStatus:
        auth_status = self.authStorage.getAuthStatus(provider)
        if auth_status.source is not None:
            return auth_status

        provider_api_key = self.providerRequestConfigs.get(provider, ProviderRequestConfig()).apiKey
        if not provider_api_key:
            return auth_status
        if provider_api_key.startswith("!"):
            return AuthStatus(configured=True, source="models_json_command")
        if os.environ.get(provider_api_key):
            return AuthStatus(configured=True, source="environment", label=provider_api_key)
        return AuthStatus(configured=True, source="models_json_key")

    def getProviderDisplayName(self, provider: str) -> str:
        registered_provider = self.registeredProviders.get(provider) or {}
        oauth_provider = next((item for item in self.authStorage.getOAuthProviders() if item.id == provider), None)
        return (
            registered_provider.get("name")
            or (
                (registered_provider.get("oauth") or {}).get("name")
                if isinstance(registered_provider.get("oauth"), dict)
                else None
            )
            or (oauth_provider.name if oauth_provider else None)
            or BUILT_IN_PROVIDER_DISPLAY_NAMES.get(provider)
            or provider
        )

    async def getApiKeyForProvider(self, provider: str) -> str | None:
        api_key = await self.authStorage.getApiKey(provider, {"includeFallback": False})
        if api_key is not None:
            return api_key
        provider_api_key = self.providerRequestConfigs.get(provider, ProviderRequestConfig()).apiKey
        return resolveConfigValueUncached(provider_api_key) if provider_api_key else None

    def isUsingOAuth(self, model: Model) -> bool:
        credential = self.authStorage.get(model.provider)
        return isinstance(credential, dict) and credential.get("type") == "oauth"

    def registerProvider(self, providerName: str, config: dict[str, Any]) -> None:
        self.validateProviderConfig(providerName, config)
        self.applyProviderConfig(providerName, config)
        self.upsertRegisteredProvider(providerName, config)

    def unregisterProvider(self, providerName: str) -> None:
        if providerName not in self.registeredProviders:
            return
        self.registeredProviders.pop(providerName, None)
        self.refresh()

    def upsertRegisteredProvider(self, providerName: str, config: dict[str, Any]) -> None:
        existing = self.registeredProviders.get(providerName)
        if existing is None:
            self.registeredProviders[providerName] = copy.deepcopy(config)
            return
        for key, value in config.items():
            if value is not None:
                existing[key] = copy.deepcopy(value)

    def validateProviderConfig(self, providerName: str, config: dict[str, Any]) -> None:
        if config.get("streamSimple") and not config.get("api"):
            raise ValueError(f'Provider {providerName}: "api" is required when registering streamSimple.')
        models = config.get("models") or []
        if not models:
            return
        if not config.get("baseUrl"):
            raise ValueError(f'Provider {providerName}: "baseUrl" is required when defining models.')
        if not config.get("apiKey") and not config.get("oauth"):
            raise ValueError(f'Provider {providerName}: "apiKey" or "oauth" is required when defining models.')
        for model_def in models:
            if not (model_def.get("api") or config.get("api")):
                raise ValueError(f'Provider {providerName}, model {model_def["id"]}: no "api" specified.')

    def applyProviderConfig(self, providerName: str, config: dict[str, Any]) -> None:
        oauth = config.get("oauth")
        if isinstance(oauth, dict):
            registerOAuthProvider(type("DynamicOAuthProvider", (), {"id": providerName, **oauth})())

        if config.get("streamSimple"):
            stream_simple = config["streamSimple"]

            def stream(model: Model, context: Any, options: Any = None) -> AssistantMessageEventStream:
                return stream_simple(model, context, options)

            register_api_provider(
                ApiProvider(api=config["api"], stream=stream, streamSimple=stream_simple),
                f"provider:{providerName}",
            )

        self.storeProviderRequestConfig(providerName, config)

        models = config.get("models") or []
        if models:
            self.models = [model for model in self.models if model.provider != providerName]
            for model_def in models:
                api = model_def.get("api") or config.get("api")
                self.storeModelHeaders(providerName, model_def["id"], model_def.get("headers"))
                self.models.append(
                    Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,
                        provider=providerName,
                        baseUrl=model_def.get("baseUrl") or config["baseUrl"],
                        reasoning=bool(model_def.get("reasoning")),
                        thinkingLevelMap=model_def.get("thinkingLevelMap"),
                        input=model_def.get("input") or ["text"],
                        cost=ModelCost.model_validate(
                            model_def.get("cost")
                            or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                        ),
                        contextWindow=model_def.get("contextWindow", 128000),
                        maxTokens=model_def.get("maxTokens", 16384),
                        headers=None,
                        compat=_coerce_compat(model_def.get("compat")),
                    )
                )
            if isinstance(oauth, dict) and oauth.get("modifyModels"):
                credential = self.authStorage.get(providerName)
                if isinstance(credential, dict) and credential.get("type") == "oauth":
                    from harnify_ai.utils.oauth.types import OAuthCredentials

                    self.models = oauth["modifyModels"](
                        self.models,
                        OAuthCredentials.model_validate(
                            {key: value for key, value in credential.items() if key != "type"}
                        ),
                    )
        elif config.get("baseUrl") or config.get("headers"):
            self.models = [
                model.model_copy(update={"baseUrl": config.get("baseUrl") or model.baseUrl})
                if model.provider == providerName
                else model
                for model in self.models
            ]


ProviderConfigInput = dict[str, Any]

__all__ = [
    "CustomModelsResult",
    "ModelRegistry",
    "ProviderConfigInput",
    "ProviderOverride",
    "ProviderRequestConfig",
    "clearApiKeyCache",
    "emptyCustomModelsResult",
    "stripJsonComments",
]
