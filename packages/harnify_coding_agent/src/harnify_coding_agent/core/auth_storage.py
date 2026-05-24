"""Credential persistence and resolution for coding-agent providers."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from harnify_ai.env_api_keys import find_env_keys, get_env_api_key
from harnify_ai.utils.oauth import (
    OAuthCredentials,
    getOAuthApiKey,
    getOAuthProvider,
    getOAuthProviders,
)

from harnify_coding_agent.config import get_auth_path
from harnify_coding_agent.core.resolve_config_value import resolveConfigValue
from harnify_coding_agent.utils.paths import normalize_path

type ApiKeyCredential = dict[str, str]
type OAuthCredential = dict[str, Any]
type AuthCredential = ApiKeyCredential | OAuthCredential
type AuthStorageData = dict[str, AuthCredential]
type AuthStatusSource = Literal[
    "stored",
    "runtime",
    "environment",
    "fallback",
    "models_json_key",
    "models_json_command",
]


@dataclass(slots=True)
class AuthStatus:
    configured: bool
    source: AuthStatusSource | None = None
    label: str | None = None


@dataclass(slots=True)
class LockResult:
    result: Any
    next: str | None = None


class AuthStorageBackend:
    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:  # pragma: no cover - protocol-like
        raise NotImplementedError

    async def withLockAsync(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:  # pragma: no cover
        raise NotImplementedError


class FileAuthStorageBackend(AuthStorageBackend):
    def __init__(self, authPath: str | None = None):
        self.authPath = normalize_path(authPath or get_auth_path())

    def _lock_path(self) -> str:
        return f"{self.authPath}.lock"

    def ensureParentDir(self) -> None:
        Path(self.authPath).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(Path(self.authPath).parent, 0o700)
        except OSError:
            pass

    def ensureFileExists(self) -> None:
        if os.path.exists(self.authPath):
            return
        with open(self.authPath, "w", encoding="utf-8") as handle:
            handle.write("{}")
        try:
            os.chmod(self.authPath, 0o600)
        except OSError:
            pass

    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:
        self.ensureParentDir()
        self.ensureFileExists()
        with FileLock(self._lock_path(), timeout=10):
            current = None
            if os.path.exists(self.authPath):
                with open(self.authPath, encoding="utf-8") as handle:
                    current = handle.read()
            outcome = fn(current)
            if outcome.next is not None:
                with open(self.authPath, "w", encoding="utf-8") as handle:
                    handle.write(outcome.next)
                try:
                    os.chmod(self.authPath, 0o600)
                except OSError:
                    pass
            return outcome.result

    async def withLockAsync(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:
        self.ensureParentDir()
        self.ensureFileExists()
        with FileLock(self._lock_path(), timeout=10):
            current = None
            if os.path.exists(self.authPath):
                with open(self.authPath, encoding="utf-8") as handle:
                    current = handle.read()
            outcome = await fn(current)
            if outcome.next is not None:
                with open(self.authPath, "w", encoding="utf-8") as handle:
                    handle.write(outcome.next)
                try:
                    os.chmod(self.authPath, 0o600)
                except OSError:
                    pass
            return outcome.result


class InMemoryAuthStorageBackend(AuthStorageBackend):
    def __init__(self) -> None:
        self.value: str | None = None

    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:
        outcome = fn(self.value)
        if outcome.next is not None:
            self.value = outcome.next
        return outcome.result

    async def withLockAsync(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:
        outcome = await fn(self.value)
        if outcome.next is not None:
            self.value = outcome.next
        return outcome.result


def _coerce_oauth_credentials(value: dict[str, Any]) -> OAuthCredentials:
    return OAuthCredentials.model_validate({key: item for key, item in value.items() if key != "type"})


class AuthStorage:
    def __init__(self, storage: AuthStorageBackend):
        self.data: AuthStorageData = {}
        self.runtimeOverrides: dict[str, str] = {}
        self.fallbackResolver: Callable[[str], str | None] | None = None
        self.loadError: Exception | None = None
        self.errors: list[Exception] = []
        self.storage = storage
        self.reload()

    @classmethod
    def create(cls, authPath: str | None = None) -> AuthStorage:
        return cls(FileAuthStorageBackend(authPath))

    @classmethod
    def fromStorage(cls, storage: AuthStorageBackend) -> AuthStorage:
        return cls(storage)

    @classmethod
    def inMemory(cls, data: AuthStorageData | None = None) -> AuthStorage:
        storage = InMemoryAuthStorageBackend()
        storage.withLock(lambda _current: LockResult(result=None, next=json.dumps(data or {}, indent=2)))
        return cls.fromStorage(storage)

    def setRuntimeApiKey(self, provider: str, apiKey: str) -> None:
        self.runtimeOverrides[provider] = apiKey

    def removeRuntimeApiKey(self, provider: str) -> None:
        self.runtimeOverrides.pop(provider, None)

    def setFallbackResolver(self, resolver: Callable[[str], str | None]) -> None:
        self.fallbackResolver = resolver

    def recordError(self, error: Any) -> None:
        self.errors.append(error if isinstance(error, Exception) else Exception(str(error)))

    def parseStorageData(self, content: str | None) -> AuthStorageData:
        if not content:
            return {}
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("auth.json must contain an object")
        return parsed

    def reload(self) -> None:
        content: str | None = None

        def capture(current: str | None) -> LockResult:
            nonlocal content
            content = current
            return LockResult(result=None)

        try:
            self.storage.withLock(capture)
            self.data = self.parseStorageData(content)
            self.loadError = None
        except Exception as error:  # noqa: BLE001
            self.loadError = error
            self.recordError(error)

    def persistProviderChange(self, provider: str, credential: AuthCredential | None) -> None:
        if self.loadError is not None:
            return

        def persist(current: str | None) -> LockResult:
            current_data = self.parseStorageData(current)
            merged = dict(current_data)
            if credential is None:
                merged.pop(provider, None)
            else:
                merged[provider] = copy.deepcopy(credential)
            return LockResult(result=None, next=json.dumps(merged, indent=2))

        try:
            self.storage.withLock(persist)
        except Exception as error:  # noqa: BLE001
            self.recordError(error)

    def get(self, provider: str) -> AuthCredential | None:
        value = self.data.get(provider)
        return copy.deepcopy(value) if value is not None else None

    def set(self, provider: str, credential: AuthCredential) -> None:
        self.data[provider] = copy.deepcopy(credential)
        self.persistProviderChange(provider, credential)

    def remove(self, provider: str) -> None:
        self.data.pop(provider, None)
        self.persistProviderChange(provider, None)

    def list(self) -> list[str]:
        return list(self.data.keys())

    def has(self, provider: str) -> bool:
        return provider in self.data

    def hasAuth(self, provider: str) -> bool:
        if provider in self.runtimeOverrides:
            return True
        if provider in self.data:
            return True
        if get_env_api_key(provider):
            return True
        if self.fallbackResolver and self.fallbackResolver(provider):
            return True
        return False

    def getAuthStatus(self, provider: str) -> AuthStatus:
        if provider in self.data:
            return AuthStatus(configured=True, source="stored")
        if provider in self.runtimeOverrides:
            return AuthStatus(configured=False, source="runtime", label="--api-key")
        env_keys = find_env_keys(provider)
        if env_keys and env_keys[0]:
            return AuthStatus(configured=False, source="environment", label=env_keys[0])
        if self.fallbackResolver and self.fallbackResolver(provider):
            return AuthStatus(configured=False, source="fallback", label="custom provider config")
        return AuthStatus(configured=False)

    def getAll(self) -> AuthStorageData:
        return copy.deepcopy(self.data)

    def drainErrors(self) -> list[Exception]:
        drained = list(self.errors)
        self.errors = []
        return drained

    async def login(self, providerId: str, callbacks: Any) -> None:
        provider = getOAuthProvider(providerId)
        if provider is None:
            raise ValueError(f"Unknown OAuth provider: {providerId}")
        credentials = await provider.login(callbacks)
        self.set(providerId, {"type": "oauth", **credentials.model_dump(exclude_none=False)})

    def logout(self, provider: str) -> None:
        self.remove(provider)

    async def refreshOAuthTokenWithLock(self, providerId: str) -> dict[str, Any] | None:
        provider = getOAuthProvider(providerId)
        if provider is None:
            return None

        async def refresh(current: str | None) -> LockResult:
            current_data = self.parseStorageData(current)
            self.data = current_data
            self.loadError = None
            credential = current_data.get(providerId)
            if not isinstance(credential, dict) or credential.get("type") != "oauth":
                return LockResult(result=None)

            oauth_credential = _coerce_oauth_credentials(credential)
            if int(time.time() * 1000) < oauth_credential.expires:
                return LockResult(
                    result={
                        "apiKey": provider.getApiKey(oauth_credential),
                        "newCredentials": oauth_credential,
                    }
                )

            oauth_credentials: dict[str, OAuthCredentials] = {}
            for key, value in current_data.items():
                if isinstance(value, dict) and value.get("type") == "oauth":
                    oauth_credentials[key] = _coerce_oauth_credentials(value)

            refreshed = await getOAuthApiKey(providerId, oauth_credentials)
            if refreshed is None:
                return LockResult(result=None)

            merged = dict(current_data)
            merged[providerId] = {
                "type": "oauth",
                **refreshed["newCredentials"].model_dump(exclude_none=False),
            }
            self.data = merged
            self.loadError = None
            return LockResult(result=refreshed, next=json.dumps(merged, indent=2))

        return await self.storage.withLockAsync(refresh)

    async def getApiKey(self, providerId: str, options: dict[str, Any] | None = None) -> str | None:
        runtime_key = self.runtimeOverrides.get(providerId)
        if runtime_key:
            return runtime_key

        credential = self.data.get(providerId)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            return resolveConfigValue(str(credential.get("key", "")))

        if isinstance(credential, dict) and credential.get("type") == "oauth":
            provider = getOAuthProvider(providerId)
            if provider is None:
                return None

            oauth_credential = _coerce_oauth_credentials(credential)
            needs_refresh = int(time.time() * 1000) >= oauth_credential.expires
            if needs_refresh:
                try:
                    refreshed = await self.refreshOAuthTokenWithLock(providerId)
                    if refreshed is not None:
                        return refreshed["apiKey"]
                except Exception as error:  # noqa: BLE001
                    self.recordError(error)
                    self.reload()
                    updated = self.data.get(providerId)
                    if isinstance(updated, dict) and updated.get("type") == "oauth":
                        updated_credentials = _coerce_oauth_credentials(updated)
                        if int(time.time() * 1000) < updated_credentials.expires:
                            return provider.getApiKey(updated_credentials)
                    return None
            return provider.getApiKey(oauth_credential)

        env_key = get_env_api_key(providerId)
        if env_key:
            return env_key

        include_fallback = True
        if options is not None and options.get("includeFallback") is False:
            include_fallback = False
        if include_fallback and self.fallbackResolver is not None:
            return self.fallbackResolver(providerId) or None
        return None

    def getOAuthProviders(self) -> list[Any]:
        return getOAuthProviders()


FileAuthStorage = FileAuthStorageBackend
InMemoryAuthStorage = InMemoryAuthStorageBackend

__all__ = [
    "ApiKeyCredential",
    "AuthCredential",
    "AuthStatus",
    "AuthStorage",
    "AuthStorageBackend",
    "AuthStorageData",
    "FileAuthStorage",
    "FileAuthStorageBackend",
    "InMemoryAuthStorage",
    "InMemoryAuthStorageBackend",
    "OAuthCredential",
]
