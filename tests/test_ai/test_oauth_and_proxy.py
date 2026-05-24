from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

import harnify_ai
import harnify_ai.oauth as oauth_alias
import harnify_ai.utils.oauth as oauth_registry
from harnify_ai.utils import node_http_proxy
from harnify_ai.utils.oauth import anthropic, device_code, github_copilot, oauth_page, openai_codex
from harnify_ai.utils.oauth.types import OAuthCredentials


def _encode_jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode("ascii").rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{header}.{body}.sig"


def test_http_proxy_helpers_resolve_env_and_respect_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "proxy.local:8080")

    proxy_url = node_http_proxy.resolve_http_proxy_url_for_target("https://outside.example/path")
    assert proxy_url is not None
    assert proxy_url.geturl() == "https://proxy.local:8080"

    agents = node_http_proxy.create_http_proxy_agents_for_target("https://outside.example/path")
    assert agents is not None
    assert agents.httpAgent == "https://proxy.local:8080"
    assert agents.httpsAgent == "https://proxy.local:8080"

    monkeypatch.setenv("NO_PROXY", "outside.example,.internal.example,api.port.example:443")
    assert node_http_proxy.resolve_http_proxy_url_for_target("https://outside.example/path") is None
    assert node_http_proxy.resolve_http_proxy_url_for_target("https://service.internal.example/path") is None
    assert node_http_proxy.resolve_http_proxy_url_for_target("https://api.port.example/path") is None


def test_http_proxy_helpers_reject_unsupported_proxy_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "socks5://proxy.local:1080")

    with pytest.raises(RuntimeError, match="Unsupported proxy protocol"):
        node_http_proxy.resolve_http_proxy_url_for_target("https://outside.example/path")


def test_http_proxy_helpers_reject_invalid_proxy_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "https://")

    with pytest.raises(RuntimeError, match='Invalid proxy URL "https://": Invalid URL'):
        node_http_proxy.resolve_http_proxy_url_for_target("https://outside.example/path")


@pytest.mark.asyncio
async def test_device_code_flow_applies_slow_down_interval_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    sleeps: list[int] = []

    async def fake_sleep(ms: int, signal: object, cancel_message: str) -> None:
        sleeps.append(ms)
        clock["now"] += ms / 1000

    async def poll():
        return next(results)

    results = iter(
        [
            {"status": "pending"},
            {"status": "slow_down"},
            {"status": "complete", "accessToken": "token-123"},
        ]
    )

    monkeypatch.setattr(device_code.time, "time", lambda: clock["now"])
    monkeypatch.setattr(device_code, "_abortable_sleep", fake_sleep)

    token = await device_code.poll_oauth_device_code_flow(
        intervalSeconds=0.2,
        expiresInSeconds=20,
        poll=poll,
    )

    assert token == "token-123"
    assert sleeps == [1000, 1000, 6000]


@pytest.mark.asyncio
async def test_device_code_flow_uses_slow_down_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    sleeps: list[int] = []

    async def fake_sleep(ms: int, signal: object, cancel_message: str) -> None:
        sleeps.append(ms)
        clock["now"] += ms / 1000

    async def poll():
        return {"status": "slow_down"}

    monkeypatch.setattr(device_code.time, "time", lambda: clock["now"])
    monkeypatch.setattr(device_code, "_abortable_sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="slow_down responses"):
        await device_code.poll_oauth_device_code_flow(
            intervalSeconds=1,
            expiresInSeconds=1.5,
            poll=poll,
        )

    assert sleeps == [1000, 500]


def test_oauth_page_html_matches_escape_contract() -> None:
    rendered = oauth_page.oauth_error_html(
        'Bad <state> "quote"',
        "details & more 'text'",
    )

    assert "&lt;state&gt;" in rendered
    assert "&quot;quote&quot;" in rendered
    assert "details &amp; more &#39;text&#39;" in rendered
    assert "<div class=\"details\">details &amp; more &#39;text&#39;</div>" in rendered


def test_github_copilot_helpers_normalize_domains_and_base_urls() -> None:
    assert github_copilot.normalize_domain("company.ghe.com") == "company.ghe.com"
    assert github_copilot.normalize_domain("https://company.ghe.com/login") == "company.ghe.com"
    assert github_copilot.normalize_domain("   ") is None

    token = "tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2"
    assert github_copilot.get_github_copilot_base_url(token) == "https://api.individual.githubcopilot.com"
    assert github_copilot.get_github_copilot_base_url(None, "enterprise.example.com") == "https://copilot-api.enterprise.example.com"
    assert github_copilot.get_github_copilot_base_url() == "https://api.individual.githubcopilot.com"


def test_anthropic_format_error_details_prefers_runtime_metadata() -> None:
    class NamedError(Exception):
        name = "NamedError"
        code = "EFAIL"
        errno = 7
        cause = ValueError("inner")
        stack = "STACK"

    details = anthropic._format_error_details(NamedError("boom"))

    assert details == "NamedError: boom; code=EFAIL; errno=7; cause=ValueError: inner; stack=STACK"


@pytest.mark.asyncio
async def test_anthropic_callback_server_uses_ts_status_lines() -> None:
    server = await anthropic._start_callback_server("expected-state")
    try:
        reader, writer = await asyncio.open_connection(anthropic.CALLBACK_HOST, anthropic.CALLBACK_PORT)
        writer.write(b"GET /wrong HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()

        assert response.decode("utf-8", "ignore").splitlines()[0] == "HTTP/1.1 404 Not Found"
    finally:
        await server.close()


def test_anthropic_oauth_module_exports_expected_names() -> None:
    assert anthropic.__all__ == [
        "anthropicOAuthProvider",
        "anthropic_oauth_provider",
        "loginAnthropic",
        "login_anthropic",
        "refreshAnthropicToken",
        "refresh_anthropic_token",
    ]


@pytest.mark.asyncio
async def test_openai_codex_helpers_build_authorization_url_and_extract_account_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate_pkce() -> SimpleNamespace:
        return SimpleNamespace(verifier="verifier-123", challenge="challenge-456")

    monkeypatch.setattr(openai_codex, "generate_pkce", fake_generate_pkce)
    monkeypatch.setattr(openai_codex, "_create_state", lambda: "state-789")

    flow = await openai_codex.create_authorization_flow("cli-test")
    parsed = urlparse(flow["url"])
    params = parse_qs(parsed.query)

    assert flow["verifier"] == "verifier-123"
    assert flow["state"] == "state-789"
    assert params["code_challenge"] == ["challenge-456"]
    assert params["originator"] == ["cli-test"]
    assert params["state"] == ["state-789"]

    token = _encode_jwt({openai_codex.JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_123"}})
    assert openai_codex._get_account_id(token) == "acct_123"


def test_oauth_registry_restores_built_ins_after_unregister() -> None:
    oauth_registry.reset_oauth_providers()
    original = oauth_registry.get_oauth_provider("github-copilot")
    assert original is not None

    class ReplacementProvider:
        id = "github-copilot"
        name = "Replacement"
        usesCallbackServer = False

        async def login(self, callbacks):
            raise NotImplementedError

        async def refreshToken(self, credentials):
            return credentials

        def getApiKey(self, credentials):
            return credentials.access

        def modifyModels(self, models, credentials):
            return models

    replacement = ReplacementProvider()
    oauth_registry.register_oauth_provider(replacement)
    assert oauth_registry.get_oauth_provider("github-copilot") is replacement

    oauth_registry.unregister_oauth_provider("github-copilot")
    assert oauth_registry.get_oauth_provider("github-copilot") is original


@pytest.mark.asyncio
async def test_oauth_registry_refreshes_expired_custom_credentials() -> None:
    oauth_registry.reset_oauth_providers()

    class CustomProvider:
        id = "custom-oauth"
        name = "Custom OAuth"
        usesCallbackServer = False

        async def login(self, callbacks):
            raise NotImplementedError

        async def refreshToken(self, credentials: OAuthCredentials) -> OAuthCredentials:
            return credentials.model_copy(update={"access": "refreshed-key", "expires": 999_999_999_999})

        def getApiKey(self, credentials: OAuthCredentials) -> str:
            return credentials.access

        def modifyModels(self, models, credentials):
            return models

    provider = CustomProvider()
    oauth_registry.register_oauth_provider(provider)

    result = await oauth_registry.get_oauth_api_key(
        "custom-oauth",
        {
            "custom-oauth": OAuthCredentials(
                refresh="refresh-token",
                access="stale-key",
                expires=0,
            )
        },
    )

    assert result is not None
    assert result["apiKey"] == "refreshed-key"
    assert result["newCredentials"].access == "refreshed-key"

    oauth_registry.unregister_oauth_provider("custom-oauth")
    assert oauth_registry.get_oauth_provider("custom-oauth") is None


def test_package_exports_include_oauth_helpers_and_event_streams() -> None:
    assert hasattr(harnify_ai, "pollOAuthDeviceCodeFlow")
    assert hasattr(harnify_ai, "OAuthPrompt")
    assert hasattr(harnify_ai, "AssistantMessageEventStream")
    assert hasattr(oauth_alias, "getOAuthProvider")
    assert hasattr(oauth_alias, "OAuthDeviceCodePollResult")
