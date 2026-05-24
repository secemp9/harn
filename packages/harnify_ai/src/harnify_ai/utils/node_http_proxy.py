"""Environment-driven HTTP proxy resolution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse

DEFAULT_PROXY_PORTS: dict[str, int] = {
    "ftp": 21,
    "gopher": 70,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}

UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; "
    "use an HTTP or HTTPS proxy URL."
)


@dataclass(frozen=True, slots=True)
class NodeHttpProxyAgents:
    httpAgent: str
    httpsAgent: str


def _get_proxy_env(key: str) -> str:
    return os.environ.get(key.lower(), "") or os.environ.get(key.upper(), "")


def _parse_proxy_target_url(target_url: str | ParseResult) -> ParseResult | None:
    if isinstance(target_url, ParseResult):
        return target_url
    parsed = urlparse(target_url)
    return parsed if parsed.scheme and parsed.netloc else None


def _should_proxy_hostname(hostname: str, port: int) -> bool:
    no_proxy = _get_proxy_env("no_proxy").lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False

    for proxy in [item for item in no_proxy.replace(" ", ",").split(",") if item]:
        proxy_hostname = proxy
        proxy_port = 0
        if ":" in proxy and proxy.count(":") == 1:
            host_part, port_part = proxy.rsplit(":", 1)
            if port_part.isdigit():
                proxy_hostname = host_part
                proxy_port = int(port_part)
        if proxy_port and proxy_port != port:
            continue

        if not proxy_hostname.startswith(("*", ".")):
            if hostname == proxy_hostname:
                return False
            continue

        normalized = proxy_hostname[1:] if proxy_hostname.startswith("*") else proxy_hostname
        if hostname.endswith(normalized):
            return False

    return True


def _get_proxy_for_url(target_url: str | ParseResult) -> str:
    parsed_url = _parse_proxy_target_url(target_url)
    if parsed_url is None or not parsed_url.scheme or not parsed_url.netloc:
        return ""

    protocol = parsed_url.scheme
    hostname = parsed_url.hostname or ""
    port = parsed_url.port or DEFAULT_PROXY_PORTS.get(protocol, 0)
    if not _should_proxy_hostname(hostname, port):
        return ""

    proxy = _get_proxy_env(f"{protocol}_proxy") or _get_proxy_env("all_proxy")
    if proxy and "://" not in proxy:
        proxy = f"{protocol}://{proxy}"
    return proxy


def resolve_http_proxy_url_for_target(target_url: str | ParseResult) -> ParseResult | None:
    proxy = _get_proxy_for_url(target_url)
    if not proxy:
        return None

    proxy_url = urlparse(proxy)
    if not proxy_url.scheme or not proxy_url.netloc:
        raise ValueError(f"Invalid proxy URL {proxy!r}")
    if proxy_url.scheme not in {"http", "https"}:
        raise ValueError(f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {proxy_url.scheme}:")
    return proxy_url


def create_http_proxy_agents_for_target(target_url: str | ParseResult) -> NodeHttpProxyAgents | None:
    proxy_url = resolve_http_proxy_url_for_target(target_url)
    if proxy_url is None:
        return None
    proxy = proxy_url.geturl()
    return NodeHttpProxyAgents(httpAgent=proxy, httpsAgent=proxy)


resolveHttpProxyUrlForTarget = resolve_http_proxy_url_for_target
createHttpProxyAgentsForTarget = create_http_proxy_agents_for_target
