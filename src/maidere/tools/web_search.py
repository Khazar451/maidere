"""Async web search tool using SearXNG JSON API."""

from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import structlog

from maidere.core.config import settings
from maidere.tools.base import BaseTool

logger = structlog.get_logger()


class _DuckDuckGoParser(HTMLParser):
    """Extract result links and snippets from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            href = _unwrap_ddg_url(dict(attrs).get("href") or "")
            self.results.append({"title": "", "url": href, "content": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "content"

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "td", "div"):
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect (``//duckduckgo.com/l/?uddg=...``)."""
    if "uddg=" in href:
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        return unquote(target)
    return href


def _format_results(results: list[dict[str, Any]], limit: int) -> str:
    formatted = []
    for i, item in enumerate(results[:limit], 1):
        title = (item.get("title") or "No title").strip()
        url = item.get("url", "")
        content = (item.get("content") or "").strip()
        formatted.append(f"{i}. {title}\n   Markdown Link: [{title}]({url})\n   URL: {url}\n   Snippet: {content}")
    return "\n\n".join(formatted)


async def _query_searxng(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    endpoint = f"{settings.searxng_url.rstrip('/')}/search"
    await logger.ainfo("searxng_request", query=query, endpoint=endpoint)
    response = await client.get(endpoint, params={"q": query, "format": "json"})
    response.raise_for_status()
    return response.json().get("results", [])


async def _query_tavily(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    response = await client.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json={"query": query, "max_results": limit},
    )
    response.raise_for_status()
    return response.json().get("results", [])


async def _query_serper(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    response = await client.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": settings.serper_api_key},
        json={"q": query, "num": limit},
    )
    response.raise_for_status()
    return [
        {"title": r.get("title"), "url": r.get("link", ""), "content": r.get("snippet", "")}
        for r in response.json().get("organic", [])
    ]


async def _query_duckduckgo(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    response = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; Maidere)"},
    )
    response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    return [r for r in parser.results if r["url"].startswith("http")]


async def _search_fallback(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, Any]]:
    """Try each configured fallback provider in order; return the first non-empty result."""
    providers: list[tuple[str, Any]] = []
    if settings.tavily_api_key:
        providers.append(("tavily", lambda: _query_tavily(client, query, limit)))
    if settings.serper_api_key:
        providers.append(("serper", lambda: _query_serper(client, query, limit)))
    providers.append(("duckduckgo", lambda: _query_duckduckgo(client, query)))

    for name, run in providers:
        try:
            results = await run()
        except httpx.HTTPError as e:
            await logger.awarn("search_fallback_failed", provider=name, error=str(e))
            continue
        if results:
            await logger.ainfo("search_fallback_used", provider=name, query=query)
            return results
    return []


async def search_searxng(query: str, num_results: int = 5) -> str:
    """Search via SearXNG, falling back to other providers if it is unreachable."""
    if not query or not query.strip():
        return "Error: Empty search query provided."

    query = query.strip()
    limit = max(1, min(num_results, 10))

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            try:
                results = await _query_searxng(client, query)
            except (httpx.ConnectError, httpx.TimeoutException):
                await logger.awarn(
                    "searxng_offline", endpoint=settings.searxng_url, falling_back=settings.search_fallback_enabled
                )
                if not settings.search_fallback_enabled:
                    return "[Search Unavailable: SearXNG container is offline]"
                results = await _search_fallback(client, query, limit)
                if not results:
                    return "[Search Unavailable: SearXNG container is offline and fallback providers returned nothing]"
    except Exception as e:
        await logger.aerror("searxng_error", query=query, error=str(e))
        return f"[Search Error: {e!s}]"

    if not results:
        return f"No search results found for '{query}'."
    return _format_results(results, limit)


class WebSearchTool(BaseTool):
    """Tool to search the web using SearXNG."""

    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date facts, documentation, or news. "
        "Do NOT call this tool for basic conversation, greetings, or answering questions about your own errors."
    )
    is_concurrent_safe: bool = True
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query keywords, e.g. 'latest AI news' or 'FastAPI tutorial'.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to retrieve (default: 5, max: 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 5, **kwargs: Any) -> str:
        return await search_searxng(query=query, num_results=num_results)
