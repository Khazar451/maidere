"""Async web search tool using SearXNG JSON API."""

from typing import Any
import httpx
import structlog

from core.config import settings
from tools.base import BaseTool

logger = structlog.get_logger()


async def search_searxng(query: str, num_results: int = 5) -> str:
    """Execute search query against SearXNG instance."""
    if not query or not query.strip():
        return "Error: Empty search query provided."

    limit = max(1, min(num_results, 10))
    endpoint = f"{settings.searxng_url.rstrip('/')}/search"
    params = {
        "q": query.strip(),
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await logger.ainfo("searxng_request", query=query, endpoint=endpoint)
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return f"No search results found for '{query}'."

            formatted_results = []
            for i, item in enumerate(results[:limit], 1):
                title = item.get("title", "No title")
                url = item.get("url", "")
                content = item.get("content", "").strip()
                formatted_results.append(
                    f"{i}. {title}\n   Markdown Link: [{title}]({url})\n   URL: {url}\n   Snippet: {content}"
                )

            return "\n\n".join(formatted_results)
    except (httpx.ConnectError, httpx.TimeoutException):
        await logger.awarn("searxng_offline", endpoint=endpoint)
        return "[Search Unavailable: SearXNG container is offline]"
    except Exception as e:
        await logger.aerror("searxng_error", query=query, error=str(e))
        return f"[Search Error: {str(e)}]"


class WebSearchTool(BaseTool):
    """Tool to search the web using SearXNG."""

    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date facts, documentation, or news. "
        "Do NOT call this tool for basic conversation, greetings, or answering questions about your own errors."
    )
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

