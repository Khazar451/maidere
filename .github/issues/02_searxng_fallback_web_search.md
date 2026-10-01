---
title: "[Feature]: Fallback Web Search Engine When SearXNG Container is Offline"
labels: ["enhancement", "tools", "resilience"]
---

### Problem Statement
Currently, `tools/web_search.py` strictly relies on a local SearXNG Docker container running on `http://localhost:8080`. If Docker is stopped, the container is unhealthy, or the user runs Maidere standalone without starting Docker services, `search_searxng` catches `httpx.ConnectError` and returns:

```text
[Search Unavailable: SearXNG container is offline]
```

This causes agent deep research and web queries to immediately fail, degrading autonomous workflows.

### Proposed Solution
Implement a resilient multi-tier search fallback strategy:

1. **Primary**: Local private SearXNG instance (`http://localhost:8080/search`).
2. **Secondary Fallback**: If SearXNG is unreachable (`httpx.ConnectError` or `httpx.TimeoutException`), seamlessly fall back to DuckDuckGo HTML / Lite API or a public SearXNG instance without requiring an API key.
3. **Optional Cloud Provider**: If `TAVILY_API_KEY` or `SERPER_API_KEY` is configured in `.env`, offer that as an optional high-accuracy fallback.

```python
async def search_searxng(query: str, num_results: int = 5) -> str:
    try:
        # Try primary SearXNG
        return await _query_searxng(query, num_results)
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("searxng_offline_falling_back_to_ddg", query=query)
        return await _query_duckduckgo_fallback(query, num_results)
```

### Alternatives Considered
- Returning error and instructing user to start Docker: disrupts autonomous operations when working without root/Docker permissions.

### Acceptance Criteria
- [ ] When SearXNG container is offline, `web_search` returns live formatted search results via fallback provider.
- [ ] Structured warning is logged indicating fallback execution.
- [ ] Unit tests in `tests/test_search.py` verify fallback triggers cleanly upon SearXNG connection timeout.
