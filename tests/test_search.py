"""Tests for WebSearchTool (SearXNG client) and search-and-browse integration."""

import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from maidere.core.agent import create_graph
from maidere.core.config import settings
from maidere.tools.web_search import WebSearchTool


class TestWebSearchTool(unittest.IsolatedAsyncioTestCase):
    """Test SearXNG web search tool."""

    async def asyncSetUp(self):
        self.search_tool = WebSearchTool()

    async def test_empty_query(self):
        """Test empty search query returns error."""
        result = await self.search_tool.execute(query="   ")
        self.assertIn("Error: Empty search query provided.", result)

    @patch("httpx.AsyncClient.get")
    async def test_successful_search_formatting(self, mock_get):
        """Test parsing and formatting of SearXNG JSON search results."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "query": "python async patterns",
                "results": [
                    {
                        "title": "Async IO in Python: A Complete Guide",
                        "url": "https://realpython.com/async-io-python/",
                        "content": "Learn how to use Python asyncio effectively for concurrent programming.",
                    },
                    {
                        "title": "Python Async Patterns and Best Practices",
                        "url": "https://example.com/async-patterns",
                        "content": "A detailed discussion on task groups, timeouts, and structured concurrency.",
                    },
                ],
            },
            request=httpx.Request("GET", f"{settings.searxng_url}/search"),
        )
        mock_get.return_value = mock_response

        result = await self.search_tool.execute(query="python async patterns", num_results=2)

        self.assertIn("1. Async IO in Python: A Complete Guide", result)
        self.assertIn("URL: https://realpython.com/async-io-python/", result)
        self.assertIn("Snippet: Learn how to use Python asyncio effectively", result)
        self.assertIn("2. Python Async Patterns and Best Practices", result)

    @patch("httpx.AsyncClient.get")
    async def test_no_results_found(self, mock_get):
        """Test when SearXNG returns an empty results list."""
        mock_response = httpx.Response(
            status_code=200,
            json={"query": "nonexistentquery12345", "results": []},
            request=httpx.Request("GET", f"{settings.searxng_url}/search"),
        )
        mock_get.return_value = mock_response

        result = await self.search_tool.execute(query="nonexistentquery12345")
        self.assertIn("No search results found", result)

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    async def test_searxng_connection_error_with_fallback_down(self, mock_get, mock_post):
        """Graceful error when SearXNG and every fallback provider are unreachable."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = await self.search_tool.execute(query="test connection failure")
        self.assertTrue(result.startswith("[Search Unavailable:"))

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    async def test_fallback_disabled_never_leaves_machine(self, mock_get, mock_post):
        """With the fallback switched off, an offline SearXNG makes no outbound request."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        with patch.object(settings, "search_fallback_enabled", False):
            result = await self.search_tool.execute(query="private query")

        self.assertIn("[Search Unavailable: SearXNG container is offline]", result)
        mock_post.assert_not_called()

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    async def test_duckduckgo_fallback_on_connect_error(self, mock_get, mock_post):
        """An offline SearXNG falls back to DuckDuckGo and returns formatted results."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        html = (
            '<div class="result"><a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&rut=x">'
            "asyncio &mdash; Asynchronous I/O</a>"
            '<a class="result__snippet" href="#">Coroutines and event loops.</a></div>'
        )
        mock_post.return_value = httpx.Response(
            200, text=html, request=httpx.Request("POST", "https://html.duckduckgo.com/html/")
        )

        with patch.object(settings, "tavily_api_key", ""), patch.object(settings, "serper_api_key", ""):
            result = await self.search_tool.execute(query="python asyncio")

        self.assertIn("1. asyncio", result)
        self.assertIn("URL: https://docs.python.org/3/library/asyncio.html", result)
        self.assertIn("Snippet: Coroutines and event loops.", result)

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    async def test_fallback_on_timeout_prefers_configured_cloud_provider(self, mock_get, mock_post):
        """A configured Tavily key is tried before DuckDuckGo; DuckDuckGo is not contacted on success."""
        mock_get.side_effect = httpx.ReadTimeout("slow")
        mock_post.return_value = httpx.Response(
            200,
            json={"results": [{"title": "Tavily hit", "url": "https://example.com/t", "content": "from tavily"}]},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )

        with patch.object(settings, "tavily_api_key", "tvly-test"):
            result = await self.search_tool.execute(query="anything")

        self.assertIn("Tavily hit", result)
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.args[0], "https://api.tavily.com/search")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    async def test_failed_cloud_provider_falls_through_to_next(self, mock_get, mock_post):
        """If the cloud provider errors, DuckDuckGo is still attempted."""
        mock_get.side_effect = httpx.ConnectError("down")
        ddg = httpx.Response(
            200,
            text='<a class="result__a" href="https://example.org/x">Example</a>',
            request=httpx.Request("POST", "https://html.duckduckgo.com/html/"),
        )
        mock_post.side_effect = [httpx.ConnectError("tavily down"), ddg]

        with patch.object(settings, "tavily_api_key", "tvly-test"), patch.object(settings, "serper_api_key", ""):
            result = await self.search_tool.execute(query="anything")

        self.assertIn("URL: https://example.org/x", result)


class TestSearchAndBrowseAgentIntegration(unittest.IsolatedAsyncioTestCase):
    """Test full multi-step agent flow: search -> browse page -> synthesize answer."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

    async def asyncTearDown(self):
        import os

        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_agent_search_and_browse_synthesis(self):
        """Test agent multi-step: user asks news -> LLM calls web_search -> LLM calls browser -> LLM synthesizes."""
        graph, checkpointer_ctx = await create_graph(self.temp_db_path)

        mock_responses = [
            # 1. LLM decides to search the web
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "latest developments in autonomous AI agents"},
                            }
                        }
                    ],
                }
            },
            # 2. LLM receives search results, decides to browse the top article
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "browser",
                                "arguments": {"url": "https://tech-news.example.com/ai-agents-2026"},
                            }
                        }
                    ],
                }
            },
            # 3. LLM synthesizes the final answer
            {
                "message": {
                    "role": "assistant",
                    "content": "According to the latest reports, local autonomous AI agents now run entirely with zero API costs using local LLMs and SQLite vector storage.",
                    "tool_calls": [],
                }
            },
        ]

        mock_searxng_res = httpx.Response(
            status_code=200,
            json={
                "results": [
                    {
                        "title": "Autonomous AI Agents Breakthrough in 2026",
                        "url": "https://tech-news.example.com/ai-agents-2026",
                        "content": "Breakthrough in local AI architecture allows full agent autonomy on consumer GPUs.",
                    }
                ]
            },
            request=httpx.Request("GET", f"{settings.searxng_url}/search"),
        )

        with (
            patch("maidere.core.llm.chat", side_effect=mock_responses) as mock_chat,
            patch("httpx.AsyncClient.get", return_value=mock_searxng_res),
            patch(
                "maidere.tools.browser.browse_url",
                new_callable=AsyncMock,
                return_value="Article Body: Autonomous agents operate 24/7 with zero VRAM thrashing using fastembed and sqlite-vec.",
            ),
        ):
            config = {"configurable": {"thread_id": "test-search-browse-thread"}}
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="What happened in tech news today regarding AI agents?")],
                    "memory_context": "No relevant memories.",
                    "thread_id": "test-search-browse-thread",
                },
                config=config,
            )

            # 3 turns in multi-step agent flow
            self.assertEqual(mock_chat.call_count, 3)

            final_message = result["messages"][-1]
            self.assertIsInstance(final_message, AIMessage)
            self.assertIn("local autonomous AI agents", final_message.content)

        await checkpointer_ctx.__aexit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
