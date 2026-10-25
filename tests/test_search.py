"""Tests for WebSearchTool (SearXNG client) and search-and-browse integration."""

import tempfile
import unittest
from pathlib import Path
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
        """User asks for news -> LLM calls web_search -> call is redirected to a researcher -> LLM synthesizes."""
        graph, checkpointer_ctx = await create_graph(self.temp_db_path)

        mock_responses = [
            # 1. LLM decides to search the web directly (the orchestrator has no raw web access)
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
            # 2. LLM synthesizes the final answer from the researcher's findings
            {
                "message": {
                    "role": "assistant",
                    "content": "According to the latest reports, local autonomous AI agents now run entirely with zero API costs using local LLMs and SQLite vector storage.",
                    "tool_calls": [],
                }
            },
        ]

        research = AsyncMock(
            return_value="[SUB-AGENT RESEARCH RESULT]\nAutonomous agents operate 24/7 using fastembed and sqlite-vec."
        )
        with (
            patch("maidere.core.llm.chat", side_effect=mock_responses) as mock_chat,
            patch("maidere.tools.delegate.AgentTool.execute", research),
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

            self.assertEqual(mock_chat.call_count, 2)
            # The direct web_search call became a researcher delegation with the same query.
            research.assert_awaited_once()
            kwargs = research.await_args.kwargs
            self.assertEqual(kwargs["subagent_type"], "researcher")
            self.assertIn("latest developments in autonomous AI agents", kwargs["prompt"])

            final_message = result["messages"][-1]
            self.assertIsInstance(final_message, AIMessage)
            self.assertIn("local autonomous AI agents", final_message.content)

        await checkpointer_ctx.__aexit__(None, None, None)


FIXTURES = Path(__file__).parent / "fixtures"


class TestDuckDuckGoParser(unittest.IsolatedAsyncioTestCase):
    """The fallback scrapes DuckDuckGo's HTML endpoint, so pin it to a realistic page."""

    @staticmethod
    def _parse(html: str) -> list[dict]:
        from maidere.tools.web_search import _DuckDuckGoParser

        parser = _DuckDuckGoParser()
        parser.feed(html)
        return parser.results

    async def _search(self, html: str) -> str:
        response = httpx.Response(200, text=html, request=httpx.Request("POST", "https://html.duckduckgo.com/html/"))
        with (
            patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("down")),
            patch("httpx.AsyncClient.post", return_value=response),
            patch.object(settings, "tavily_api_key", ""),
            patch.object(settings, "serper_api_key", ""),
        ):
            return await WebSearchTool().execute(query="python asyncio", num_results=10)

    async def test_extracts_real_results_in_order(self):
        result = await self._search((FIXTURES / "duckduckgo_results.html").read_text())

        urls = [line.split("URL: ", 1)[1] for line in result.splitlines() if line.strip().startswith("URL:")]
        self.assertEqual(
            urls,
            [
                "https://docs.python.org/3/library/asyncio.html",
                "https://realpython.com/async-io-python/?utm_source=ddg&page=2",
                "https://example.org/direct-link",
            ],
        )
        self.assertTrue(result.startswith("1. asyncio"))

    async def test_sponsored_results_are_dropped(self):
        result = await self._search((FIXTURES / "duckduckgo_results.html").read_text())

        self.assertNotIn("Sponsored", result)
        self.assertNotIn("y.js", result)
        self.assertNotIn("duckduckgo.com", result)

    async def test_titles_and_snippets_are_clean_text(self):
        results = self._parse((FIXTURES / "duckduckgo_results.html").read_text())
        real = [r for r in results if "y.js" not in r["url"]]

        self.assertEqual(" ".join(real[0]["title"].split()), "asyncio — Asynchronous I/O — Python 3.12.4 documentation")
        snippet = " ".join(real[0]["content"].split())
        self.assertTrue(
            snippet.startswith("asyncio is a library to write concurrent code using the async/await syntax.")
        )
        self.assertNotIn("<b>", snippet)
        self.assertEqual(
            " ".join(real[1]["content"].split()),
            "Learn how async IO works: coroutines, event loops & tasks. Tom & Jerry's guide to await.",
        )
        self.assertEqual(" ".join(real[1]["title"].split()), "Async IO in Python: A Complete Walkthrough – Real Python")

    async def test_result_without_snippet_still_listed(self):
        result = await self._search((FIXTURES / "duckduckgo_results.html").read_text())

        self.assertIn("Result with a direct, non-redirect link", result)

    async def test_snippets_do_not_leak_between_results(self):
        results = [r for r in self._parse((FIXTURES / "duckduckgo_results.html").read_text()) if "y.js" not in r["url"]]

        self.assertEqual(results[2]["content"].strip(), "")
        self.assertNotIn("Learn how", results[0]["content"])

    async def test_page_without_results_reports_none_found(self):
        result = await self._search("<html><body><div class='no-results'>No results.</div></body></html>")

        self.assertTrue(result.startswith("[Search Unavailable:"))

    async def test_formatted_output_has_single_line_fields(self):
        result = await self._search((FIXTURES / "duckduckgo_results.html").read_text())

        for block in result.split("\n\n"):
            self.assertEqual(len(block.splitlines()), 4, block)
