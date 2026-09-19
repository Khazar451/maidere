"""Tests for WebSearchTool (SearXNG client) and search-and-browse integration."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from langchain_core.messages import AIMessage, HumanMessage

from core.agent import create_graph
from core.config import settings
from tools.registry import execute_tool, get_tool
from tools.web_search import WebSearchTool, search_searxng


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

    @patch("httpx.AsyncClient.get")
    async def test_searxng_connection_error(self, mock_get):
        """Test graceful error message when SearXNG is unreachable."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        result = await self.search_tool.execute(query="test connection failure")
        self.assertIn("[Search Unavailable: SearXNG container is offline]", result)



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

        with patch("core.llm.chat", side_effect=mock_responses) as mock_chat, \
             patch("httpx.AsyncClient.get", return_value=mock_searxng_res), \
             patch("tools.browser.browse_url", new_callable=AsyncMock, return_value="Article Body: Autonomous agents operate 24/7 with zero VRAM thrashing using fastembed and sqlite-vec."):

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
