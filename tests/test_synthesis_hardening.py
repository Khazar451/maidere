import re
import unittest
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from unittest.mock import AsyncMock, patch

from core.agent import (
    get_system_prompt,
    normalize_citations_and_sources,
    strip_critic_fourth_wall_leaks,
    restore_markdown_urls,
    refine_node,
    critique_node,
)


class TestSynthesisHardening(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying fixes for:
    - Breakdown 1: Naked Integer Regex Evasion
    - Breakdown 2: Source Mismatch Hallucination
    - Breakdown 3: Fourth-Wall Break (Critic Leak)
    - Breakdown 4: Surprise Variable Synthesis Bug
    """

    def test_breakdown_1_naked_integer_repair(self):
        """Verify normalize_citations_and_sources wraps naked integers in [N] and normalizes Sources."""
        raw_text = (
            "A YouTube video demonstrates how to use Perplexity for competitor research... "
            "highlighting its utility for competitive analysis and cybersecurity firms 5. "
            "Furthermore, another study confirms performance 2.\n\n"
            "In step 2, we evaluated model 3 and checked table 1.\n\n"
            "## Sources\n"
            "[1] [Official Website](https://perplexity.ai)\n"
            "2 [Benchmark Results](https://benchmarks.com)\n"
            "5 Feeding a large local codebase to the model possible? - Reddit, https://www.reddit.com/r/perplexity_ai\n"
        )

        normalized = normalize_citations_and_sources(raw_text)

        # Body naked numbers wrapped
        self.assertIn("cybersecurity firms [5].", normalized)
        self.assertIn("performance [2].", normalized)

        # Excluded words preserved as regular numbers
        self.assertIn("step 2", normalized)
        self.assertIn("model 3", normalized)
        self.assertIn("table 1", normalized)

        # Sources block normalized
        self.assertIn("[2] [Benchmark Results](https://benchmarks.com)", normalized)
        self.assertIn(
            "[5] [Feeding a large local codebase to the model possible? - Reddit](https://www.reddit.com/r/perplexity_ai)",
            normalized,
        )

    def test_breakdown_1_comma_separated_sources_repair(self):
        """Verify comma-separated plain sources are split and normalized into lines."""
        raw_text = (
            "Model achieved high score [1] and low latency [2].\n\n"
            "## Sources\n"
            "[1] [Paper](https://arxiv.org/1), 2 [Code](https://github.com/test), [3] https://docs.com\n"
        )

        normalized = normalize_citations_and_sources(raw_text)
        self.assertIn("[1] [Paper](https://arxiv.org/1)", normalized)
        self.assertIn("[2] [Code](https://github.com/test)", normalized)
        self.assertIn("[3] [https://docs.com](https://docs.com)", normalized)

    def test_breakdown_3_strip_critic_fourth_wall_leaks(self):
        """Verify fourth-wall critic and review leaks are cleanly stripped from final output."""
        leaky_text = (
            "Perplexity AI combines search indexing with direct LLM synthesis [1]. "
            "It allows real-time web retrieval.\n\n"
            "This summary provides a detailed and accurate analysis of Perplexity AI, adhering to the corrections and verifications suggested by the Socratic Critic.\n\n"
            "## Sources\n"
            "[1] [Official Website](https://perplexity.ai)"
        )

        cleaned = strip_critic_fourth_wall_leaks(leaky_text)

        self.assertNotIn("Socratic Critic", cleaned)
        self.assertNotIn("adhering to the corrections", cleaned)
        self.assertIn("Perplexity AI combines search indexing with direct LLM synthesis [1].", cleaned)
        self.assertIn("## Sources", cleaned)

    def test_breakdown_1_and_2_and_4_system_prompt_directives(self):
        """Verify system prompt contains hardened citation, anti-mismatch, and surprise variable rules."""
        prompt = get_system_prompt()
        now = datetime.now()
        formatted = prompt.format(
            memory_context="No memories",
            current_date=now.strftime("%A, %B %d, %Y"),
            current_year=str(now.year),
        )

        # Breakdown 1: Naked Integer prohibition
        self.assertIn("CRITICAL: Every single citation MUST be wrapped in square brackets", formatted)
        self.assertIn("FORBIDDEN from using naked numbers like '1' or '2'", formatted)

        # Breakdown 2: Source-to-claim alignment
        self.assertIn("Strict Source-to-Claim Mapping (Anti-Hallucination)", formatted)
        self.assertIn("Every inline citation [N] MUST directly support the specific claim it is attached to", formatted)
        self.assertIn("attributing claims to the wrong media or platform", formatted)

        # Breakdown 4: Surprise variable prohibition
        self.assertIn("No Surprise Variables or Metrics in Summaries", formatted)
        self.assertIn("Every metric, statistic, number, or variable included in a summary table", formatted)
        self.assertIn("FORBIDDEN from introducing novel statistics or surprise variables in concluding summary sections", formatted)

    async def test_refine_node_strips_fourth_wall_and_enforces_hardening(self):
        """Verify refine_node strips fourth-wall leaks even if LLM includes them in response."""
        state = {
            "messages": [
                HumanMessage(content="Analyze Perplexity AI architecture"),
                AIMessage(content="Draft about Perplexity architecture."),
            ],
            "critique": "1. Need citation brackets. 2. Verify query volume in body.",
            "thread_id": "test-leak-thread",
            "model": "qwen2.5:7b-instruct",
            "num_ctx": 8192,
        }

        mock_leaky_llm_response = {
            "message": {
                "role": "assistant",
                "content": (
                    "Perplexity uses hybrid search and LLM synthesis [1]. "
                    "Competitor analysis utility is noted across cybersecurity firms 5.\n\n"
                    "This summary provides a detailed and accurate analysis of Perplexity AI, adhering to the corrections and verifications suggested by the Socratic Critic.\n\n"
                    "## Sources\n"
                    "[1] [Perplexity](https://perplexity.ai)\n"
                    "5 [Cybersecurity Report](https://cyber.com)\n"
                ),
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_leaky_llm_response):
            result = await refine_node(state)
            messages = result["messages"]
            self.assertEqual(len(messages), 1)
            final_text = messages[0].content

            # Fourth-wall leak must be completely absent
            self.assertNotIn("Socratic Critic", final_text)
            self.assertNotIn("adhering to the corrections", final_text)

            # Naked citation 5 must be wrapped in [5]
            self.assertIn("cybersecurity firms [5].", final_text)
            self.assertIn("[5] [Cybersecurity Report](https://cyber.com)", final_text)


if __name__ == "__main__":
    unittest.main()
