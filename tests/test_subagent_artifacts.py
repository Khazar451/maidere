"""Tests for Phase 1: Subagent Artifact Bus & Handoff Isolation."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from core.subagent import (
    SubAgentDefinition,
    SubAgentResult,
    generate_handoff_brief,
    load_subagent_definitions,
    prune_artifacts,
    run_subagent,
    write_subagent_artifact,
)
from tools.delegate import DelegateTaskTool


class TestSubagentArtifacts(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying artifact creation, pruning, and context reduction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_dir = Path(self.temp_dir.name) / ".maidere" / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reviewer_and_staffer_personas_registered(self):
        """Verify that reviewer and staffer personas are registered in builtins."""
        defs = load_subagent_definitions()
        self.assertIn("reviewer", defs)
        self.assertIn("staffer", defs)
        self.assertIn("researcher", defs)
        self.assertIn("code-reviewer", defs)

        self.assertEqual(defs["reviewer"].name, "reviewer")
        self.assertEqual(defs["staffer"].name, "staffer")
        self.assertIn("read_file", defs["reviewer"].tools)
        self.assertIn("web_search", defs["staffer"].tools)

    def test_artifact_file_creation(self):
        """Verify that subagent outputs are written to collision-proof artifact files."""
        task = "Analyze RTX 5060 Tensor Cores"
        content = "## Summary\n- 8GB GDDR7 VRAM\n- 384 GB/s bandwidth\n[NVIDIA](https://nvidia.com)"
        tools_used = [{"tool_name": "web_search", "success": True}]

        file_path = write_subagent_artifact(
            task=task,
            content=content,
            subagent_type="researcher",
            tools_used=tools_used,
            duration_ms=450,
            turns_used=2,
            model="qwen2.5:7b-instruct",
            artifacts_dir=self.artifacts_dir,
        )

        self.assertTrue(file_path.exists())
        self.assertTrue(file_path.name.endswith(".md"))
        self.assertIn("researcher", file_path.name)

        text = file_path.read_text(encoding="utf-8")
        self.assertIn("Sub-Agent Research Artifact: Analyze RTX 5060 Tensor Cores", text)
        self.assertIn("8GB GDDR7 VRAM", text)
        self.assertIn("450ms", text)
        self.assertIn("qwen2.5:7b-instruct", text)

    def test_collision_prevention(self):
        """Verify that two subagents writing concurrently create distinct artifact files."""
        path1 = write_subagent_artifact(
            task="Task 1",
            content="Content 1",
            subagent_type="researcher",
            tools_used=[],
            duration_ms=100,
            turns_used=1,
            model="test-model",
            artifacts_dir=self.artifacts_dir,
        )
        path2 = write_subagent_artifact(
            task="Task 2",
            content="Content 2",
            subagent_type="researcher",
            tools_used=[],
            duration_ms=100,
            turns_used=1,
            model="test-model",
            artifacts_dir=self.artifacts_dir,
        )

        self.assertNotEqual(path1, path2)
        self.assertTrue(path1.exists())
        self.assertTrue(path2.exists())

    def test_artifact_pruning_ttl_and_cap(self):
        """Verify that prune_artifacts respects both TTL (max_age_days) and capacity cap (max_artifacts)."""
        now = time.time()

        # Create 8 dummy artifacts
        created_paths = []
        for i in range(8):
            p = self.artifacts_dir / f"test_artifact_{i:02d}.md"
            p.write_text(f"Artifact {i}", encoding="utf-8")
            created_paths.append(p)

        # Make 2 files look 20 days old (older than 14-day default)
        old_time = now - (20 * 86400)
        import os
        os.utime(created_paths[0], (old_time, old_time))
        os.utime(created_paths[1], (old_time, old_time))

        # Prune with max_artifacts=4, max_age_days=14
        pruned = prune_artifacts(
            artifacts_dir=self.artifacts_dir,
            max_artifacts=4,
            max_age_days=14,
        )

        # 2 old files pruned via TTL + 2 more files pruned to satisfy max_artifacts=4
        self.assertGreaterEqual(pruned, 4)
        remaining = list(self.artifacts_dir.glob("*.md"))
        self.assertLessEqual(len(remaining), 4)
        self.assertNotIn(created_paths[0], remaining)
        self.assertNotIn(created_paths[1], remaining)

    def test_brief_generation_context_reduction(self):
        """Verify that generate_handoff_brief delivers >= 70% context compression on dense outputs."""
        task = "Deep research on LLM inference engines in 2026"
        dummy_file = self.artifacts_dir / "20260925_research_test.md"
        dummy_file.write_text("dummy", encoding="utf-8")

        # Simulate a dense 4,500-character research payload with tables, metrics, and citations
        dense_paragraphs = [
            "### Architectural Overview\n"
            "High-throughput LLM inference in 2026 relies heavily on vLLM and TensorRT-LLM with PagedAttention v3. "
            "Quantization formats like FP4 and FP8 have reduced memory footprint by 50% without quality degradation.",
            "- vLLM achieves 14,200 tokens/sec across 8x H200 GPUs [vLLM Benchmarks](https://vllm.ai/benchmarks)",
            "- TensorRT-LLM delivers 2.1x lower latency on FP8 GEMM kernels [NVIDIA Dev](https://developer.nvidia.com/trt)",
            "- SGLang introduces RadixAttention caching 85% of repeated prompt prefixes [SGLang Docs](https://sglang.ai)",
            "- Memory bandwidth saturation reaches 4.8 TB/s on Blackwell architecture [Blackwell Architecture](https://nvidia.com/blackwell)",
            "- Average TTFT (Time-to-First-Token) reduced to 18ms for 70B parameter models at batch size 32",
            "- Enterprise deployments report $1.2 million quarterly savings on cloud compute grants",
            "- Power efficiency measured at 125 tokens per Joule under sustained load",
        ]
        full_content = "\n\n".join(dense_paragraphs * 5)  # ~4,700 characters
        raw_char_count = len(full_content)
        self.assertGreater(raw_char_count, 4000)

        brief = generate_handoff_brief(
            task=task,
            full_content=full_content,
            artifact_path=dummy_file,
            tools_used=[{"tool_name": "web_search", "success": True}],
            duration_ms=1200,
            turns_used=4,
            subagent_type="researcher",
        )

        brief_char_count = len(brief)
        # Verify brief is compact
        self.assertLessEqual(brief_char_count, 1400)
        self.assertGreaterEqual(brief_char_count, 300)

        # Context reduction calculation: (raw - brief) / raw
        reduction_pct = (raw_char_count - brief_char_count) / raw_char_count
        self.assertGreaterEqual(
            reduction_pct,
            0.70,
            f"Expected at least 70% context reduction, got {reduction_pct:.1%}",
        )

        # Verify key elements exist in brief
        self.assertIn("Key Findings:", brief)
        self.assertIn("Full Artifact Stored At:", brief)
        self.assertIn("https://vllm.ai/benchmarks", brief)

    @patch("core.subagent.llm.chat")
    @patch("core.subagent._store_subagent_memory", new_callable=AsyncMock)
    async def test_run_subagent_produces_artifact_and_brief(self, mock_store, mock_chat):
        """Test full run_subagent execution returns artifact_path and brief."""
        mock_chat.return_value = {
            "message": {
                "role": "assistant",
                "content": (
                    "### Findings\n"
                    "- Finding 1: Local Qwen 2.5 7B achieves 35 tok/s on RTX 5060.\n"
                    "- Finding 2: FastEmbed runs on ONNX CPU with 0 MB VRAM [FastEmbed Docs](https://fastembed.org).\n"
                    "- Finding 3: SQLite WAL mode supports concurrent reads and writes."
                ),
            }
        }

        with patch("core.subagent.settings.agent_workspace", self.temp_dir.name):
            result = await run_subagent(
                task="Test subagent artifact production",
                subagent_type="researcher",
                max_turns=1,
            )

        self.assertTrue(result.success)
        self.assertNotEqual(result.artifact_path, "")
        self.assertTrue(Path(result.artifact_path).exists())
        self.assertIn("Finding 1", result.brief)
        self.assertIn("Full Artifact Stored At", result.brief)

    async def test_delegate_task_formats_brief(self):
        """Test DelegateTaskTool returns the brief and artifact pointer."""
        tool = DelegateTaskTool()
        tool.reset_spawn_count()

        mock_result = SubAgentResult(
            task="Evaluate security diff",
            summary="Full 3000-char markdown text here...",
            subagent_type="reviewer",
            tools_used=[{"tool_name": "read_file", "success": True}],
            duration_ms=300,
            success=True,
            turns_used=2,
            artifact_path="/workspace/.maidere/artifacts/review_01.md",
            brief="• Finding 1: Safe subprocess args used.\n• Finding 2: Zero shell injection.",
        )

        with patch("tools.delegate.run_subagent", new_callable=AsyncMock, return_value=mock_result):
            res_str = await tool.execute(prompt="Evaluate security diff", subagent_type="reviewer")

        self.assertIn("[SUB-AGENT RESEARCH RESULT]", res_str)
        self.assertIn("Agent Type: reviewer", res_str)
        self.assertIn("Artifact: /workspace/.maidere/artifacts/review_01.md", res_str)
        self.assertIn("Finding 1: Safe subprocess args used", res_str)
        self.assertNotIn("Full 3000-char markdown text here", res_str)


if __name__ == "__main__":
    unittest.main()
