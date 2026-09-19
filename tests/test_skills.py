"""Tests for Skill Management, Discovery, Parsing, Activation Tools, and Agent Integration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from core.agent import create_graph, remember_node
from core.skills import (
    Skill,
    get_skills_overview,
    load_all_skills,
    match_skills,
    parse_skill_file,
)
from tools.registry import execute_tool, get_tool
from tools.skills import ListSkillsTool, LoadSkillTool, list_available_skills, load_skill


class TestSkillDiscoveryAndParsing(unittest.TestCase):
    """Test skill parsing and discovery."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.skill_dir = Path(self.temp_dir.name) / "test-skill"
        self.skill_dir.mkdir(parents=True)
        self.skill_file = self.skill_dir / "SKILL.md"
        self.skill_file.write_text(
            """---
name: custom-testing
description: Specialized testing instructions for end-to-end flows.
triggers:
  - test
  - e2e
  - validate
---

# Custom Testing Instructions

1. Always run tests in isolated sandbox.
2. Verify exit code equals 0.
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_skill_file(self):
        """Test parsing YAML frontmatter and markdown body."""
        skill = parse_skill_file(self.skill_file)
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "custom-testing")
        self.assertEqual(
            skill.description,
            "Specialized testing instructions for end-to-end flows.",
        )
        self.assertIn("test", skill.triggers)
        self.assertIn("e2e", skill.triggers)
        self.assertIn("validate", skill.triggers)
        self.assertIn("Always run tests in isolated sandbox", skill.instructions)

    def test_builtin_skills_loaded(self):
        """Test that default built-in skills (git-workflow, api-tester, data-analysis, deep-research) are discovered."""
        skills = load_all_skills()
        self.assertIn("git-workflow", skills)
        self.assertIn("api-tester", skills)
        self.assertIn("data-analysis", skills)
        self.assertIn("deep-research", skills)
        self.assertIn("agent-ethics", skills)
        self.assertIn("obsidian", skills)

        git_skill = skills["git-workflow"]
        self.assertIn("commit", git_skill.triggers)
        self.assertIn("Conventional Commits", git_skill.instructions)

        deep_skill = skills["deep-research"]
        self.assertIn("obsidian", deep_skill.triggers)
        self.assertIn("research", deep_skill.triggers)
        self.assertIn("Deep Research Directives", deep_skill.instructions)
        self.assertIn("delegate_task", deep_skill.instructions)



        ethics_skill = skills["agent-ethics"]
        self.assertIn("ethics", ethics_skill.triggers)
        self.assertIn("privacy", ethics_skill.triggers)
        self.assertIn("Pre-Storage Redaction", ethics_skill.instructions)

        obs_skill = skills["obsidian"]
        self.assertIn("vault", obs_skill.triggers)
        self.assertIn("Obsidian Markdown Guidelines", obs_skill.instructions)


    def test_get_skills_overview(self):
        """Test overview string generation."""
        overview = get_skills_overview()
        self.assertIn("`git-workflow`:", overview)
        self.assertIn("`api-tester`:", overview)
        self.assertIn("`data-analysis`:", overview)
        self.assertIn("`deep-research`:", overview)
        self.assertIn("`agent-ethics`:", overview)
        self.assertIn("`obsidian`:", overview)



    def test_match_skills(self):
        """Test keyword trigger matching."""
        matches = match_skills("How should I format this git commit message?")
        match_names = [s.name for s in matches]
        self.assertIn("git-workflow", match_names)

        data_matches = match_skills("Please analyze this csv data file with pandas")
        data_names = [s.name for s in data_matches]
        self.assertIn("data-analysis", data_names)

        research_matches = match_skills("Conduct deep research on neural search and write note in obsidian format")
        research_names = [s.name for s in research_matches]
        self.assertIn("deep-research", research_names)

        no_matches = match_skills("Hello, what is the weather today?")
        self.assertEqual(len(no_matches), 0)



class TestSkillTools(unittest.IsolatedAsyncioTestCase):
    """Test list_available_skills and load_skill tools."""

    async def asyncSetUp(self):
        self.list_tool = ListSkillsTool()
        self.load_tool = LoadSkillTool()

    async def test_list_available_skills(self):
        """Test listing skills tool."""
        res = await self.list_tool.execute()
        self.assertIn("Available Skills:", res)
        self.assertIn("git-workflow", res)
        self.assertIn("api-tester", res)

    async def test_load_skill_success(self):
        """Test loading an existing skill."""
        res = await self.load_tool.execute(skill_name="git-workflow")
        self.assertIn("=== LOADED SKILL: git-workflow ===", res)
        self.assertIn("Conventional Commits", res)
        self.assertIn("Never stage `.env`", res)


    async def test_load_skill_not_found(self):
        """Test loading a nonexistent skill returns a clean error."""
        res = await self.load_tool.execute(skill_name="quantum-teleportation")
        self.assertIn("Error: Skill 'quantum-teleportation' not found", res)
        self.assertIn("Available skills:", res)

    async def test_execute_tool_dispatcher(self):
        """Test executing skill tools via registry dispatcher."""
        res = await execute_tool(name="list_available_skills", args={})
        self.assertIn("Available Skills:", res)

        res2 = await execute_tool(name="load_skill", args={"skill_name": "data-analysis"})
        self.assertIn("=== LOADED SKILL: data-analysis ===", res2)


class TestSkillsAgentIntegration(unittest.IsolatedAsyncioTestCase):
    """Test auto-matching in remember_node and agent graph invocation."""

    async def test_remember_node_skill_matching(self):
        """Test remember_node auto-injects [ACTIVE SKILLS] when user message triggers a skill."""
        state = {
            "messages": [HumanMessage(content="Can you help me prepare a git branch and commit?")],
            "thread_id": "test-skills-thread",
        }
        res = await remember_node(state)
        self.assertIn("memory_context", res)
        self.assertIn("[ACTIVE SKILLS]", res["memory_context"])
        self.assertIn("git-workflow", res["memory_context"])
        self.assertIn("Conventional Commits", res["memory_context"])

    async def test_agent_graph_with_skill_load(self):
        """Test full LangGraph loop where LLM loads a skill dynamically."""
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db_path = temp_db.name
        temp_db.close()

        graph, checkpointer_ctx = await create_graph(temp_db_path)

        mock_responses = [
            # 1. LLM decides to load the api-tester skill
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "load_skill",
                                "arguments": {"skill_name": "api-tester"},
                            }
                        }
                    ],
                }
            },
            # 2. LLM finishes with instructions from loaded skill
            {
                "message": {
                    "role": "assistant",
                    "content": "Loaded api-tester skill. I will now test both 200 OK and 404 status codes.",
                    "tool_calls": [],
                }
            },
        ]

        with patch("core.llm.chat", side_effect=mock_responses) as mock_chat:
            config = {"configurable": {"thread_id": "test-graph-skill-thread"}}
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="I need to test an API endpoint")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-graph-skill-thread",
                },
                config=config,
            )

            self.assertEqual(mock_chat.call_count, 2)
            final_msg = result["messages"][-1]
            self.assertIsInstance(final_msg, AIMessage)
            self.assertIn("Loaded api-tester skill", final_msg.content)

        await checkpointer_ctx.__aexit__(None, None, None)
        import os
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


if __name__ == "__main__":
    unittest.main()
