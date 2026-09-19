---
name: deep-research
description: Autonomous multi-step technical research, query decomposition, and structured Obsidian note synthesis using sub-agents.
triggers:
  - research
  - deep research
  - investigate
  - obsidian
  - fellowship
  - internship
---

# Deep Research Directives

You are equipped with the `delegate_task` tool to run sub-agents with dedicated fresh context windows.

CRITICAL INSTRUCTIONS:
1. Do NOT write narrative plans or procedural text (NEVER output phrases like "Phase 1", "According to the protocol", "Let's delegate", or "I will research").
2. Your very first action MUST be emitting `delegate_task` tool calls for 2-3 specific sub-tasks tailored to the user's topic.
3. Wait for the sub-agents to complete. Synthesize all verified data into a comprehensive report with comparison tables, metrics, and citations.

