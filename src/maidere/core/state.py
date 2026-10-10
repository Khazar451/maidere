"""LangGraph state schema for Maidere."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Maidere agent state for LangGraph.

    messages: Conversation history managed by add_messages reducer.
    memory_context: Retrieved memories from sqlite-vec, injected each turn.
    thread_id: Unique conversation thread identifier.
    model: Model name to use for this execution.
    has_delegated: Whether delegate_task was used this query (blocks re-delegation).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    memory_context: str
    thread_id: str
    model: str
    num_ctx: int
    tool_loop_count: int
    has_delegated: bool
    subagent_results: list[dict]
    thinking_mode: bool
    deep_reasoning: bool
    critique: str
    refinement_count: int
    reasoning_steps: list[str]
    complexity: Any
    username: str
    images: list[str]
    attachments: list[dict]
