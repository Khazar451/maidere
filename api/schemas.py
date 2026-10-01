"""Pydantic models for Maidere API requests and responses."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message."""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="The user's message"
    )
    thread_id: str | None = Field(
        None,
        description="Conversation thread ID. Auto-generated if not provided.",
    )
    model: str | None = Field(
        None,
        description="Optional Ollama model name (e.g. qwen2.5:7b-instruct). Defaults to configured model.",
    )
    num_ctx: int | None = Field(
        None,
        description="Optional context window size in tokens (e.g. 8192, 16384, 32768, 65536, 131072).",
    )
    thinking_mode: bool | None = Field(
        False,
        description="Optional flag to enable deep thinking/reasoning mode.",
    )
    deep_reasoning: bool | None = Field(
        False,
        description="Optional flag to enable human-style deep System 2 reasoning and deliberation.",
    )
    username: str | None = Field(
        None,
        max_length=64,
        description="The user's nickname or chosen username.",
    )
    images: list[str] | None = Field(
        None,
        description="Optional list of base64-encoded images for multimodal vision reasoning.",
    )
    attachments: list[dict] | None = Field(
        None,
        description="Optional list of attachment dicts ({name, type, data, size}) for documents, PDFs, code, or images.",
    )


class UserLoginRequest(BaseModel):
    """User nickname login request without passwords."""

    username: str = Field(
        ..., min_length=1, max_length=64, description="User nickname"
    )


class UserProfileResponse(BaseModel):
    """Active user profile and known nicknames."""

    username: str = Field(..., description="Active user nickname")
    known_users: list[str] = Field(
        default_factory=list, description="Previously known nicknames"
    )



class ToolExecutionItem(BaseModel):
    """Details of a tool execution in a turn."""

    tool_name: str = Field(..., description="Name of the executed tool")
    args: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    result: str = Field(..., description="Output produced by the tool")
    duration_ms: int = Field(0, description="Execution duration in milliseconds")
    success: bool = Field(True, description="Whether execution was successful")


class ChatResponse(BaseModel):
    """Agent response."""

    response: str = Field(..., description="The agent's response")
    thread_id: str = Field(
        ..., description="Conversation thread ID for continuity"
    )
    trimmed: bool = Field(
        False, description="Whether messages were trimmed in this turn"
    )
    tools_executed: list[ToolExecutionItem] = Field(
        default_factory=list, description="List of tools executed during this turn"
    )
    memory_context: str | None = Field(
        None, description="Memory context retrieved for this turn"
    )



class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    ollama_available: bool = Field(
        ..., description="Whether Ollama is reachable and model is loaded"
    )
    model: str = Field(..., description="Configured LLM model")
    db_path: str = Field(..., description="Database file path")


class MessageItem(BaseModel):
    """A single chat message in thread history."""

    role: str = Field(..., description="Message sender role: user, assistant, system, tool")
    content: str = Field(..., description="Content of the message")
    timestamp: str | None = Field(None, description="ISO timestamp if available")


class HistoryResponse(BaseModel):
    """Conversation history for a thread."""

    thread_id: str = Field(..., description="Conversation thread identifier")
    messages: list[MessageItem] = Field(default_factory=list, description="Messages in chronological order")
    count: int = Field(..., description="Total message count")


class ThreadItem(BaseModel):
    """Conversation thread metadata."""

    thread_id: str = Field(..., description="Thread identifier")
    title: str = Field("New Conversation", description="Thread title or snippet")
    summary: str | None = Field(None, description="AI-generated conversation summary")
    message_count: int = Field(0, description="Number of messages in thread")
    last_updated: str | None = Field(None, description="Last activity timestamp")



class ThreadsListResponse(BaseModel):
    """List of past conversation threads."""

    threads: list[ThreadItem] = Field(default_factory=list, description="List of threads")


class MemoryItem(BaseModel):
    """Stored semantic memory."""


    content: str = Field(..., description="Memory text content")
    timestamp: str = Field(..., description="Timestamp when memory was recorded")
    distance: float | None = Field(None, description="Distance metric if recalled via query")


class MemoryRecallResponse(BaseModel):
    """List of recalled semantic memories."""

    memories: list[MemoryItem] = Field(default_factory=list, description="Top matching memories")

