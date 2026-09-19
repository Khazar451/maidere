"""FastAPI routes for Maidere API."""

import asyncio
from pathlib import Path
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    MemoryItem,
    MemoryRecallResponse,
    MessageItem,
    ThreadItem,
    ThreadsListResponse,
    ToolExecutionItem,
)

from core import llm
from core.agent import (
    get_system_prompt,
    register_stream_callback,
    set_custom_system_prompt,
    unregister_stream_callback,
)
from core.config import settings
from core.db import get_db, get_db_context
from core.memory import delete_memory, list_all_memories, recall, store_memory

logger = structlog.get_logger()

router = APIRouter()

# The compiled graph is injected at startup via app lifespan
_graph = None

# Active generation tasks for cancellation tracking
_active_generation_tasks: dict[str, asyncio.Task] = {}


def set_graph(graph) -> None:
    """Set the compiled LangGraph instance. Called from app lifespan."""
    global _graph
    _graph = graph



SUMMARIZATION_INTERVAL = 20


async def update_thread_metadata_and_title(
    thread_id: str,
    messages: list[Any],
) -> str:
    """Intelligently assign/update AI-generated thread title and rolling summary."""
    human_msgs = [m for m in messages if isinstance(m, HumanMessage) and getattr(m, "content", None)]
    if not human_msgs:
        return "New Conversation"

    first_user_text = str(human_msgs[0].content).strip()
    total_count = len([m for m in messages if isinstance(m, (HumanMessage, AIMessage))])

    try:
        async with get_db_context(settings.db_path) as db:
            # Check existing meta
            row = await db.execute_fetchall(
                "SELECT title, summary FROM threads_meta WHERE thread_id = ?",
                [thread_id],
            )
            existing_title = row[0][0] if row and row[0] else None
            existing_summary = row[0][1] if row and row[0] else None

            title = existing_title
            summary = existing_summary

            # Generate AI title if missing or default
            if not title or title in ("New Conversation", "New Thread", "Conversation"):
                try:
                    title_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "You are a specialized assistant that creates concise, descriptive 3 to 5 word titles "
                                "for conversations. Respond ONLY with the title. Never use quotation marks, periods, or conversational preamble."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Title for a conversation starting with: '{first_user_text}'",
                        },
                    ]
                    resp = await llm.chat(title_prompt, tools=None)
                    raw_title = resp.get("message", {}).get("content", "").strip()
                    clean_title = raw_title.strip('"\'`').replace("\n", " ").strip()
                    if clean_title and len(clean_title) <= 50:
                        title = clean_title
                    else:
                        title = (first_user_text[:35] + "...") if len(first_user_text) > 35 else first_user_text
                except Exception:
                    title = (first_user_text[:35] + "...") if len(first_user_text) > 35 else first_user_text

            # Update rolling summary every 4 turns
            if total_count >= 2 and (not summary or total_count % 4 == 0):
                try:
                    transcript = "\n".join(
                        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                        for m in messages[-4:]
                        if isinstance(m, (HumanMessage, AIMessage))
                    )
                    sum_prompt = [
                        {
                            "role": "system",
                            "content": "Summarize the essence of this dialogue in 1 clear sentence.",
                        },
                        {"role": "user", "content": transcript},
                    ]
                    sum_resp = await llm.chat(sum_prompt, tools=None)
                    raw_sum = sum_resp.get("message", {}).get("content", "").strip()
                    if raw_sum:
                        summary = raw_sum
                except Exception:
                    pass

            await db.execute(
                """
                INSERT INTO threads_meta (thread_id, title, summary, message_count, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(thread_id) DO UPDATE SET
                    title = excluded.title,
                    summary = COALESCE(excluded.summary, threads_meta.summary),
                    message_count = excluded.message_count,
                    updated_at = datetime('now')
                """,
                [thread_id, title, summary, total_count],
            )
            await db.commit()
            return title
    except Exception as e:
        await logger.aerror("update_thread_metadata_failed", thread_id=thread_id, error=str(e))
        return (first_user_text[:35] + "...") if len(first_user_text) > 35 else first_user_text



async def maybe_summarize_conversation(
    thread_id: str,
    messages: list[Any],
) -> None:
    """Summarize conversation when it reaches every 20 turns and store to memories."""
    human_and_ai_msgs = [
        m for m in messages if isinstance(m, (HumanMessage, AIMessage))
    ]
    count = len(human_and_ai_msgs)
    if count >= SUMMARIZATION_INTERVAL and count % SUMMARIZATION_INTERVAL == 0:
        try:
            transcript = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in human_and_ai_msgs[-SUMMARIZATION_INTERVAL:]
            )
            summary_prompt = [
                {
                    "role": "system",
                    "content": "You are a concise summarizer. Summarize the following dialogue in 1-2 sentences focusing on key facts, user preferences, and important decisions made.",
                },
                {"role": "user", "content": transcript},
            ]
            response = await llm.chat(summary_prompt)
            summary_text = response.get("message", {}).get("content", "").strip()
            if summary_text:
                async with get_db_context(settings.db_path) as db:
                    await store_memory(
                        db,
                        content=f"Summary of conversation turn {count}: {summary_text}",
                        session_id=thread_id,
                    )
                    await logger.ainfo(
                        "conversation_summarized",
                        thread_id=thread_id,
                        turn_count=count,
                    )
        except Exception as e:
            await logger.aerror("summarization_failed", error=str(e))




@router.get("/models")
@router.get("/api/models")
async def list_available_models() -> dict[str, list[str]]:
    """Fetch installed local models directly from Ollama, with auto routing."""
    from core.router import get_available_models

    try:
        models = await get_available_models(force_refresh=True)
        unique_models = []
        for m in models:
            if m not in unique_models and m != "auto":
                unique_models.append(m)
        return {"models": ["auto"] + unique_models}
    except Exception:
        thinking_model = getattr(settings, "ollama_thinking_model", "deepseek-r1:7b")
        return {"models": ["auto", settings.ollama_model, settings.ollama_fast_model, thinking_model, "deepseek-r1:8b"]}



def should_store_memory(text: str) -> bool:
    """Filter out short greetings, casual chat, and low-information queries from permanent vector storage."""
    cleaned = text.strip().lower()
    if len(cleaned) < 12:
        return False
    ignore_prefixes = (
        "hello", "hi ", "hey", "how are you", "who are you", "what can you do",
        "why dont you answer", "well why dont you", "test", "testing"
    )
    if any(cleaned.startswith(p) for p in ignore_prefixes) and len(cleaned) < 35:
        return False
    return True


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to Maidere and get a response."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    thread_id = request.thread_id or str(uuid.uuid4())
    model_name = request.model or settings.ollama_model

    await logger.ainfo(
        "chat_request",
        thread_id=thread_id,
        model=model_name,
        message_length=len(request.message),
    )

    # 1. Ingest and embed substantive user messages into vector memories
    if should_store_memory(request.message):
        try:
            async with get_db_context(settings.db_path) as db:
                await store_memory(db, content=request.message, session_id=thread_id)
        except Exception as e:
            await logger.awarn("memory_ingest_skipped", error=str(e))


    config = {"configurable": {"thread_id": thread_id}}

    # Reset sub-agent spawn counter for this new query
    from tools.registry import get_tool
    delegate_tool = get_tool("delegate_task")
    if delegate_tool and hasattr(delegate_tool, "reset_spawn_count"):
        delegate_tool.reset_spawn_count()

    # Capture message count before invocation to isolate new turn tools
    prior_msg_count = 0
    try:
        state_before = await _graph.aget_state(config)
        if state_before and state_before.values:
            prior_msg_count = len(state_before.values.get("messages", []))
    except Exception:
        pass

    try:
        result = await _graph.ainvoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "memory_context": "No relevant memories found.",
                "thread_id": thread_id,
                "model": model_name,
                "num_ctx": request.num_ctx or settings.ollama_num_ctx,
                "has_delegated": False,
                "tool_loop_count": 0,
                "subagent_results": [],
                "thinking_mode": bool(request.thinking_mode),
                "deep_reasoning": bool(request.deep_reasoning),
                "critique": "",
                "refinement_count": 0,
                "reasoning_steps": [],
            },
            config=config,
        )

    except Exception as e:
        await logger.aerror("chat_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable. Is Ollama running? Error: {type(e).__name__}",
        )

    # 2. Check for conversation summarization trigger & update AI thread title
    all_messages = result.get("messages", [])
    await maybe_summarize_conversation(thread_id, all_messages)
    await update_thread_metadata_and_title(thread_id, all_messages)

    new_messages = all_messages[prior_msg_count:]
    tools_executed = []
    for m in new_messages:
        if isinstance(m, ToolMessage):
            extra = getattr(m, "additional_kwargs", {}) or {}
            tools_executed.append(
                ToolExecutionItem(
                    tool_name=getattr(m, "name", "tool"),
                    args=extra.get("args", {}),
                    result=str(m.content),
                    duration_ms=extra.get("duration_ms", 0),
                    success=not str(m.content).startswith("Error:"),
                )
            )

    ai_message = all_messages[-1]
    response_text = (
        ai_message.content if hasattr(ai_message, "content") else str(ai_message)
    )


    await logger.ainfo(
        "chat_response",
        thread_id=thread_id,
        response_length=len(response_text),
        tools_count=len(tools_executed),
    )

    return ChatResponse(
        response=response_text,
        thread_id=thread_id,
        trimmed=False,
        tools_executed=tools_executed,
        memory_context=result.get("memory_context"),
    )



@router.get("/threads", response_model=ThreadsListResponse)
async def list_threads() -> ThreadsListResponse:
    """List past conversation threads with smart AI titles and summaries."""
    if _graph is None:
        return ThreadsListResponse(threads=[])

    try:
        async with get_db_context(settings.db_path) as db:
            # 1. Query indexed threads from threads_meta
            meta_rows = await db.execute_fetchall(
                "SELECT thread_id, title, summary, message_count, updated_at FROM threads_meta ORDER BY updated_at DESC"
            )
            meta_dict = {r[0]: r for r in meta_rows}

            # 2. Check checkpoints table for any legacy unindexed threads
            cp_rows = await db.execute_fetchall(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
            )
            all_tids = []
            seen = set()
            for r in meta_rows:
                all_tids.append(r[0])
                seen.add(r[0])
            for r in cp_rows:
                if r[0] and r[0] not in seen:
                    all_tids.append(r[0])
                    seen.add(r[0])

        thread_items = []
        for tid in all_tids:
            if tid in meta_dict:
                r = meta_dict[tid]
                thread_items.append(
                    ThreadItem(
                        thread_id=r[0],
                        title=r[1],
                        summary=r[2],
                        message_count=r[3],
                        last_updated=r[4],
                    )
                )
            else:
                try:
                    state = await _graph.aget_state({"configurable": {"thread_id": tid}})
                    msgs = state.values.get("messages", [])
                    title = "Conversation"
                    for m in msgs:
                        if isinstance(m, HumanMessage) and m.content:
                            content_str = str(m.content).strip()
                            title = (content_str[:35] + "...") if len(content_str) > 35 else content_str
                            break
                    thread_items.append(
                        ThreadItem(
                            thread_id=tid,
                            title=title,
                            message_count=len(msgs),
                        )
                    )
                except Exception:
                    thread_items.append(ThreadItem(thread_id=tid, title="Conversation"))

        return ThreadsListResponse(threads=thread_items)
    except Exception as e:
        await logger.aerror("list_threads_failed", error=str(e))
        return ThreadsListResponse(threads=[])


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Delete a conversation thread, checkpoints, and metadata."""
    try:
        async with get_db_context(settings.db_path) as db:
            await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", [thread_id])
            await db.execute("DELETE FROM writes WHERE thread_id = ?", [thread_id])
            await db.execute("DELETE FROM threads_meta WHERE thread_id = ?", [thread_id])
            await db.commit()
        return {"status": "deleted", "thread_id": thread_id}
    except Exception as e:
        await logger.aerror("delete_thread_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time chat streaming with token-level emissions and cancellation."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            # Handle explicit cancellation request
            if data.get("type") == "cancel":
                cancel_tid = data.get("thread_id")
                if cancel_tid and cancel_tid in _active_generation_tasks:
                    task = _active_generation_tasks[cancel_tid]
                    if not task.done():
                        task.cancel()
                        await websocket.send_json({"type": "cancelled", "thread_id": cancel_tid})
                continue

            message_text = data.get("message", "").strip()
            thread_id = data.get("thread_id") or str(uuid.uuid4())

            if not message_text:
                continue

            if _graph is None:
                await websocket.send_json({"type": "error", "message": "Agent not initialized."})
                continue

            # Ingest substantive memory with deduplication
            if should_store_memory(message_text):
                try:
                    async with get_db_context(settings.db_path) as db:
                        await store_memory(db, content=message_text, session_id=thread_id)
                except Exception:
                    pass

            # Create a thread-safe send lock for WebSocket
            ws_lock = asyncio.Lock()

            async def safe_ws_send(payload: dict) -> None:
                async with ws_lock:
                    try:
                        await websocket.send_json(payload)
                    except Exception as e:
                        await logger.adebug("ws_send_failed", error=str(e))

            # Notify UI that thinking started
            model_name = data.get("model") or settings.ollama_model
            num_ctx = data.get("num_ctx") or settings.ollama_num_ctx
            thinking_mode = bool(data.get("thinking_mode", False))
            deep_reasoning = bool(data.get("deep_reasoning", False))
            await safe_ws_send({"type": "stage", "stage": "thinking", "thread_id": thread_id})

            config = {"configurable": {"thread_id": thread_id}}

            # Reset sub-agent spawn counter for this new query
            from tools.registry import get_tool as _get_tool
            _delegate = _get_tool("delegate_task")
            if _delegate and hasattr(_delegate, "reset_spawn_count"):
                _delegate.reset_spawn_count()

            # Wire rich agent telemetry and sub-agent progress to WebSocket
            from core.agent import register_event_callback, unregister_event_callback
            from core.subagent import set_progress_callback, clear_progress_callback

            register_event_callback(thread_id, safe_ws_send)
            set_progress_callback(safe_ws_send)

            # Capture message count before invocation to isolate new turn tools
            prior_msg_count = 0
            try:
                state_before = await _graph.aget_state(config)
                if state_before and state_before.values:
                    prior_msg_count = len(state_before.values.get("messages", []))
            except Exception:
                pass

            async def token_stream_handler(token_text: str) -> None:
                await safe_ws_send({
                    "type": "token",
                    "text": token_text,
                    "thread_id": thread_id,
                })

            register_stream_callback(thread_id, token_stream_handler)
            generation_task = asyncio.create_task(
                _graph.ainvoke(
                    {
                        "messages": [HumanMessage(content=message_text)],
                        "memory_context": "No relevant memories found.",
                        "thread_id": thread_id,
                        "model": model_name,
                        "num_ctx": num_ctx,
                        "has_delegated": False,
                        "tool_loop_count": 0,
                        "subagent_results": [],
                        "thinking_mode": thinking_mode,
                        "deep_reasoning": deep_reasoning,
                        "critique": "",
                        "refinement_count": 0,
                        "reasoning_steps": [],
                    },
                    config=config,
                )
            )
            _active_generation_tasks[thread_id] = generation_task

            try:
                result = await generation_task
                all_messages = result.get("messages", [])
                await maybe_summarize_conversation(thread_id, all_messages)
                await update_thread_metadata_and_title(thread_id, all_messages)

                ai_message = all_messages[-1]
                response_text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)

                await safe_ws_send({
                    "type": "response",
                    "content": response_text,
                    "response": response_text,
                    "thread_id": thread_id,
                    "memory_context": result.get("memory_context", ""),
                })
                await safe_ws_send({"type": "done", "thread_id": thread_id})

            except asyncio.CancelledError:
                await logger.ainfo("generation_cancelled", thread_id=thread_id)
                await safe_ws_send({"type": "cancelled", "thread_id": thread_id})
            except Exception as e:
                await logger.aerror("ws_chat_failed", thread_id=thread_id, error=str(e))
                await safe_ws_send({"type": "error", "message": f"Error: {str(e)}"})
            finally:
                unregister_stream_callback(thread_id)
                unregister_event_callback(thread_id)
                clear_progress_callback()
                _active_generation_tasks.pop(thread_id, None)

    except WebSocketDisconnect:
        await logger.ainfo("ws_disconnected")


@router.post("/chat/cancel")
async def cancel_chat(payload: dict[str, str]) -> dict[str, Any]:
    """Cancel an ongoing generation task for a specific thread."""
    thread_id = payload.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="Missing thread_id")

    task = _active_generation_tasks.get(thread_id)
    if task and not task.done():
        task.cancel()
        return {"status": "cancelled", "thread_id": thread_id}
    return {"status": "not_running", "thread_id": thread_id}


@router.get("/history/{thread_id}", response_model=HistoryResponse)
@router.get("/threads/{thread_id}/history", response_model=HistoryResponse)
async def get_history(thread_id: str) -> HistoryResponse:
    """Retrieve full conversation history for a given thread."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    config = {"configurable": {"thread_id": thread_id}}
    state = await _graph.aget_state(config)

    messages = state.values.get("messages", [])
    items: list[MessageItem] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "unknown"

        content = msg.content if hasattr(msg, "content") else str(msg)
        items.append(MessageItem(role=role, content=content))

    return HistoryResponse(
        thread_id=thread_id,
        messages=items,
        count=len(items),
    )


@router.get("/threads/{thread_id}/export")
async def export_thread(thread_id: str, format: str = "md") -> Response:
    """Export conversation thread as Markdown or JSON."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    config = {"configurable": {"thread_id": thread_id}}
    state = await _graph.aget_state(config)
    messages = state.values.get("messages", [])

    if format.lower() == "json":
        data = [
            {"role": type(m).__name__, "content": getattr(m, "content", "")}
            for m in messages
        ]
        return JSONResponse(content={"thread_id": thread_id, "messages": data})

    # Default Markdown format
    lines = [f"# Maidere Conversation Export — Thread `{thread_id}`", ""]
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"### [User]\n\n{m.content}\n")
        elif isinstance(m, AIMessage):
            lines.append(f"### [Maidere]\n\n{m.content}\n")
        elif isinstance(m, ToolMessage):
            lines.append(f"> **Tool Execution ({getattr(m, 'name', 'tool')}):**\n```\n{m.content}\n```\n")

    return Response(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="maidere_thread_{thread_id[:8]}.md"'},
    )


@router.post("/threads/{thread_id}/fork")
async def fork_thread(thread_id: str, from_message: int | None = None) -> dict[str, Any]:
    """Fork an existing conversation into a new thread."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    new_thread_id = str(uuid.uuid4())
    config_old = {"configurable": {"thread_id": thread_id}}
    config_new = {"configurable": {"thread_id": new_thread_id}}

    state = await _graph.aget_state(config_old)
    messages = state.values.get("messages", [])
    if from_message is not None and from_message > 0:
        messages = messages[:from_message]

    if messages:
        await _graph.aupdate_state(config_new, {"messages": messages})
        await update_thread_metadata_and_title(new_thread_id, messages)

    return {"status": "forked", "new_thread_id": new_thread_id, "message_count": len(messages)}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a file to workspace/uploads/."""
    uploads_dir = Path(settings.agent_workspace) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    return {
        "status": "uploaded",
        "filename": file.filename,
        "path": f"uploads/{file.filename}",
        "size": len(content),
    }


@router.get("/memories")
async def get_memories(
    query: str | None = Query(None, description="Query text to search semantically"),
    top_k: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """List or search semantic memories."""
    async with get_db_context(settings.db_path) as db:
        if query and query.strip():
            results = await recall(db, query=query.strip(), top_k=top_k)
            return {"memories": results, "query": query}
        results = await list_all_memories(db, limit=top_k)
        return {"memories": results}


@router.delete("/memories/{memory_id}")
async def remove_memory(memory_id: int) -> dict[str, Any]:
    """Delete a memory entry by ID."""
    async with get_db_context(settings.db_path) as db:
        success = await delete_memory(db, memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "id": memory_id}


@router.get("/settings/system-prompt")
async def get_system_prompt_setting() -> dict[str, Any]:
    """Get the current system prompt."""
    return {"system_prompt": get_system_prompt()}


@router.put("/settings/system-prompt")
async def update_system_prompt_setting(payload: dict[str, Any]) -> dict[str, Any]:
    """Update custom system prompt."""
    prompt = payload.get("system_prompt")
    set_custom_system_prompt(prompt)
    return {"status": "updated", "system_prompt": get_system_prompt()}


@router.get("/skills")
async def list_skills_endpoint() -> dict[str, Any]:
    """List all available loaded skills and their metadata."""
    from core.skills import load_all_skills
    skills = load_all_skills()
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "triggers": s.triggers,
                "instructions": s.instructions,
                "path": str(s.path),
            }
            for s in skills.values()
        ]
    }



@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check service health and Ollama connectivity."""
    ollama_ok = await llm.is_available()
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_available=ollama_ok,
        model=settings.ollama_model,
        db_path=settings.db_path,
    )


@router.get("/obsidian/vault")
async def get_obsidian_vault() -> dict[str, Any]:
    """Get active Obsidian vault info."""
    from core.config import get_obsidian_vault_info
    vpath, vname = get_obsidian_vault_info()
    return {"vault_path": vpath, "vault_name": vname}


@router.post("/obsidian/open")
async def open_obsidian_note(payload: dict[str, str] = {}) -> dict[str, Any]:
    """Open a note or vault in the desktop Obsidian application."""
    from tools.obsidian import ObsidianTool
    tool = ObsidianTool()
    file_title = payload.get("file") or payload.get("title")
    res = await tool.execute(action="open_note", title=file_title)
    return {"status": "ok", "message": res}




