"""Async Ollama client via httpx.

Direct HTTP calls to Ollama's REST API. No LiteLLM dependency.
Always enforces num_ctx=8192 to prevent VRAM overflow on RTX 5060 8GB.
"""

import time
from typing import Any, Callable, Awaitable
import httpx
import structlog

from core.config import settings
from core.metrics import LLM_DURATION_SECONDS, LLM_REQUESTS_TOTAL, LLM_TOKENS_TOTAL

logger = structlog.get_logger()


async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    on_token: Any = None,
) -> dict:
    """Send a chat request to Ollama. Returns the full response dict.

    If on_token callback is provided, streams chunks from Ollama in real-time
    and invokes await on_token(delta) token-by-token while assembling the full response.

    Response includes:
    - message.content: the text response
    - message.tool_calls: list of tool calls (if tools provided and model decides to use them)
    """
    import json

    target_model = model or settings.ollama_model
    target_num_ctx = num_ctx or settings.ollama_num_ctx
    payload: dict = {
        "model": target_model,
        "messages": messages,
        "options": {"num_ctx": target_num_ctx},
        "stream": on_token is not None,
    }
    if tools:
        payload["tools"] = tools

    start_time = time.perf_counter()
    timeout_cfg = httpx.Timeout(settings.ollama_timeout, connect=settings.ollama_connect_timeout)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        await logger.ainfo(
            "ollama_request",
            model=target_model,
            message_count=len(messages),
            has_tools=tools is not None,
            streaming=on_token is not None,
        )
        try:
            if on_token is not None:
                content_chunks = []
                tool_calls = []
                final_data = {}
                async with client.stream("POST", f"{settings.ollama_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        token = msg.get("content", "")
                        if token:
                            content_chunks.append(token)
                            try:
                                await on_token(token)
                            except Exception:
                                pass
                        if msg.get("tool_calls"):
                            tool_calls.extend(msg["tool_calls"])
                        if chunk.get("done"):
                            final_data = chunk
                            duration = time.perf_counter() - start_time
                            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
                            LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()
                            prompt_eval_count = chunk.get("prompt_eval_count", 0)
                            eval_count = chunk.get("eval_count", 0)
                            if prompt_eval_count:
                                LLM_TOKENS_TOTAL.labels(model=target_model, token_type="prompt").inc(prompt_eval_count)
                            if eval_count:
                                LLM_TOKENS_TOTAL.labels(model=target_model, token_type="completion").inc(eval_count)

                full_content = "".join(content_chunks)
                result_dict = {
                    "model": final_data.get("model", target_model),
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "done": True,
                    "prompt_eval_count": final_data.get("prompt_eval_count", 0),
                    "eval_count": final_data.get("eval_count", 0),
                }
                if tool_calls:
                    result_dict["message"]["tool_calls"] = tool_calls
                return result_dict

            response = await client.post(
                f"{settings.ollama_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()

            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)
            if prompt_eval_count:
                LLM_TOKENS_TOTAL.labels(model=target_model, token_type="prompt").inc(prompt_eval_count)
            if eval_count:
                LLM_TOKENS_TOTAL.labels(model=target_model, token_type="completion").inc(eval_count)

            await logger.ainfo(
                "ollama_response",
                model=data.get("model"),
                eval_count=eval_count,
                eval_duration_ns=data.get("eval_duration"),
                duration_s=round(duration, 3),
            )
            return data
        except Exception as e:
            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="error").inc()
            raise


async def chat_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
):
    """Stream chat response chunks from Ollama.

    Yields individual chunk dicts as they arrive.
    The final chunk typically contains 'done': True along with eval metrics.
    """
    import json

    target_model = model or settings.ollama_model
    target_num_ctx = num_ctx or settings.ollama_num_ctx
    payload: dict = {
        "model": target_model,
        "messages": messages,
        "options": {"num_ctx": target_num_ctx},
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    start_time = time.perf_counter()
    timeout_cfg = httpx.Timeout(settings.ollama_timeout, connect=settings.ollama_connect_timeout)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        await logger.ainfo(
            "ollama_stream_request",
            model=target_model,
            message_count=len(messages),
            has_tools=tools is not None,
        )
        try:
            async with client.stream("POST", f"{settings.ollama_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        duration = time.perf_counter() - start_time
                        LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
                        LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()
                        prompt_eval = chunk.get("prompt_eval_count", 0)
                        eval_c = chunk.get("eval_count", 0)
                        if prompt_eval:
                            LLM_TOKENS_TOTAL.labels(model=target_model, token_type="prompt").inc(prompt_eval)
                        if eval_c:
                            LLM_TOKENS_TOTAL.labels(model=target_model, token_type="completion").inc(eval_c)
                    yield chunk
        except Exception as e:
            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="error").inc()
            raise




async def is_available() -> bool:
    """Check if Ollama is running and the configured model is pulled."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]
            return any(settings.ollama_model in m for m in models)
    except (httpx.HTTPError, Exception):
        return False
