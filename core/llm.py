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


def is_cloud_model(model: str | None) -> bool:
    """Determine if a requested model should be routed to an OpenAI-compatible Cloud API."""
    if not model:
        return False
    m = model.strip().lower()
    if m.startswith(("nvidia/", "meta/", "openai/", "mistralai/", "deepseek-ai/", "google/")):
        return True
    configured = (getattr(settings, "cloud_model", "") or "").strip().lower()
    if configured and m == configured:
        return True
    return False


def _get_cloud_request_params(target_model: str) -> tuple[str, dict[str, str]]:
    """Resolve cloud completions endpoint URL and authentication headers."""
    api_key, base_url, _ = settings.get_effective_cloud_config()
    if not api_key:
        raise ValueError(
            f"Cloud model '{target_model}' requested, but no NVIDIA_API_KEY or CLOUD_API_KEY "
            "is configured in your environment or .env file."
        )

    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return url, headers


def _normalize_messages_for_cloud(messages: list[dict]) -> list[dict]:
    """Ensure message list adheres strictly to OpenAI / NVIDIA NIM tool calling and message role specifications.

    1. Assistant tool calls must include:
       - 'id' (str)
       - 'type': 'function'
       - 'function': {'name': ..., 'arguments': <JSON str>}
    2. Tool responses ('role': 'tool') must include:
       - 'tool_call_id' matching the preceding assistant tool call.
    3. Handles stringification of dictionary arguments.
    """
    import json
    import uuid

    normalized: list[dict] = []
    pending_tool_call_ids: list[str] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            norm_msg: dict = {"role": "assistant", "content": content or ""}
            raw_tool_calls = msg.get("tool_calls")
            if raw_tool_calls:
                norm_tool_calls = []
                for idx, tc in enumerate(raw_tool_calls):
                    call_id = tc.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}"
                    pending_tool_call_ids.append(str(call_id))
                    func_obj = tc.get("function") or {}
                    name = func_obj.get("name") or tc.get("name", "")
                    raw_args = func_obj.get("arguments")
                    if raw_args is None:
                        raw_args = tc.get("args", {})
                    args_str = json.dumps(raw_args) if isinstance(raw_args, (dict, list)) else str(raw_args or "{}")

                    norm_tool_calls.append({
                        "id": str(call_id),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    })
                norm_msg["tool_calls"] = norm_tool_calls
            normalized.append(norm_msg)

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id and pending_tool_call_ids:
                tool_call_id = pending_tool_call_ids.pop(0)
            elif not tool_call_id:
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            normalized.append({
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(content or ""),
            })

        elif role == "user":
            images = msg.get("images")
            if images:
                parts: list[dict] = [{"type": "text", "text": str(content or "")}]
                for img in images:
                    img_str = str(img).strip()
                    if not img_str.startswith("data:"):
                        img_str = f"data:image/jpeg;base64,{img_str}"
                    parts.append({"type": "image_url", "image_url": {"url": img_str}})
                normalized.append({"role": "user", "content": parts})
            else:
                normalized.append({"role": "user", "content": str(content or "")})

        else:
            normalized.append(dict(msg))

    return normalized


async def _chat_cloud(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    on_token: Any = None,
) -> dict:
    """Send a chat request to an OpenAI-compatible Cloud AI provider (e.g. NVIDIA NIM).

    Supports:
    - Real-time token streaming via on_token callback
    - Native extraction of reasoning_content (NVIDIA Nemotron / DeepSeek) wrapped in <think> tags
    - OpenAI-compatible tool calling
    """
    import json

    target_model = model or settings.cloud_model
    url, headers = _get_cloud_request_params(target_model)

    is_reasoning_model = any(k in target_model.lower() for k in ("nemotron", "deepseek", "r1", "thinking"))
    norm_messages = _normalize_messages_for_cloud(messages)

    payload: dict = {
        "model": target_model,
        "messages": norm_messages,
        "temperature": 1.0 if "nemotron" in target_model.lower() else 0.7,
        "top_p": 0.95 if "nemotron" in target_model.lower() else 1.0,
        "max_tokens": 16384,
        "stream": on_token is not None,
    }
    if is_reasoning_model:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if tools:
        payload["tools"] = tools

    start_time = time.perf_counter()
    timeout_cfg = httpx.Timeout(settings.cloud_timeout, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        await logger.ainfo(
            "cloud_llm_request",
            model=target_model,
            url=url,
            message_count=len(messages),
            has_tools=tools is not None,
            streaming=on_token is not None,
        )
        try:
            if on_token is not None:
                content_chunks: list[str] = []
                reasoning_chunks: list[str] = []
                streamed_tool_calls: list[dict] = []
                in_thinking = False

                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        if not raw_line or not raw_line.strip():
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # 1. Reasoning tokens (e.g. Nemotron-3, DeepSeek)
                        reasoning_token = delta.get("reasoning_content")
                        if reasoning_token:
                            reasoning_chunks.append(reasoning_token)
                            if not in_thinking:
                                in_thinking = True
                                await on_token("<think>\n")
                            try:
                                await on_token(reasoning_token)
                            except Exception:
                                pass

                        # 2. Main content tokens
                        content_token = delta.get("content")
                        if content_token:
                            if in_thinking:
                                in_thinking = False
                                await on_token("\n</think>\n")
                            content_chunks.append(content_token)
                            try:
                                await on_token(content_token)
                            except Exception:
                                pass

                        # 3. Tool calls delta
                        tc_delta = delta.get("tool_calls")
                        if tc_delta:
                            for tc in tc_delta:
                                idx = tc.get("index", len(streamed_tool_calls))
                                while len(streamed_tool_calls) <= idx:
                                    streamed_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                if tc.get("id"):
                                    streamed_tool_calls[idx]["id"] = tc["id"]
                                func = tc.get("function", {})
                                if func.get("name"):
                                    streamed_tool_calls[idx]["function"]["name"] += func["name"]
                                if func.get("arguments"):
                                    streamed_tool_calls[idx]["function"]["arguments"] += func["arguments"]

                if in_thinking:
                    try:
                        await on_token("\n</think>\n")
                    except Exception:
                        pass

                full_content = ""
                if reasoning_chunks:
                    full_content += f"<think>\n{''.join(reasoning_chunks)}\n</think>\n"
                full_content += "".join(content_chunks)

                duration = time.perf_counter() - start_time
                LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
                LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()

                result_dict = {
                    "model": target_model,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "done": True,
                }
                if streamed_tool_calls:
                    result_dict["message"]["tool_calls"] = streamed_tool_calls
                return result_dict

            # Non-streaming request
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()

            choices = data.get("choices", [])
            msg = choices[0].get("message", {}) if choices else {}
            reasoning = msg.get("reasoning_content")
            if reasoning and "<think>" not in (msg.get("content") or ""):
                msg["content"] = f"<think>\n{reasoning}\n</think>\n" + (msg.get("content") or "")

            return {
                "model": data.get("model", target_model),
                "message": msg,
                "done": True,
            }
        except Exception:
            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="error").inc()
            raise


async def _chat_cloud_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
):
    """Stream chat response chunks from Cloud AI provider, yielding in Ollama chunk format."""
    import json

    target_model = model or settings.cloud_model
    url, headers = _get_cloud_request_params(target_model)
    is_reasoning_model = any(k in target_model.lower() for k in ("nemotron", "deepseek", "r1", "thinking"))
    norm_messages = _normalize_messages_for_cloud(messages)

    payload: dict = {
        "model": target_model,
        "messages": norm_messages,
        "temperature": 1.0 if "nemotron" in target_model.lower() else 0.7,
        "top_p": 0.95 if "nemotron" in target_model.lower() else 1.0,
        "max_tokens": 16384,
        "stream": True,
    }
    if is_reasoning_model:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if tools:
        payload["tools"] = tools

    start_time = time.perf_counter()
    timeout_cfg = httpx.Timeout(settings.cloud_timeout, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line or not raw_line.strip():
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    yield {
                        "model": target_model,
                        "message": {"role": "assistant", "content": token},
                        "done": False,
                    }

            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="success").inc()
            yield {"model": target_model, "message": {"role": "assistant", "content": ""}, "done": True}
        except Exception:
            duration = time.perf_counter() - start_time
            LLM_DURATION_SECONDS.labels(model=target_model).observe(duration)
            LLM_REQUESTS_TOTAL.labels(model=target_model, status="error").inc()
            raise


async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    on_token: Any = None,
) -> dict:
    """Send a chat request to Ollama or Cloud AI provider. Returns the full response dict.

    If on_token callback is provided, streams chunks in real-time
    and invokes await on_token(delta) token-by-token while assembling the full response.
    """
    target_model = model or settings.ollama_model
    if is_cloud_model(target_model):
        return await _chat_cloud(messages=messages, tools=tools, model=target_model, on_token=on_token)

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
    """Stream chat response chunks from Ollama or Cloud AI provider.

    Yields individual chunk dicts as they arrive.
    The final chunk typically contains 'done': True along with eval metrics.
    """
    target_model = model or settings.ollama_model
    if is_cloud_model(target_model):
        async for chunk in _chat_cloud_stream(messages=messages, tools=tools, model=target_model):
            yield chunk
        return

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
    """Check if LLM backend is available (local Ollama or Cloud AI provider)."""
    api_key, _, _ = settings.get_effective_cloud_config()
    if is_cloud_model(settings.ollama_model) and api_key:
        return True
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]
            return any(settings.ollama_model in m for m in models)
    except (httpx.HTTPError, Exception):
        return bool(api_key)

