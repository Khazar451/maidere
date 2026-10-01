---
title: "[Bug]: Thread-Scoped Subagent Progress Callbacks to Prevent Concurrency Race Conditions"
labels: ["bug", "concurrency", "backend"]
---

### Problem Statement
In `core/subagent.py`, the callback used to stream real-time progress events over WebSocket (`_progress_callback`) is currently stored as a single global variable:

```python
# core/subagent.py: line 51
_progress_callback: Callable[[dict], Coroutine] | None = None

def set_progress_callback(callback: Callable[[dict], Coroutine] | None) -> None:
    global _progress_callback
    _progress_callback = callback
```

When two different browser sessions or background tasks run subagent delegations concurrently (e.g. User A triggers deep research on Thread 1 while User B triggers plan generation on Thread 2), the second invocation overwrites `_progress_callback`. Consequently, subagent progress updates are either sent to the wrong client or dropped completely.

### Steps to Reproduce
1. Open Maidere in two separate browser windows (Window 1 on Thread A, Window 2 on Thread B).
2. Concurrently execute `/plan` or `[deep-research]` in both windows.
3. Observe that only the most recently triggered window receives live subagent step updates in its terminal and banner.

### Proposed Solution
1. **Thread-Keyed Registry**:
   Convert `_progress_callback` to a thread-keyed dictionary:
   ```python
   _progress_callbacks: dict[str, Callable[[dict], Coroutine]] = {}

   def register_progress_callback(thread_id: str, callback: Callable[[dict], Coroutine]) -> None:
       _progress_callbacks[thread_id] = callback

   def unregister_progress_callback(thread_id: str) -> None:
       _progress_callbacks.pop(thread_id, None)
   ```

2. **Scoped Dispatch**:
   Pass `thread_id` to `run_subagent(..., thread_id=thread_id)` and dispatch progress events using `_progress_callbacks.get(thread_id)`.

3. **Lifecycle Cleanup**:
   Ensure `api/routes.py` registers the callback before invoking LangGraph and unregisters it inside a `finally:` block.

### Acceptance Criteria
- [ ] Concurrent subagent delegations stream progress exclusively to their respective thread WebSockets.
- [ ] Unit test added in `tests/test_subagent.py` asserting concurrent executions maintain isolated callback state.
