---
title: "[Enhancement]: Multi-Turn Conversational Visual Memory & Image Retention"
labels: ["enhancement", "vision", "multimodal"]
---

### Problem Statement
In the current implementation of multimodal vision, `images` are attached to the LangGraph state for the active turn only:

```python
# core/agent.py: line 931
if i == last_user_idx and turn_images:
    user_msg_dict["images"] = turn_images
```

When a conversation progresses to turn 2 and the user asks a follow-up question regarding an uploaded UI mockup or screenshot (e.g. *"What changes should I make to the primary button in that design?"*), turn 2 has no new images attached, so `turn_images` is empty. The model is forced to hallucinate or rely strictly on text summaries without the ability to inspect the visual artifact.

### Proposed Solution
1. **LRU Visual Memory Buffer**:
   - Maintain a sliding window of the last $N$ turns of image references (default: $N=2$) associated with the thread checkpointer.
   - For queries containing visual deixis ("that image", "the screenshot", "the diagram", "in the UI mockup"), include recent image payloads in the prompt context.
2. **Context & Token Pruning Guardrails**:
   - Beyond 2 turns, prune raw base64 data to keep KV-cache footprint compact and prevent VRAM exhaustion.
   - Downscale images to max 1920x1080 resolution before storing in checkpoint memory.

### Alternatives Considered
- Persisting all images permanently: exhausts context window and GPU memory rapidly on local 8GB RTX workstations.
- Storing images on disk in `workspace/.artifacts/images/` and passing file paths: excellent complementary storage model.

### Acceptance Criteria
- [ ] Users can ask follow-up questions about uploaded screenshots in the subsequent turn without re-uploading.
- [ ] Older images are automatically pruned to prevent context overflow.
- [ ] Automated tests in `tests/test_multimodal.py` verify 2-turn visual memory continuity.
