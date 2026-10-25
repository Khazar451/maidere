"""Short-term visual memory: keep recent uploaded images available for follow-ups.

Images arrive with a single user turn. Without this module the model only sees
them on that turn, so "what should I change on the button in that mockup?" on
the next turn has nothing to look at.

Policy:
- Images are remembered per user message (``message_id``) in ``image_history``,
  which lives in the LangGraph checkpoint next to the conversation.
- An image stays available for ``image_memory_turns`` user turns (the turn it was
  uploaded on counts as the first) and at most ``image_memory_max_images`` images
  are kept overall; anything older is dropped from the checkpoint.
- Stored images are downscaled to fit within 1920x1080.
- Remembered images are only re-attached when the new query refers back to a
  visual ("that screenshot", "the diagram", ...), so unrelated follow-ups do not
  pay the vision-model cost.
"""

import asyncio
import base64
import hashlib
import io
import re
from typing import Any

import structlog

from maidere.core.config import settings

logger = structlog.get_logger()

MAX_IMAGE_WIDTH = 1920
MAX_IMAGE_HEIGHT = 1080

_VISUAL_NOUNS = (
    r"image|images|picture|pictures|photo|photos|screenshot|screenshots|screen\s?cap|diagram|diagrams|"
    r"mock-?up|mock-?ups|wireframe|design|ui|chart|graph|figure|plot|logo|icon|layout|illustration|"
    r"drawing|sketch|scan|pic|pics|attachment|upload(?:ed)?"
)
# "that image", "the screenshot", "in the UI mockup", "the attached picture", "above image" ...
_DEIXIS = re.compile(
    rf"\b(?:(?:that|this|those|these|the|my|your|previous|earlier|last|above|attached|uploaded|same|first|second)\s+"
    rf"(?:\w+\s+){{0,2}}(?:{_VISUAL_NOUNS})|(?:{_VISUAL_NOUNS})\s+(?:above|from before|from earlier|i (?:sent|uploaded|attached)))\b",
    re.IGNORECASE,
)


def references_prior_image(query: str) -> bool:
    """True when the query points back at an image from an earlier turn."""
    if not query:
        return False
    return bool(_DEIXIS.search(query))


def _split_data_uri(value: str) -> tuple[str, str]:
    if value.startswith("data:") and "," in value:
        prefix, _, payload = value.partition(",")
        return prefix + ",", payload
    return "", value


def _downscale_sync(value: str) -> str:
    prefix, payload = _split_data_uri(value.strip())
    try:
        from PIL import Image

        raw = base64.b64decode(payload, validate=False)
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            if width <= MAX_IMAGE_WIDTH and height <= MAX_IMAGE_HEIGHT:
                return value
            fmt = (img.format or "PNG").upper()
            resized = img.copy()
            resized.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
            if fmt not in ("PNG", "JPEG", "WEBP"):
                fmt = "PNG"
            if fmt == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            buf = io.BytesIO()
            save_args: dict[str, Any] = {"quality": 85} if fmt in ("JPEG", "WEBP") else {}
            resized.save(buf, format=fmt, **save_args)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        if prefix:
            mime = "image/" + fmt.lower()
            return f"data:{mime};base64,{encoded}"
        return encoded
    except Exception as e:  # unreadable/unsupported image: keep it as uploaded
        logger.warning("image_downscale_skipped", error=str(e))
        return value


async def downscale_image(value: str) -> str:
    """Fit an image (raw base64 or data URI) within 1920x1080; never raises."""
    return await asyncio.to_thread(_downscale_sync, value)


def _digest(value: str) -> str:
    return hashlib.sha256(_split_data_uri(value.strip())[1].encode("ascii", "ignore")).hexdigest()


def _human_ids(messages: list[Any]) -> list[str]:
    from langchain_core.messages import HumanMessage

    return [str(m.id) for m in messages if isinstance(m, HumanMessage) and getattr(m, "id", None)]


async def remember_images(
    history: list[dict] | None,
    messages: list[Any],
    turn_images: list[str] | None,
) -> list[dict]:
    """Return the updated ``image_history`` for this turn.

    Records the current turn's images against the latest user message, then drops
    entries that fell out of the window or exceed the image cap. Idempotent: an
    image already stored (same bytes) is never added twice, which also protects
    against a caller that re-sends a stale ``images`` value.
    """
    turns = settings.image_memory_turns
    if turns <= 0:
        return []

    entries = [dict(e) for e in (history or []) if e.get("images")]
    human_ids = _human_ids(messages)
    current_id = human_ids[-1] if human_ids else None

    if turn_images and current_id:
        known = {_digest(img) for e in entries for img in e["images"]}
        fresh: list[str] = []
        for img in turn_images:
            if img and _digest(img) not in known:
                known.add(_digest(img))
                fresh.append(await downscale_image(img))
        if fresh:
            existing = next((e for e in entries if e["message_id"] == current_id), None)
            if existing:
                existing["images"] = [*existing["images"], *fresh]
            else:
                entries.append({"message_id": current_id, "images": fresh})

    in_window = set(human_ids[-turns:])
    entries = [e for e in entries if e["message_id"] in in_window]

    cap = max(0, settings.image_memory_max_images)
    kept: list[dict] = []
    budget = cap
    for entry in reversed(entries):  # newest first
        if budget <= 0:
            break
        images = entry["images"][-budget:]
        budget -= len(images)
        kept.append({"message_id": entry["message_id"], "images": images})
    kept.reverse()

    logger.debug("visual_memory_updated", images=sum(len(e["images"]) for e in kept), turns=turns)
    return kept


def recall_images(history: list[dict] | None, current_message_id: str | None, query: str) -> dict[str, list[str]]:
    """Images from earlier turns to re-attach, keyed by the user message they came with.

    Empty unless the query refers back to a visual. The current message's own
    images are excluded; they are attached through the normal per-turn path.
    """
    if not history or not references_prior_image(query):
        return {}
    return {
        str(e["message_id"]): list(e["images"])
        for e in history
        if e.get("images") and str(e["message_id"]) != str(current_message_id)
    }
