"""CPU-based text embeddings via fastembed (ONNX runtime).

Uses bge-small-en-v1.5: 384 dimensions, ~50MB model, ~5-10ms per query.
Zero VRAM usage — keeps Qwen pinned in GPU memory 24/7.
"""

import os
import structlog
from fastembed import TextEmbedding

from core.config import settings

logger = structlog.get_logger()

_model = None


class FallbackEmbedding:
    """Deterministic fallback embedding generator when offline or model is unavailable."""

    _STOP_WORDS = {
        "what", "which", "is", "are", "the", "and", "does", "in", "a", "an",
        "to", "for", "of", "with", "this", "that", "it", "on", "at", "by"
    }

    def embed(self, texts: list[str]):
        import zlib
        import numpy as np
        for t in texts:
            vec = np.zeros(384, dtype=np.float32)
            words = [w.strip('.,?!\'\"') for w in t.lower().split()]
            for w in words:
                if len(w) > 1 and w not in self._STOP_WORDS:
                    idx = zlib.crc32(w.encode("utf-8")) % 384
                    vec[idx] += 1.0
                    if len(w) > 3:
                        prefix_idx = zlib.crc32(w[:4].encode("utf-8")) % 384
                        vec[prefix_idx] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            yield vec


def _get_model():
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        # Try local cache first
        try:
            _model = TextEmbedding(model_name=settings.embedding_model, local_files_only=True)
            logger.info("embedding_model_loaded", model=settings.embedding_model)
            return _model
        except Exception:
            pass

        if os.environ.get("HF_HUB_OFFLINE") in ("1", "TRUE", "true", "yes", "ON"):
            logger.info("embedding_model_offline_fallback", model=settings.embedding_model)
            _model = FallbackEmbedding()
            return _model

        try:
            _model = TextEmbedding(model_name=settings.embedding_model)
            logger.info("embedding_model_loaded", model=settings.embedding_model)
        except Exception as e:
            logger.warning("embedding_model_load_failed", error=str(e))
            _model = FallbackEmbedding()
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts on CPU. Returns list of 384-dim float vectors."""
    model = _get_model()
    return [e.tolist() for e in model.embed(texts)]


def embed_one(text: str) -> list[float]:
    """Embed a single text. Returns a 384-dim float vector."""
    return embed([text])[0]
