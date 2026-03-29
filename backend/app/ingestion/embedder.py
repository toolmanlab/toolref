# NOTE: Temporary lightweight model configuration.
# Using sentence-transformers/all-MiniLM-L6-v2 (~90 MB, 384-d) instead of
# BGE-M3 (2.3 GB, 1024-d) to unblock local development.
# To revert: restore the FlagEmbedding import + BGEM3FlagModel in _load_model,
# flip embed_texts back to returning real lexical weights, and set env vars
# EMBEDDING_MODEL=BAAI/bge-m3 and EMBEDDING_DIM=1024.

"""Embedding service backed by sentence-transformers (lightweight mode).

Generates **dense** (384-d, L2-normalised) embeddings only.
Sparse embeddings are not available with this model; ``embed_texts``
always returns an empty list for the sparse component so that callers
can detect the absence and substitute dummy values where required
(e.g. Milvus SPARSE_FLOAT_VECTOR insertion via the ingestion pipeline).

The BGE-M3 interface is intentionally preserved:
  ``embed_texts(texts) -> tuple[ndarray, list[dict[str, float]]]``
so switching back is a one-file change.

The model is loaded lazily on first call to :meth:`embed_texts` and
reused for the lifetime of the process (singleton pattern).
"""

import logging
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate dense embeddings using sentence-transformers.

    Sparse embeddings are not produced; callers receive an empty list
    and must substitute dummy values when writing to Milvus.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Maximum texts per inference batch.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None  # Lazy-loaded SentenceTransformer

    # ── Lazy model loading ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the sentence-transformers model into memory."""
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        logger.info("Loading embedding model '%s' …", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")

    def warmup(self) -> None:
        """Pre-load the model so the first real call is fast.

        Intended to be called at worker startup.
        """
        if self._model is None:
            self._load_model()

    # ── Public API ───────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Generate dense embeddings for a list of texts.

        Sparse embeddings are *not* available with this lightweight model.
        The second element of the returned tuple is always an empty list
        ``[]``.  Callers that must write to Milvus should substitute a
        dummy sparse vector ``{0: 0.0001}`` per chunk; callers performing
        search should skip the sparse ANN request entirely.

        Args:
            texts: Input strings to embed.

        Returns:
            A tuple ``(dense_embeddings, sparse_embeddings)`` where:

            * ``dense_embeddings`` is a numpy array of shape ``(N, 384)``
              with L2-normalised rows.
            * ``sparse_embeddings`` is always ``[]``.
        """
        if self._model is None:
            self._load_model()

        all_dense: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs: np.ndarray = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_dense.append(vecs)

        dense_embeddings = (
            np.concatenate(all_dense, axis=0)
            if all_dense
            else np.empty((0, settings.embedding_dim))
        )
        logger.info(
            "Generated embeddings for %d texts (dense shape=%s)",
            len(texts),
            dense_embeddings.shape,
        )
        # Sparse is not produced by this lightweight model.
        # Return empty list so downstream code can detect and handle absence.
        return dense_embeddings, []


# Module-level singleton — shared by the ingestion pipeline.
embedding_service = EmbeddingService()
