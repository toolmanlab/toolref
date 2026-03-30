"""Unit tests for app.ingestion.embedder.EmbeddingService.

The BGE-M3 model (~570 MB) is never actually loaded during these tests.
All calls to _load_model are intercepted with unittest.mock so the suite
runs offline and quickly in the Docker container.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ingestion.embedder import EmbeddingService


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _make_mock_model(dense_dim: int = 1024) -> MagicMock:
    """Return a MagicMock that mimics BGEM3FlagModel.encode output.

    Each call to encode(batch, ...) returns a dict with:
      - dense_vecs: np.ndarray of shape (len(batch), dense_dim)
      - lexical_weights: list of dicts, one per text
    """
    mock_model = MagicMock()

    def _encode(batch, **kwargs):  # noqa: ANN001
        n = len(batch)
        return {
            "dense_vecs": np.random.rand(n, dense_dim).astype(np.float32),
            "lexical_weights": [{"42": 0.5, "99": 0.3} for _ in range(n)],
        }

    mock_model.encode.side_effect = _encode
    return mock_model


# ═════════════════════════════════════════════════════════════════════════════
# Interface / return-type tests
# ═════════════════════════════════════════════════════════════════════════════


def test_embed_texts_returns_tuple_of_ndarray_and_list() -> None:
    """embed_texts must return (np.ndarray, list[dict]) regardless of input size."""
    service = EmbeddingService(batch_size=8)
    service._model = _make_mock_model()

    texts = ["hello world", "foo bar baz"]
    dense, sparse = service.embed_texts(texts)

    assert isinstance(dense, np.ndarray), "dense output must be np.ndarray"
    assert isinstance(sparse, list), "sparse output must be list"
    assert len(sparse) == len(texts), "sparse list length must equal number of input texts"


def test_embed_texts_dense_shape() -> None:
    """Dense embeddings have shape (N, embedding_dim)."""
    dim = 1024
    service = EmbeddingService(batch_size=8)
    service._model = _make_mock_model(dense_dim=dim)

    texts = ["a", "b", "c"]
    dense, _ = service.embed_texts(texts)

    assert dense.shape == (3, dim), f"Expected (3, {dim}), got {dense.shape}"


def test_embed_texts_sparse_entries_are_dicts() -> None:
    """Each sparse embedding entry is a dict mapping token-id strings to weights."""
    service = EmbeddingService(batch_size=8)
    service._model = _make_mock_model()

    _, sparse = service.embed_texts(["test sentence"])

    assert len(sparse) == 1
    entry = sparse[0]
    assert isinstance(entry, dict)
    for key, val in entry.items():
        assert isinstance(key, str), "token id keys must be strings"
        assert isinstance(val, float), "token weight values must be floats"


# ═════════════════════════════════════════════════════════════════════════════
# warmup
# ═════════════════════════════════════════════════════════════════════════════


def test_warmup_calls_load_model_when_model_is_none() -> None:
    """warmup() triggers _load_model when the model has not been loaded yet."""
    service = EmbeddingService()
    assert service._model is None

    with patch.object(service, "_load_model") as mock_load:
        service.warmup()
        mock_load.assert_called_once()


def test_warmup_skips_load_model_when_already_loaded() -> None:
    """warmup() is a no-op when the model is already loaded."""
    service = EmbeddingService()
    service._model = _make_mock_model()

    with patch.object(service, "_load_model") as mock_load:
        service.warmup()
        mock_load.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# Batch splitting logic
# ═════════════════════════════════════════════════════════════════════════════


def test_batch_splitting_5_texts_batch_size_2() -> None:
    """5 texts with batch_size=2 produce exactly 3 encode() calls (2+2+1)."""
    service = EmbeddingService(batch_size=2)
    mock_model = _make_mock_model()
    service._model = mock_model

    texts = ["t1", "t2", "t3", "t4", "t5"]
    service.embed_texts(texts)

    assert mock_model.encode.call_count == 3, (
        f"Expected 3 batch calls for 5 texts @ batch_size=2, "
        f"got {mock_model.encode.call_count}"
    )


def test_batch_splitting_call_sizes() -> None:
    """Verify the actual batch sizes passed to encode: [2, 2, 1] for 5 texts."""
    service = EmbeddingService(batch_size=2)
    mock_model = _make_mock_model()
    service._model = mock_model

    service.embed_texts(["t1", "t2", "t3", "t4", "t5"])

    call_sizes = [len(call.args[0]) for call in mock_model.encode.call_args_list]
    assert call_sizes == [2, 2, 1], f"Unexpected batch sizes: {call_sizes}"


def test_batch_splitting_exact_multiple() -> None:
    """4 texts with batch_size=2 produce exactly 2 encode() calls (2+2)."""
    service = EmbeddingService(batch_size=2)
    mock_model = _make_mock_model()
    service._model = mock_model

    service.embed_texts(["a", "b", "c", "d"])

    assert mock_model.encode.call_count == 2


# ═════════════════════════════════════════════════════════════════════════════
# Empty input edge case
# ═════════════════════════════════════════════════════════════════════════════


def test_empty_input_returns_empty_array_and_list() -> None:
    """embed_texts([]) returns an empty ndarray and an empty list without errors."""
    service = EmbeddingService(batch_size=8)
    service._model = _make_mock_model()

    dense, sparse = service.embed_texts([])

    assert isinstance(dense, np.ndarray)
    assert dense.shape[0] == 0, f"Expected 0 rows, got {dense.shape[0]}"
    assert sparse == [], f"Expected empty list, got {sparse}"


def test_empty_input_does_not_call_encode() -> None:
    """embed_texts([]) must not call model.encode at all."""
    service = EmbeddingService(batch_size=8)
    mock_model = _make_mock_model()
    service._model = mock_model

    service.embed_texts([])

    mock_model.encode.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# Lazy loading — model is loaded on first embed_texts call
# ═════════════════════════════════════════════════════════════════════════════


def test_embed_texts_triggers_load_model_when_model_is_none() -> None:
    """embed_texts calls _load_model when _model is None (lazy-load path)."""
    service = EmbeddingService(batch_size=8)
    assert service._model is None

    def _fake_load():
        service._model = _make_mock_model()

    with patch.object(service, "_load_model", side_effect=_fake_load) as mock_load:
        service.embed_texts(["test"])
        mock_load.assert_called_once()
