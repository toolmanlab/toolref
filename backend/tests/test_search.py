"""Unit tests for app.retrieval.search.

Covers:
  - reciprocal_rank_fusion: pure RRF score computation, empty inputs, dedup merging
  - hybrid_search: sparse-embedding validation (ValueError on empty sparse list)

No live Milvus or embedding model is required — all external calls are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.retrieval.search import reciprocal_rank_fusion


# ═════════════════════════════════════════════════════════════════════════════
# reciprocal_rank_fusion — pure function, no mocking needed
# ═════════════════════════════════════════════════════════════════════════════


def _doc(chunk_id: str, score: float = 1.0) -> dict:
    """Construct a minimal retrieval result dict."""
    return {"chunk_id": chunk_id, "doc_id": "d1", "score": score}


def test_rrf_basic_score_calculation() -> None:
    """RRF score = Σ 1/(k + rank + 1). Verify the formula for a simple case.

    With k=60 and a doc at dense rank 0 and sparse rank 0:
      score = 1/(60+0+1) + 1/(60+0+1) = 2/61 ≈ 0.03279
    """
    dense = [_doc("A"), _doc("B")]
    sparse = [_doc("A"), _doc("C")]

    result = reciprocal_rank_fusion(dense, sparse, k=60)

    # Doc A appears in both lists at rank 0 → highest score
    result_by_id = {r["chunk_id"]: r["rrf_score"] for r in result}

    expected_a = 1 / (60 + 0 + 1) + 1 / (60 + 0 + 1)
    assert result_by_id["A"] == pytest.approx(expected_a, rel=1e-6)


def test_rrf_result_is_sorted_descending() -> None:
    """Fused results are ordered by RRF score, highest first."""
    dense = [_doc("A"), _doc("B"), _doc("C")]
    sparse = [_doc("C"), _doc("B"), _doc("A")]

    result = reciprocal_rank_fusion(dense, sparse, k=60)
    scores = [r["rrf_score"] for r in result]

    assert scores == sorted(scores, reverse=True), "Results must be sorted descending by rrf_score"


def test_rrf_empty_sparse_results() -> None:
    """When sparse_results is empty, only dense ranks contribute to scores."""
    dense = [_doc("X"), _doc("Y")]

    result = reciprocal_rank_fusion(dense, sparse_results=[], k=60)

    assert len(result) == 2
    ids = [r["chunk_id"] for r in result]
    assert "X" in ids and "Y" in ids

    # Doc X at dense rank 0 → score = 1/61; Doc Y at rank 1 → 1/62
    scores_by_id = {r["chunk_id"]: r["rrf_score"] for r in result}
    assert scores_by_id["X"] == pytest.approx(1 / 61, rel=1e-6)
    assert scores_by_id["Y"] == pytest.approx(1 / 62, rel=1e-6)


def test_rrf_empty_dense_results() -> None:
    """When dense_results is empty, only sparse ranks contribute."""
    sparse = [_doc("P"), _doc("Q")]

    result = reciprocal_rank_fusion(dense_results=[], sparse_results=sparse, k=60)

    assert len(result) == 2
    scores_by_id = {r["chunk_id"]: r["rrf_score"] for r in result}
    assert scores_by_id["P"] == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_dedup_merging() -> None:
    """Docs appearing in both lists are deduplicated; score is the sum of both contributions."""
    # Doc "shared" at dense rank 0 and sparse rank 0
    dense = [_doc("shared"), _doc("only_dense")]
    sparse = [_doc("shared"), _doc("only_sparse")]

    result = reciprocal_rank_fusion(dense, sparse, k=60)
    ids = [r["chunk_id"] for r in result]

    assert len(result) == 3, "Three unique chunk_ids expected"
    assert ids.count("shared") == 1, "Shared doc must appear exactly once"


def test_rrf_both_empty() -> None:
    """Empty dense and sparse inputs return an empty list."""
    result = reciprocal_rank_fusion([], [], k=60)
    assert result == []


def test_rrf_augments_rrf_score_field() -> None:
    """Every returned document must have an 'rrf_score' key."""
    dense = [_doc("A"), _doc("B")]
    sparse = [_doc("B"), _doc("C")]

    for doc in reciprocal_rank_fusion(dense, sparse):
        assert "rrf_score" in doc, f"Missing rrf_score in {doc}"


# ═════════════════════════════════════════════════════════════════════════════
# hybrid_search — integration points that require mocking
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_hybrid_search_raises_on_empty_sparse_embeddings() -> None:
    """hybrid_search must raise ValueError when embed_texts returns an empty sparse list.

    BGE-M3 should always produce sparse weights; an empty list signals a model
    loading failure and must be caught before hitting Milvus.
    """
    # embed_texts returns a valid dense array but an empty sparse list
    dense_fake = np.array([[0.1] * 1024], dtype=np.float32)
    empty_sparse: list = []

    with patch(
        "app.retrieval.search.embedding_service.embed_texts",
        return_value=(dense_fake, empty_sparse),
    ):
        from app.retrieval.search import hybrid_search

        with pytest.raises(ValueError, match="empty sparse embeddings"):
            await hybrid_search("some query", namespace="test_ns")


@pytest.mark.asyncio
async def test_hybrid_search_calls_milvus_with_correct_types() -> None:
    """hybrid_search converts sparse str-key dict to int-key dict before Milvus call."""
    dense_fake = np.array([[0.2] * 1024], dtype=np.float32)
    # BGE-M3 emits string token ids: {"42": 0.5}
    sparse_fake = [{"42": 0.5, "100": 0.3}]

    captured_sparse: dict = {}

    def _fake_milvus_search(dense_emb, sparse_emb, namespace, top_k):  # noqa: ANN001
        captured_sparse.update(sparse_emb)
        return []

    with patch("app.retrieval.search.embedding_service.embed_texts", return_value=(dense_fake, sparse_fake)), \
         patch("app.retrieval.search._milvus_search_single_sync", side_effect=_fake_milvus_search):

        from app.retrieval.search import hybrid_search
        await hybrid_search("test query", namespace="ns")

    # All keys must have been converted from str → int
    for key in captured_sparse:
        assert isinstance(key, int), f"Expected int key, got {type(key)}: {key}"


@pytest.mark.asyncio
async def test_hybrid_search_returns_list() -> None:
    """hybrid_search returns a list of result dicts (empty list on Milvus error)."""
    dense_fake = np.array([[0.1] * 1024], dtype=np.float32)
    sparse_fake = [{"1": 0.9}]
    milvus_results = [
        {"chunk_id": "c1", "doc_id": "d1", "parent_chunk_id": None,
         "namespace": "ns", "score": 0.9, "rrf_score": 0.015},
    ]

    with patch("app.retrieval.search.embedding_service.embed_texts", return_value=(dense_fake, sparse_fake)), \
         patch("app.retrieval.search._milvus_search_single_sync", return_value=milvus_results):

        from app.retrieval.search import hybrid_search
        results = await hybrid_search("query", namespace="ns", top_k=5)

    assert isinstance(results, list)
