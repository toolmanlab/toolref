"""Unit tests for eval/metrics.py — IR metric computation.

Tests cover compute_ir_metrics and aggregate_ir_metrics with a range of
hit / miss / edge-case scenarios, verifying all four metrics
(hit rate, MRR, precision@k, recall@k).
"""

from __future__ import annotations

import pytest

from eval.metrics import IRMetrics, aggregate_ir_metrics, compute_ir_metrics


# ═════════════════════════════════════════════════════════════════════════════
# compute_ir_metrics
# ═════════════════════════════════════════════════════════════════════════════


def test_perfect_hit_first_rank() -> None:
    """Expected doc appears at rank 1 → hit=True, rr=1.0, recall=1.0."""
    result = compute_ir_metrics(
        retrieved_doc_titles=["Doc A", "Doc B"],
        expected_doc_titles=["Doc A"],
    )
    assert result.hit is True
    assert result.reciprocal_rank == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)


def test_hit_at_second_rank() -> None:
    """Expected doc at rank 2 → rr=0.5."""
    result = compute_ir_metrics(
        retrieved_doc_titles=["Doc B", "Doc A"],
        expected_doc_titles=["Doc A"],
    )
    assert result.hit is True
    assert result.reciprocal_rank == pytest.approx(0.5)


def test_complete_miss() -> None:
    """None of the expected docs are in the results → hit=False, rr=0.0."""
    result = compute_ir_metrics(
        retrieved_doc_titles=["Doc B", "Doc C"],
        expected_doc_titles=["Doc A"],
    )
    assert result.hit is False
    assert result.reciprocal_rank == pytest.approx(0.0)
    assert result.recall_at_k == pytest.approx(0.0)


def test_partial_recall_multi_expected() -> None:
    """Two expected docs, only one retrieved → recall = 0.5."""
    result = compute_ir_metrics(
        retrieved_doc_titles=["A", "C"],
        expected_doc_titles=["A", "B"],
    )
    assert result.hit is True
    assert result.recall_at_k == pytest.approx(0.5)


def test_out_of_scope_empty_expected() -> None:
    """Empty expected list (out-of-scope query) → all metrics are 1.0 / True.

    This is the special-case logic that marks out-of-scope queries as
    correctly handled.
    """
    result = compute_ir_metrics(
        retrieved_doc_titles=["X"],
        expected_doc_titles=[],
    )
    assert result.hit is True
    assert result.reciprocal_rank == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)


def test_k_cutoff_respected() -> None:
    """Doc appearing beyond k should NOT count as a hit."""
    # k=2, expected doc is at position 3 (0-indexed=2)
    result = compute_ir_metrics(
        retrieved_doc_titles=["X", "Y", "Doc A"],
        expected_doc_titles=["Doc A"],
        k=2,
    )
    assert result.hit is False
    assert result.reciprocal_rank == pytest.approx(0.0)


def test_precision_at_k_calculation() -> None:
    """Precision@k = relevant_in_k / k, independently of recall."""
    # k=5, 2 relevant in top-5 → precision = 2/5
    result = compute_ir_metrics(
        retrieved_doc_titles=["A", "X", "B", "Y", "Z"],
        expected_doc_titles=["A", "B", "C"],
        k=5,
    )
    assert result.precision_at_k == pytest.approx(2 / 5)


def test_all_expected_docs_found() -> None:
    """All expected docs in top-k → recall=1.0 and hit=True."""
    result = compute_ir_metrics(
        retrieved_doc_titles=["A", "B", "C"],
        expected_doc_titles=["A", "B"],
        k=5,
    )
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.hit is True


# ═════════════════════════════════════════════════════════════════════════════
# aggregate_ir_metrics
# ═════════════════════════════════════════════════════════════════════════════


def test_aggregate_mean_values() -> None:
    """aggregate_ir_metrics averages metrics across multiple results."""
    r1 = IRMetrics(hit=True, reciprocal_rank=1.0, precision_at_k=1.0, recall_at_k=1.0)
    r2 = IRMetrics(hit=False, reciprocal_rank=0.0, precision_at_k=0.0, recall_at_k=0.0)

    agg = aggregate_ir_metrics([r1, r2])

    assert agg["hit_rate"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.5)
    assert agg["precision_at_k"] == pytest.approx(0.5)
    assert agg["recall_at_k"] == pytest.approx(0.5)


def test_aggregate_empty_list() -> None:
    """aggregate_ir_metrics on empty list returns all zeros (no division-by-zero)."""
    agg = aggregate_ir_metrics([])

    assert agg["hit_rate"] == pytest.approx(0.0)
    assert agg["mrr"] == pytest.approx(0.0)
    assert agg["precision_at_k"] == pytest.approx(0.0)
    assert agg["recall_at_k"] == pytest.approx(0.0)


def test_aggregate_single_result() -> None:
    """aggregate_ir_metrics with a single result equals that result's values."""
    r = IRMetrics(hit=True, reciprocal_rank=0.5, precision_at_k=0.2, recall_at_k=0.75)
    agg = aggregate_ir_metrics([r])

    assert agg["hit_rate"] == pytest.approx(1.0)
    assert agg["mrr"] == pytest.approx(0.5)
    assert agg["precision_at_k"] == pytest.approx(0.2)
    assert agg["recall_at_k"] == pytest.approx(0.75)


def test_aggregate_three_queries() -> None:
    """MRR calculation verified manually across three queries."""
    # MRR = (1/1 + 1/2 + 1/3) / 3 ≈ 0.6111
    r1 = IRMetrics(hit=True, reciprocal_rank=1.0, precision_at_k=0.2, recall_at_k=1.0)
    r2 = IRMetrics(hit=True, reciprocal_rank=0.5, precision_at_k=0.2, recall_at_k=1.0)
    r3 = IRMetrics(hit=True, reciprocal_rank=1 / 3, precision_at_k=0.2, recall_at_k=1.0)

    agg = aggregate_ir_metrics([r1, r2, r3])

    expected_mrr = (1.0 + 0.5 + 1 / 3) / 3
    assert agg["mrr"] == pytest.approx(expected_mrr, rel=1e-5)
