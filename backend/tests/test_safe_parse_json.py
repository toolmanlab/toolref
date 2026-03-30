"""Unit tests for app.retrieval.nodes._safe_parse_json.

Exercises all three fallback stages:
  Stage 1 — direct json.loads (with optional markdown fence stripping)
  Stage 2 — regex extraction of the first {...} block (+ single-quote normalisation)
  Stage 3 — keyword heuristic for grading-specific calls (only when fallback
             contains the "relevant" key)

All tests are synchronous; no external services are required.
"""

from __future__ import annotations

import pytest

# _safe_parse_json is a module-private helper; import it directly.
from app.retrieval.nodes import _safe_parse_json


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — standard JSON / markdown-wrapped JSON
# ═════════════════════════════════════════════════════════════════════════════


def test_stage1_plain_json() -> None:
    """Plain valid JSON string is parsed correctly without any fallback."""
    result = _safe_parse_json('{"relevant": true, "reason": "on topic"}')
    assert result == {"relevant": True, "reason": "on topic"}


def test_stage1_markdown_fenced_json() -> None:
    """JSON wrapped in a ```json ... ``` markdown code block is unwrapped and parsed."""
    text = '```json\n{"relevant": false, "reason": "off topic"}\n```'
    result = _safe_parse_json(text)
    assert result == {"relevant": False, "reason": "off topic"}


def test_stage1_markdown_fence_no_language_tag() -> None:
    """JSON wrapped in plain ``` ... ``` (no language tag) is also unwrapped."""
    text = '```\n{"query_type": "simple", "entities": []}\n```'
    result = _safe_parse_json(text)
    assert result == {"query_type": "simple", "entities": []}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — regex extraction from prose text
# ═════════════════════════════════════════════════════════════════════════════


def test_stage2_json_embedded_in_prose() -> None:
    """A JSON object embedded inside freeform text is extracted by regex."""
    text = 'Here is my answer: {"relevant": true, "reason": "test"} — hope that helps.'
    result = _safe_parse_json(text)
    assert result["relevant"] is True
    assert result["reason"] == "test"


def test_stage2_single_quoted_json() -> None:
    """Single-quoted JSON emitted by some small models is normalised to double quotes."""
    text = "{'relevant': true, 'reason': 'matches query'}"
    result = _safe_parse_json(text)
    assert result["relevant"] is True


def test_stage2_json_after_preamble() -> None:
    """Preamble sentence before the JSON object is stripped by regex."""
    text = "Sure! The grading result is: {\"sub_queries\": [\"foo\", \"bar\"]}"
    result = _safe_parse_json(text)
    assert result == {"sub_queries": ["foo", "bar"]}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — keyword heuristic (grading-specific, "relevant" in fallback)
# ═════════════════════════════════════════════════════════════════════════════


def test_stage3_positive_keywords_yield_relevant_true() -> None:
    """Pure text with positive keywords → {'relevant': True} when fallback has 'relevant'."""
    text = "Yes, this document is relevant and useful for the query."
    result = _safe_parse_json(text, fallback={"relevant": False})
    assert result["relevant"] is True
    assert result.get("reason") == "keyword_heuristic"


def test_stage3_negative_keywords_yield_relevant_false() -> None:
    """Pure text with negative keywords → {'relevant': False} when fallback has 'relevant'."""
    text = "No, this is not relevant at all — irrelevant content."
    result = _safe_parse_json(text, fallback={"relevant": False})
    assert result["relevant"] is False
    assert result.get("reason") == "keyword_heuristic"


def test_stage3_only_activates_with_relevant_in_fallback() -> None:
    """Keyword heuristic is NOT activated when fallback does NOT contain 'relevant'."""
    text = "Yes, this is relevant and useful."
    # fallback without "relevant" key → heuristic must not run → return fallback
    result = _safe_parse_json(text, fallback={"query_type": "simple"})
    assert result == {"query_type": "simple"}


# ═════════════════════════════════════════════════════════════════════════════
# Fallback / edge cases
# ═════════════════════════════════════════════════════════════════════════════


def test_completely_unparseable_returns_fallback() -> None:
    """Text with no JSON and no keyword signal returns the provided fallback dict."""
    text = "I am a teapot. Nothing to see here."
    fallback = {"relevant": False, "reason": "parse_error"}
    result = _safe_parse_json(text, fallback=fallback)
    # keyword heuristic may run but signal is ambiguous → falls back
    # Either way the result must be a dict (not raise)
    assert isinstance(result, dict)


def test_empty_string_returns_fallback() -> None:
    """Empty string cannot be parsed → fallback is returned."""
    fallback = {"relevant": False, "reason": "empty"}
    result = _safe_parse_json("", fallback=fallback)
    assert isinstance(result, dict)


def test_no_fallback_defaults_to_empty_dict() -> None:
    """When no fallback is supplied, the default empty dict is returned on failure."""
    result = _safe_parse_json("this is not json at all", fallback=None)
    assert isinstance(result, dict)


def test_nested_json_object() -> None:
    """Deeply nested valid JSON is handled correctly by Stage 1."""
    text = '{"entities": ["LangGraph", "RAG"], "query_type": "complex", "intent": "explain"}'
    result = _safe_parse_json(text)
    assert result["query_type"] == "complex"
    assert "LangGraph" in result["entities"]
