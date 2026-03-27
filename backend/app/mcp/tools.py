"""ToolRef MCP tool implementations.

Each function here corresponds to one MCP tool registered in *server.py*.
The RAG capability is provided by calling the existing FastAPI endpoint
(``POST /api/v1/query``) so that we fully reuse the LangGraph pipeline
defined in ``app.retrieval.graph`` without duplicating infrastructure
setup (Postgres / Redis / Milvus connections).

The target endpoint URL is read from the ``TOOLREF_API_URL`` environment
variable (default: ``http://localhost:8000``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

#: Base URL of the ToolRef backend API.  Override via ``TOOLREF_API_URL``.
_API_URL: str = os.environ.get("TOOLREF_API_URL", "http://localhost:8000").rstrip("/")

#: HTTP timeouts (seconds).  RAG pipelines can be slow on first inference.
_CONNECT_TIMEOUT: float = 10.0
_READ_TIMEOUT: float = 120.0

_TIMEOUT = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=10.0, pool=5.0)


# ── Tool implementations ──────────────────────────────────────────────────────


async def toolref_query(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute an Agentic RAG query against the ToolRef knowledge base.

    Delegates to the backend ``POST /api/v1/query`` endpoint which runs
    the full LangGraph pipeline (analyse → retrieve → rerank → grade →
    rewrite → generate).

    Args:
        query:     Natural-language question to answer.
        namespace: Document namespace to search within (default: ``"default"``).
        top_k:     Maximum number of source chunks to retrieve (1–20).

    Returns:
        A dict with the following keys:

        * ``answer``     — Generated answer string.
        * ``sources``    — List of source dicts ``{doc_title, chunk_text, url, score}``.
        * ``confidence`` — Heuristic confidence score in ``[0.0, 1.0]``.

    Raises:
        RuntimeError: When the backend returns a non-2xx response or is unreachable.
    """
    # Clamp top_k to the API's accepted range
    top_k = max(1, min(top_k, 20))

    payload: dict[str, Any] = {
        "query": query,
        "namespace": namespace,
        "top_k": top_k,
        "use_cache": True,
    }

    logger.debug(
        "toolref_query: query=%r namespace=%r top_k=%d → %s",
        query[:80],
        namespace,
        top_k,
        _API_URL,
    )

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                f"{_API_URL}/api/v1/query",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        msg = f"Cannot connect to ToolRef backend at {_API_URL}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except httpx.TimeoutException as exc:
        msg = f"ToolRef backend request timed out after {_READ_TIMEOUT}s: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except httpx.HTTPStatusError as exc:
        msg = (
            f"ToolRef backend returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc

    data: dict[str, Any] = response.json()

    answer: str = data.get("answer", "")
    sources: list[dict[str, Any]] = data.get("sources", [])
    rewrite_count: int = data.get("rewrite_count", 0)
    cached: bool = data.get("cached", False)

    # ── Heuristic confidence score ─────────────────────────────────────────
    # Derive a simple [0, 1] score from available signals:
    #   • More rewrites needed → lower confidence
    #   • Cache hit → slightly higher confidence (previously validated)
    #   • Average relevance score from sources (if present)
    if sources:
        avg_score: float = sum(
            float(s.get("score", 0.5)) for s in sources
        ) / len(sources)
    else:
        avg_score = 0.0

    rewrite_penalty: float = rewrite_count * 0.15
    cache_bonus: float = 0.05 if cached else 0.0
    confidence: float = max(0.0, min(1.0, avg_score - rewrite_penalty + cache_bonus))

    logger.info(
        "toolref_query completed: answer_len=%d sources=%d confidence=%.2f cached=%s",
        len(answer),
        len(sources),
        confidence,
        cached,
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 3),
    }
