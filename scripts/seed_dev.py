#!/usr/bin/env python3
"""seed_dev.py — Insert sample documents into the dev environment.

Usage:
    python scripts/seed_dev.py [--base-url http://localhost:8000]

Requires the backend to be running (docker compose up -d).
"""
from __future__ import annotations

import argparse
import io
import sys
import textwrap
import urllib.request
import json


SAMPLE_DOCS: list[dict] = [
    {
        "namespace": "demo",
        "title": "ToolRef Architecture Overview",
        "filename": "architecture.md",
        "content": textwrap.dedent(
            """\
            # ToolRef Architecture Overview

            ToolRef is a production-grade RAG engine providing domain expertise
            via both a Chat UI and an MCP interface.

            ## Components
            - **FastAPI** — REST API gateway
            - **LangGraph** — Agentic RAG state machine
            - **Milvus** — Dense + sparse vector storage
            - **Redis** — Semantic cache (cosine sim ≥ 0.92)
            - **PostgreSQL** — Document metadata + query history

            ## Retrieval Flow
            1. analyze_query → route
            2. hybrid_retrieve (dense + sparse + RRF)
            3. cross-encoder rerank
            4. LLM-as-judge grading
            5. generate (with citations)
            """
        ),
    },
    {
        "namespace": "demo",
        "title": "Quick Start Guide",
        "filename": "quickstart.md",
        "content": textwrap.dedent(
            """\
            # Quick Start

            ## Prerequisites
            - Docker + Docker Compose
            - Node.js 20+
            - Python 3.12+

            ## Steps
            1. `cp .env.example .env`
            2. `docker compose up -d`
            3. `make migrate`
            4. `cd frontend && npm install && npm run dev`
            5. Visit http://localhost:5173

            ## First Query
            Upload a document via the API or Chat UI, then ask a question.
            The RAG engine will retrieve, rerank, and generate an answer
            with citations.
            """
        ),
    },
]


def upload_document(base_url: str, doc: dict) -> None:
    """POST a markdown document to the upload endpoint."""
    url = f"{base_url}/api/v1/documents"
    content = doc["content"].encode("utf-8")
    boundary = "----ToolRefBoundary"
    body_parts: list[bytes] = []

    # file field
    body_parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{doc['filename']}\"\r\nContent-Type: text/markdown\r\n\r\n".encode()
    )
    body_parts.append(content)
    body_parts.append(b"\r\n")

    # namespace field
    body_parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"namespace\"\r\n\r\n"
        f"{doc['namespace']}\r\n".encode()
    )

    # title field
    body_parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\n"
        f"{doc['title']}\r\n".encode()
    )

    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            print(f"  ✓  {doc['title']} → id={result.get('id', '?')}")
    except urllib.error.HTTPError as exc:
        body_resp = exc.read().decode()
        print(f"  ✗  {doc['title']} → HTTP {exc.code}: {body_resp[:120]}")
    except Exception as exc:
        print(f"  ✗  {doc['title']} → {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ToolRef with sample documents.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()

    print(f"🌱  Seeding ToolRef at {args.base_url} ...")
    for doc in SAMPLE_DOCS:
        upload_document(args.base_url, doc)
    print("Done.")


if __name__ == "__main__":
    main()
