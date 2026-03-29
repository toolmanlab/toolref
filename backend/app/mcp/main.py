"""ToolRef MCP Server entry point.

Supports two transport modes:

* **stdio** (default) — used by Claude Desktop and similar local agents.
  Start with::

      python -m app.mcp.main

* **SSE** — used by network clients (browsers, remote agents, P3 ToolArch).
  Start with::

      python -m app.mcp.main --transport sse --port 8080

Environment variables:

* ``TOOLREF_API_URL`` — Base URL of the ToolRef backend API
  (default: ``http://localhost:8000``).
* ``LOG_LEVEL`` — Logging verbosity (default: ``INFO``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _configure_logging(level_name: str) -> None:
    """Set up root logger with a readable format."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,  # keep stdout clean for stdio MCP transport
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.mcp.main",
        description=(
            "ToolRef MCP Server — expose Agentic RAG capabilities via the "
            "Model Context Protocol (MCP)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport mode.  Use 'stdio' for Claude Desktop, 'sse' for network clients.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port to listen on (SSE mode only).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/interface to bind to (SSE mode only).",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help=(
            "Override the ToolRef backend API URL "
            "(default: $TOOLREF_API_URL or http://localhost:8000)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        metavar="LEVEL",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and start the MCP server."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Apply API URL override before importing server / tools ────────────
    if args.api_url:
        os.environ["TOOLREF_API_URL"] = args.api_url
        logger.info("TOOLREF_API_URL overridden to: %s", args.api_url)
    else:
        api_url = os.environ.get("TOOLREF_API_URL", "http://localhost:8000")
        logger.info("Using TOOLREF_API_URL: %s", api_url)

    # Import server after env is configured so tools.py picks up the URL
    from app.mcp.server import mcp  # noqa: PLC0415

    # ── Launch ────────────────────────────────────────────────────────────
    if args.transport == "sse":
        logger.info(
            "Starting ToolRef MCP Server (SSE) on %s:%d", args.host, args.port
        )
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        logger.info("Starting ToolRef MCP Server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
