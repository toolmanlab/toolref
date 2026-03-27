"""ToolRef MCP Server package.

Exposes the Agentic RAG retrieval engine as an MCP-compatible server so
that external agents (Claude Desktop, P3 ToolArch, etc.) can invoke RAG
capabilities via the Model Context Protocol.

Entry point::

    # stdio mode (Claude Desktop)
    python -m app.mcp.main

    # SSE mode (network clients)
    python -m app.mcp.main --transport sse --port 8080
"""
