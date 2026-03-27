# ToolRef MCP Server

ToolRef 通过 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 将其 Agentic RAG 能力暴露给外部 Agent。任何兼容 MCP 的客户端（Claude Desktop、P3 ToolArch、自定义 SDK Agent 等）都可以通过本服务查询 ToolRef 知识库。

---

## 架构概览

```
Claude Desktop / P3 ToolArch / 其他 MCP Client
        │
        │  MCP Protocol (stdio 或 SSE)
        ▼
 ┌──────────────────┐
 │  MCP Server      │   app/mcp/
 │  (FastMCP)       │──────────────────────────────────┐
 └──────────────────┘                                  │
        │  HTTP  POST /api/v1/query                    │
        ▼                                              │
 ┌──────────────────┐                                  │
 │  FastAPI Backend │   app/api/query.py               │
 │  (port 8000)     │                                  │
 └──────────────────┘                                  │
        │                                              │
        ▼                                              │
 LangGraph RAG Pipeline  ◄─────────────────────────────┘
 (app/retrieval/graph.py)
```

MCP Server 是一层**薄代理**：它接收 MCP 协议请求，通过 HTTP 转发至 FastAPI 后端，由后端执行完整的 LangGraph 检索流程。

---

## 工具列表

### `toolref_query`

搜索 ToolRef 知识库并返回有依据的答案。

| 参数        | 类型   | 默认值      | 说明                          |
|-------------|--------|-------------|-------------------------------|
| `query`     | string | *必填*      | 自然语言问题（最多 2000 字符）|
| `namespace` | string | `"default"` | 文档命名空间，用于多租户隔离  |
| `top_k`     | int    | `5`         | 检索的源文档数量（1–20）      |

**返回值**

```json
{
  "answer": "生成的答案文本...",
  "sources": [
    {
      "doc_title": "文档标题",
      "chunk_text": "相关段落原文...",
      "url": "https://...",
      "score": 0.87
    }
  ],
  "confidence": 0.82
}
```

---

## 快速上手

### 前置条件

- ToolRef 后端正常运行（`http://localhost:8000`）
- Python ≥ 3.12，已安装依赖：`pip install -e ".[dev]"`

---

## Claude Desktop 配置

在 `~/Library/Application Support/Claude/claude_desktop_config.json`（macOS）中添加：

```json
{
  "mcpServers": {
    "toolref": {
      "command": "python",
      "args": ["-m", "app.mcp.main"],
      "cwd": "/path/to/toolref/backend",
      "env": {
        "TOOLREF_API_URL": "http://localhost:8000",
        "PYTHONPATH": "/path/to/toolref/backend"
      }
    }
  }
}
```

> **提示**：将 `/path/to/toolref/backend` 替换为实际的 `backend/` 目录绝对路径。

配置保存后重启 Claude Desktop，工具栏中将出现 **toolref_query** 工具。

---

## SSE 模式（网络客户端）

SSE 模式适用于需要通过网络连接的客户端，或在 Docker Compose 环境中运行。

### 本地启动

```bash
cd backend
python -m app.mcp.main --transport sse --port 8080
```

### 环境变量

| 变量名              | 默认值                    | 说明                              |
|---------------------|---------------------------|-----------------------------------|
| `TOOLREF_API_URL`   | `http://localhost:8000`   | ToolRef 后端 API 地址             |
| `LOG_LEVEL`         | `INFO`                    | 日志级别（DEBUG/INFO/WARNING）    |

### SSE 端点

| 端点      | 说明                                |
|-----------|-------------------------------------|
| `GET /sse`  | SSE 事件流（MCP 消息通道）        |
| `POST /messages` | MCP 客户端消息上行端点       |

### Python SDK 客户端示例

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8080/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        result = await session.call_tool(
            "toolref_query",
            arguments={
                "query": "What is RAG and how does it work?",
                "namespace": "default",
                "top_k": 5,
            },
        )
        print(result.content[0].text)
```

---

## Docker Compose

MCP Server 已集成在 `docker-compose.yml` 中，作为独立服务运行：

```bash
# 启动全部服务（含 MCP Server）
docker compose up -d

# 仅启动 MCP Server（后端服务需已运行）
docker compose up -d mcp-server

# 查看日志
docker compose logs -f mcp-server
```

容器内 MCP Server 通过 `http://backend:8000` 访问后端（由 `TOOLREF_API_URL` 环境变量控制）。

对外暴露 **8080 端口**（SSE 模式）：

```
http://localhost:8080/sse
```

---

## 命令行参数

```
python -m app.mcp.main --help

usage: python -m app.mcp.main [-h] [--transport {stdio,sse}] [--port PORT]
                               [--host HOST] [--api-url URL] [--log-level LEVEL]

ToolRef MCP Server — expose Agentic RAG capabilities via the Model Context Protocol (MCP).

options:
  --transport {stdio,sse}   MCP transport mode (default: stdio)
  --port PORT               TCP port for SSE mode (default: 8080)
  --host HOST               Host to bind for SSE mode (default: 0.0.0.0)
  --api-url URL             Override TOOLREF_API_URL
  --log-level LEVEL         Logging level (default: INFO)
```

---

## 故障排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Cannot connect to ToolRef backend` | 后端未启动或 URL 错误 | 检查 `TOOLREF_API_URL`，确认后端健康 |
| `HTTP 422` 错误 | 请求参数不合法（如 `top_k` 超出范围） | 检查参数范围：`top_k` 为 1–20 |
| `Timeout` 错误 | RAG 流程耗时过长（首次加载模型） | 等待模型预热后重试，或增大超时配置 |
| Claude Desktop 中看不到工具 | 配置路径错误或 PYTHONPATH 未设置 | 检查 `claude_desktop_config.json` 中的 `cwd` 和 `PYTHONPATH` |
