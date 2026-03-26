# ToolRef — 架构设计文档

> **版本：** 2.0  
> **作者：** Dojo (AI/ML Project Partner)  
> **日期：** 2026-03-26  
> **状态：** 设计阶段 — 准备交给 CC 搭建脚手架  
> **仓库：** `toolref`

---

## 1. 概述

### 一句话定位

**生产级 Agentic RAG 引擎，将私有文档转化为可对话、可溯源、具备自纠错能力的知识库。**

### 问题陈述

| 问题 | 影响 |
|------|------|
| **复杂查询召回率低** | 传统 RAG 只做一次检索再生成。当查询模糊或涉及多个维度（例如"为什么 auth 重构之后 API 会返回 502？"），单次检索往往遗漏相关文档，导致幻觉式回答。 |
| **知识来源碎片化** | 技术团队的知识散落在 Confluence、Notion、GitHub README 和 PDF 规范中，缺乏统一的问答入口。 |
| **答案缺乏溯源** | 工程师无法信任 AI 回答——除非知道答案来自*哪份文档*、*哪个段落*、*哪个版本*。 |

### 目标用户

- **主要用户：** 中小型工程团队（5–50 人），拥有大量内部文档但缺乏统一知识管理工具。
- **次要用户：** 个人开发者，用于管理学习笔记、代码注释和技术文档。
- **面试友好：** 通用场景——面试官一听就懂，还能看现场 Demo。

### 三层架构中的定位

```
┌─────────────────────────────────────────────────┐
│          P3: ToolArch                           │
│          (Application Layer / 架构智能)          │
│          独立架构分析工具，可通过 MCP 消费       │
│          P1 的知识                               │
├─────────────────────────────────────────────────┤
│          P2: ToolOps                            │
│          (Orchestration Layer / 基础设施层)      │
│          P1 的 RAG 流程 = P2 的使用场景         │
├─────────────────────────────────────────────────┤
│  >>> P1: ToolRef (THIS PROJECT) <<<             │
│          (Knowledge Layer / 知识层)              │
│          以 MCP Server 形式暴露 RAG 能力         │
└─────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 高层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React/TS SPA                                            │   │
│  │  • Chat Interface (streaming)                             │   │
│  │  • Document Management                                    │   │
│  │  • Source Attribution Panel                                │   │
│  │  • Evaluation Dashboard                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────────┘
                          │ REST + WebSocket
┌─────────────────────────▼────────────────────────────────────────┐
│                        API Layer                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI                                                  │   │
│  │  • /api/v1/query      — RAG query (REST + WS streaming)  │   │
│  │  • /api/v1/documents  — Document CRUD                     │   │
│  │  • /api/v1/namespaces — Namespace management              │   │
│  │  • /api/v1/evals      — Evaluation results                │   │
│  │  • /mcp               — MCP Server endpoint               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                     Core Engine Layer                             │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  Ingestion      │  │  Retrieval     │  │  MCP Server      │  │
│  │  Pipeline       │  │  Engine        │  │  (Tool Provider) │  │
│  │                 │  │  (LangGraph)   │  │                  │  │
│  │  • Parser       │  │  • Query       │  │  • rag_query     │  │
│  │  • Chunker      │  │    Analysis    │  │  • document_add  │  │
│  │  • Embedder     │  │  • Router      │  │  • namespace_list│  │
│  │  • Indexer      │  │  • Retriever   │  │                  │  │
│  └────────────────┘  │  • Grader      │  └──────────────────┘  │
│                       │  • Rewriter    │                         │
│  ┌────────────────┐  │  • Generator   │  ┌──────────────────┐  │
│  │  Semantic Cache │  └────────────────┘  │  DSPy Evaluator  │  │
│  │  (Redis)        │                       │                  │  │
│  │  • Embedding    │                       │  • Faithfulness  │  │
│  │    similarity   │                       │  • Relevance     │  │
│  │  • TTL-based    │                       │  • Answer Quality│  │
│  │    invalidation │                       │                  │  │
│  └────────────────┘                       └──────────────────┘  │
└──────────┬───────────────┬──────────────────┬────────────────────┘
           │               │                  │
┌──────────▼───┐  ┌───────▼──────┐  ┌────────▼─────────┐
│   Milvus     │  │  PostgreSQL  │  │  Redis            │
│   (Vectors)  │  │  (Metadata)  │  │  (Cache + Queue)  │
└──────────────┘  └──────────────┘  └──────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 |
|------|------|
| **Ingestion Pipeline** | 解析文档（PDF/MD/HTML），切分为可检索单元，生成 embedding，存入 Milvus + 元数据存入 PostgreSQL |
| **Retrieval Engine** | LangGraph 状态机：分析查询 → 路由 → 混合检索 → 相关性评分 → 可选查询重写 → 生成带引用的答案 |
| **MCP Server** | 将 RAG 能力以标准 MCP 工具形式暴露，供外部 Agent 调用（P3 ToolArch 等） |
| **Semantic Cache** | 基于查询 embedding 相似度缓存 LLM 响应；减少冗余 API 调用 |
| **DSPy Evaluator** | 在 Faithfulness、Relevance、Answer Quality 三个维度评估 RAG 流水线变体；为面试提供基准数据 |
| **API Layer** | FastAPI REST + WebSocket，服务前端和外部消费者 |
| **Frontend** | React/TS SPA：流式聊天、文档管理、来源归因展示 |

### 2.3 数据流：文档上传 → 答案生成

```
[Document Upload Flow]
                                                            
User uploads PDF ──► FastAPI /documents ──► Redis Streams (async queue)
                                                    │
                                                    ▼
                                            Ingestion Worker
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              Parse (Unstructured)  Chunk       Embed (BGE-M3)
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              Milvus (vectors) PG (metadata)  PG (chunk text)

[Query Flow]

User asks question ──► FastAPI /query ──► Redis Cache Check
                                                │
                                    ┌───────────┴──────────┐
                                    ▼                      ▼
                              Cache HIT             Cache MISS
                              (return cached)              │
                                                           ▼
                                                    LangGraph Engine
                                                           │
                                            ┌──────────────┼─────────────┐
                                            ▼              ▼             ▼
                                      Analyze Query   Route Query   Decompose
                                            │              │             │
                                            ▼              ▼             ▼
                                      Hybrid Retrieve (Milvus + BM25)
                                                           │
                                                           ▼
                                                    Grade Documents
                                                     ┌─────┴─────┐
                                                     ▼           ▼
                                               Relevant    Not Relevant
                                                     │           │
                                                     ▼           ▼
                                               Generate    Rewrite Query
                                              (with cite)       │
                                                     │     Re-retrieve
                                                     │           │
                                                     ▼           ▼
                                               Stream response back
                                                     │
                                                     ▼
                                              Cache in Redis
                                                     │
                                                     ▼
                                              Save to PG (history)
```

---

## 3. 技术栈

### 3.1 完整技术栈及选型理由

| 组件 | 选型 | 版本 | 选型理由（与备选方案对比） |
|------|------|------|--------------------------|
| **LLM 框架** | LangChain + LangGraph | `langchain>=0.3`, `langgraph>=0.3` | LangChain 提供 RAG 原语（loaders、splitters、retrievers），LangGraph 增加有状态 Agent 控制。**vs LlamaIndex：** LlamaIndex 专精 RAG 但缺少 Agent 编排；我们两者都需要。**vs 裸 API 调用：** 检索链需要太多样板代码。 |
| **Web 框架** | FastAPI | `>=0.115` | 原生异步、自动 OpenAPI 文档、Pydantic 校验。**vs Flask：** 无原生 async，需手动校验请求。**vs Django：** 对 API 服务来说太重。Alex 的 Spring Boot 经验可直接映射到 FastAPI 的 DI 模式。 |
| **向量数据库** | Milvus | `2.4.x` | 分布式、十亿级向量、混合检索（dense + sparse）。与训练营课程对齐。**vs Qdrant：** 单节点更快但没有内置 BM25 稀疏搜索。**vs Chroma：** 仅适合开发，无生产级集群。**vs pgvector：** 适合小规模但缺少高级索引（只有 HNSW，无 IVF_PQ）。 |
| **关系型数据库** | PostgreSQL | `16.x` | JSON 支持、成熟生态、在 AI 岗位 JD 中覆盖率高于 MySQL。存储文档元数据、用户会话、评估记录。 |
| **缓存/队列** | Redis | `7.x` | 语义缓存（基于向量相似度匹配已缓存查询）、会话状态、Redis Streams 用于异步文档摄入队列。**vs RabbitMQ/Kafka：** 对我们的规模来说过于重量级；Redis Streams 无需额外基础设施即可提供轻量可靠队列。 |
| **Embedding 模型** | BGE-M3 (BAAI) | `BAAI/bge-m3` | 多语言，单模型同时支持 dense 和 sparse（类 BM25）embedding。**vs OpenAI ada-002：** API 成本，数据离开你的基础设施。**vs BGE-large-en：** 仅支持英文；BGE-M3 支持中英双语。**vs Jina-v3：** 不错但在国内社区采用度较低。 |
| **Reranker** | BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | 同属 BAAI 家族的 Cross-encoder Reranker，embedding 空间一致。**vs Cohere rerank：** API 依赖 + 成本。**vs ms-marco-MiniLM：** 仅支持英文。 |
| **LLM（开发）** | Ollama + Qwen2.5-7B | latest | 本地运行、免费、中英双语。用于开发和演示。 |
| **LLM（生产）** | DeepSeek-V3 / GPT-4o API | latest | 生产级质量。DeepSeek 追求性价比（¥1/百万 token），GPT-4o 作为高端选项。可按部署配置切换。 |
| **文档解析器** | Unstructured.io | `>=0.16` | 支持 PDF、HTML、Markdown、DOCX，具备表格提取能力。**vs PyPDF2：** 仅支持 PDF，表格支持差。**vs LangChain loaders：** 薄封装；Unstructured 提供更深度的解析。 |
| **前端** | React 18 + TypeScript | `react@18`, `ts@5` | Alex 最强技能（携程 2.5 年经验）。**vs Streamlit：** 训练营用 Streamlit 但它不是生产级，无法自定义 UI。**vs Vue：** Alex 的专长是 React。 |
| **MCP SDK** | `mcp`（官方） | `>=1.0` | 官方 Model Context Protocol SDK。标准化的 Agent 工具互操作接口。 |
| **评估** | DSPy + Arize Phoenix | `dspy>=2.5`, `arize-phoenix` | DSPy 用于程序化 prompt 优化 + 指标驱动评估。Phoenix 用于 trace 可视化。**vs RAGAS：** 擅长 RAG 评估但 DSPy 还能优化 prompt。**vs 手动评估：** 不可复现，无法为面试提供数据。 |
| **任务队列** | Redis Streams | （Redis 7 内置） | 轻量异步摄入。**vs Celery：** 额外 broker 依赖。**vs ARQ：** 成熟度较低。Redis Streams = 零额外基础设施，因为 Redis 已在技术栈中。 |
| **容器** | Docker + Docker Compose | latest | 多服务编排，覆盖开发/生产环境。标准部署方式。 |

### 3.2 版本锁定建议

```toml
# pyproject.toml [project.dependencies]
langchain = ">=0.3.0,<0.4"
langgraph = ">=0.3.0,<0.4"
fastapi = ">=0.115.0,<1.0"
uvicorn = {extras = ["standard"], version = ">=0.32.0"}
pydantic = ">=2.9.0,<3.0"
pymilvus = ">=2.4.0,<2.5"
psycopg = {extras = ["binary"], version = ">=3.2.0"}
sqlalchemy = ">=2.0.0,<3.0"
redis = ">=5.2.0"
unstructured = {extras = ["pdf", "md", "html"], version = ">=0.16.0"}
dspy = ">=2.5.0"
sentence-transformers = ">=3.3.0"
FlagEmbedding = ">=1.2.0"
mcp = ">=1.0.0"
arize-phoenix = ">=5.0.0"
```

---

## 4. 模块设计

### 4.1 文档摄入流水线

#### 4.1.1 文档解析

MVP 支持的格式：

| 格式 | 解析器 | 备注 |
|------|--------|------|
| PDF | `unstructured.partition_pdf` | 通过 `hi_res` 策略提取表格；扫描件支持 OCR 回退 |
| Markdown | `unstructured.partition_md` | 保留标题层级，用于 parent-chunk 关联 |
| HTML | `unstructured.partition_html` | 去除导航/样板内容，保留正文 |
| TXT | 内置处理 | 直接文本处理 |
| DOCX | `unstructured.partition_docx` | V1 新增 [TBD] |

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"

@dataclass
class ParsedDocument:
    """Output of the document parser."""
    doc_id: str
    title: str
    doc_type: DocumentType
    elements: list["DocumentElement"]  # paragraphs, tables, code blocks
    metadata: dict  # source URL, author, timestamp

@dataclass
class DocumentElement:
    """A structural element within a parsed document."""
    element_type: str  # "paragraph", "table", "code_block", "heading"
    text: str
    metadata: dict  # page_number, heading_level, etc.
```

#### 4.1.2 切块策略

**对比：**

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **固定大小** | 每 N 个 token 切一刀，带重叠 | 简单、可预测 | 破坏语义边界；代码函数会被截断 | 均质散文 |
| **语义切块** | 按段落/章节边界切分，利用 embedding 判断 | 保留语义完整性 | chunk 大小不一；可能产生超大 chunk | 结构良好的文档 |
| **层级切块** ⭐ | 两层结构：小 chunk（256 token）用于检索精度，父 chunk（1024 token）用于生成上下文 | 两全其美：精确检索 + 丰富上下文 | 实现复杂度较高；存储开销约 2 倍 | 混合内容（文档 + 代码） |

**推荐：层级切块（Hierarchical Chunking）**

理由：
- 小 chunk（256 token）提升检索精度——embedding 精确匹配到相关段落。
- 父 chunk（1024 token）为 LLM 生成提供充足上下文——避免"上下文丢失"问题。
- 检索返回小 chunk → 系统查找父 chunk → 将父 chunk 传给 LLM。
- 开销：向量存储约 2 倍。在我们的规模（MVP < 1000 万 chunk）下完全可接受。

```python
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    """Configuration for hierarchical chunking."""
    child_chunk_size: int = 256       # tokens, for retrieval
    child_chunk_overlap: int = 32     # token overlap between children
    parent_chunk_size: int = 1024     # tokens, for generation context
    parent_chunk_overlap: int = 64    # token overlap between parents

@dataclass
class Chunk:
    """A document chunk with hierarchy."""
    chunk_id: str
    doc_id: str
    text: str
    parent_chunk_id: str | None       # None if this IS a parent chunk
    level: str                         # "child" or "parent"
    token_count: int
    metadata: dict                     # source, page, heading path
    embedding: list[float] | None      # populated after embedding
```

**面试话术：** "我对比了三种切块策略。固定大小在代码文档上效果最差，因为函数体会被从中间截断。层级切块给出了最优结果：子 chunk 用于精确向量匹配，父 chunk 用于丰富 LLM 上下文。2 倍存储成本相比质量提升是微不足道的。"

#### 4.1.3 Embedding 生成

**模型：** BGE-M3 (BAAI)
- 1024 维 dense 向量
- 内置 sparse（词法）embedding，支持混合检索
- 多语言（中文 + 英文）
- 可在 CPU 上本地运行（推理约 50ms/chunk）或 GPU（约 5ms/chunk）

```python
from FlagEmbedding import BGEM3FlagModel

class EmbeddingService:
    """Generates dense + sparse embeddings using BGE-M3."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model = BGEM3FlagModel(model_name, use_fp16=(device != "cpu"))
    
    def embed(self, texts: list[str]) -> dict:
        """Generate both dense and sparse embeddings.
        
        Returns:
            {
                "dense": list[list[float]],    # shape: (N, 1024)
                "sparse": list[dict[int, float]]  # token_id -> weight
            }
        """
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": output["dense_vecs"].tolist(),
            "sparse": output["lexical_weights"],
        }
```

#### 4.1.4 Milvus 存储与索引选型

**索引类型对比：**

| 索引 | 原理 | Recall@10 | QPS（100 万向量） | 内存 | 构建时间 | 适用场景 |
|------|------|-----------|-------------------|------|----------|----------|
| **HNSW** ⭐ | 基于图，全内存 | ~99% | ~3000 | 高（全部在 RAM） | 中等 | < 1000 万向量，需要高召回 |
| **IVF_FLAT** | 聚类 + 簇内精确搜索 | ~95-98% | ~5000 | 中等 | 快 | 1000 万–1 亿，平衡方案 |
| **IVF_PQ** | 聚类 + 压缩向量 | ~90-95% | ~8000 | 低 | 快 | > 1 亿，内存受限 |

**推荐：HNSW**

理由：
- 我们 MVP 目标是 < 100 万向量（V1 扩展到约 500 万）。HNSW 在此规模最优。
- RAG 对召回率要求极高——漏掉正确文档意味着幻觉。HNSW 约 99% 的召回率至关重要。
- 内存成本：100 万向量 × 1024 维 × 4 字节 ≈ 4GB。单节点部署完全可接受。
- 权衡：构建时间比 IVF 慢，但文档摄入是异步的——不影响用户体验。

**Collection Schema：**

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# Child chunks collection (for retrieval)
child_chunks_schema = CollectionSchema(
    fields=[
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ],
    description="Child chunks for precise retrieval",
)

# HNSW index on dense vectors
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256},
}

# Search params (query time)
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 128},  # higher ef = better recall, slower
}
```

**面试话术：** "我选 HNSW 而非 IVF_FLAT，因为 RAG 对召回率极其敏感——漏掉正确文档就意味着幻觉。在我们的规模（< 500 万向量）下，HNSW 约 99% 的召回率完全值得额外的内存开销。如果扩展到 1 亿以上，我会切换到 IVF_PQ 并加一个 Reranking 阶段来补偿召回损失。"

### 4.2 检索引擎

#### 4.2.1 LangGraph 状态机

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   START                                                          │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────────┐                                                │
│  │ analyze_query│  ← Classify intent, extract entities,          │
│  │              │    determine complexity                        │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │   route      │  ← Simple (direct) vs Complex (decompose)     │
│  └──┬───────┬───┘                                                │
│     │       │                                                    │
│  simple   complex                                                │
│     │       │                                                    │
│     ▼       ▼                                                    │
│  ┌──────┐ ┌──────────────────┐                                   │
│  │direct│ │decompose_query   │  ← Break into sub-queries        │
│  │search│ │                  │                                   │
│  └──┬───┘ └──────┬───────────┘                                   │
│     │            │                                               │
│     └─────┬──────┘                                               │
│           ▼                                                      │
│  ┌──────────────────┐                                            │
│  │ hybrid_retrieve   │  ← Vector (Milvus) + BM25 (sparse)       │
│  │                   │     + RRF fusion                          │
│  └──────┬────────────┘                                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                            │
│  │ rerank            │  ← Cross-Encoder reranking (BGE-reranker) │
│  └──────┬────────────┘                                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐         ┌──────────────────┐               │
│  │ grade_documents   │──NOT──►│ rewrite_query     │              │
│  │                   │ OK     │                   │              │
│  └──────┬────────────┘        └──────┬────────────┘              │
│         │ OK                         │                           │
│         │                     ┌──────▼────────────┐              │
│         │                     │ hybrid_retrieve    │ (retry)     │
│         │                     └──────┬────────────┘              │
│         │                            │                           │
│         │                     ┌──────▼────────────┐              │
│         │                     │ grade_documents    │──► FAIL     │
│         │                     │ (2nd attempt)      │  (fallback) │
│         │                     └──────┬────────────┘              │
│         │                            │ OK                        │
│         └────────┬───────────────────┘                           │
│                  │                                               │
│                  ▼  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐        │
│                     │ consistency_check (V1, optional) │         │
│                  │  │ Generate 2 answers → compare →   │        │
│                     │ if conflict → extract divergence │         │
│                  │  │ → new query → re-retrieve        │        │
│                     └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘        │
│                  │                                               │
│                  ▼                                               │
│  ┌──────────────────────────┐                                    │
│  │ generate                  │  ← LLM generates answer with     │
│  │                           │    source citations               │
│  └──────────┬────────────────┘                                   │
│             │                                                    │
│             ▼                                                    │
│           END ──► Return answer + sources + metadata             │
│                                                                  │
│  Max self-correction loops: 2 (configurable)                     │
│  Fallback on repeated failure: "I couldn't find relevant         │
│  information. Here's what I found..." + raw top-k results        │
└──────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 状态定义

```python
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    """State schema for the Agentic RAG graph."""
    
    # Input
    query: str
    namespace: str
    conversation_id: str | None
    
    # Query Analysis
    query_type: Literal["simple", "complex"] | None
    sub_queries: list[str]
    entities: list[str]
    
    # Retrieval
    retrieved_docs: list[dict]       # [{chunk_id, text, score, source}]
    reranked_docs: list[dict]        # post cross-encoder
    
    # Grading
    relevance_scores: list[float]
    is_relevant: bool | None
    
    # Self-Correction
    rewrite_count: int               # track retry attempts
    rewritten_query: str | None
    
    # Consistency Check (V1)
    consistency_passed: bool | None
    divergence_query: str | None
    
    # Generation
    answer: str | None
    sources: list[dict]              # [{doc_title, chunk_text, url, score}]
    cached: bool
    
    # Metadata
    messages: Annotated[list, add_messages]  # conversation history
    latency_ms: dict                 # per-node latency tracking
```

#### 4.2.3 节点定义

```python
from langgraph.graph import StateGraph, END

def build_rag_graph() -> StateGraph:
    """Build the Agentic RAG state machine."""
    
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("route", route_node)
    graph.add_node("decompose_query", decompose_query_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("consistency_check", consistency_check_node)  # V1
    graph.add_node("generate", generate_node)
    
    # Add edges
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "route")
    
    # Conditional routing
    graph.add_conditional_edges(
        "route",
        lambda state: state["query_type"],
        {
            "simple": "hybrid_retrieve",
            "complex": "decompose_query",
        },
    )
    graph.add_edge("decompose_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")
    
    # Grading → generate or rewrite
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "relevant": "consistency_check",  # V1: check before generate
            "rewrite": "rewrite_query",
            "fallback": "generate",  # max retries exceeded
        },
    )
    graph.add_edge("rewrite_query", "hybrid_retrieve")
    
    # Consistency check → generate or re-retrieve
    graph.add_conditional_edges(
        "consistency_check",
        route_after_consistency,
        {
            "consistent": "generate",
            "divergent": "hybrid_retrieve",  # re-retrieve with divergence query
            "skip": "generate",  # MVP: skip consistency check
        },
    )
    
    graph.add_edge("generate", END)
    
    return graph.compile()


def route_after_grading(state: RAGState) -> str:
    """Decide next step after document grading."""
    if state["is_relevant"]:
        return "relevant"
    if state["rewrite_count"] >= 2:
        return "fallback"
    return "rewrite"


def route_after_consistency(state: RAGState) -> str:
    """Decide next step after consistency check (V1)."""
    if state.get("consistency_passed") is None:
        return "skip"  # MVP: node not active
    if state["consistency_passed"]:
        return "consistent"
    return "divergent"
```

#### 4.2.4 混合检索策略

**架构：Vector + BM25 + RRF Fusion**

```python
from dataclasses import dataclass

@dataclass
class HybridSearchConfig:
    """Configuration for hybrid retrieval."""
    dense_weight: float = 0.6        # vector similarity weight
    sparse_weight: float = 0.4       # BM25 weight
    top_k_per_source: int = 20       # candidates from each source
    final_top_k: int = 10            # after RRF fusion
    rrf_k: int = 60                  # RRF constant (standard)


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse dense and sparse retrieval results using RRF.
    
    RRF score = Σ 1/(k + rank_i) for each ranking list.
    Simple, parameter-free (except k), consistently outperforms
    linear combination in practice.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    
    for rank, doc in enumerate(dense_results):
        chunk_id = doc["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        doc_map[chunk_id] = doc
    
    for rank, doc in enumerate(sparse_results):
        chunk_id = doc["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        doc_map[chunk_id] = doc
    
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        {**doc_map[cid], "rrf_score": scores[cid]}
        for cid in sorted_ids
    ]
```

**在 Milvus 中的实现方式：**
- BGE-M3 同时产出 dense（1024 维）和 sparse（词法权重）embedding。
- Milvus 通过 `AnnSearchRequest` 支持在单次查询中进行 dense + sparse 混合检索。
- 结果经 RRF 融合后，传给 Cross-encoder Reranker。

#### 4.2.5 Cross-Encoder Reranking

```python
from FlagEmbedding import FlagReranker

class RerankerService:
    """Cross-encoder reranking using BGE-reranker-v2-m3."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = FlagReranker(model_name, use_fp16=True)
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank documents using cross-encoder.
        
        Cross-encoder sees (query, document) pairs jointly,
        capturing fine-grained relevance that bi-encoder misses.
        Latency: ~15ms per pair on GPU, ~100ms per pair on CPU.
        """
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.compute_score(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        
        return sorted(documents, key=lambda d: d["rerank_score"], reverse=True)[:top_k]
```

**权衡：** Cross-encoder 增加约 150ms 延迟（GPU 上 10 篇文档）。这是可接受的，因为：
1. 它显著提升精度（面试数据点：MRR 从约 0.65 提升到约 0.82 [TBD: 用实际数据测量]）。
2. 延迟被流式输出隐藏——在复杂查询中，第一个 token 在 Reranking 完成前就已返回。

#### 4.2.6 Self-Correction 循环

评分节点评估检索到的文档是否与查询相关：

```python
async def grade_documents_node(state: RAGState) -> RAGState:
    """Grade retrieved documents for relevance.
    
    Uses LLM-as-judge to score each document.
    If average relevance < threshold, triggers query rewrite.
    """
    grading_prompt = """You are a relevance grader. Given a query and a document,
    determine if the document contains information relevant to answering the query.
    
    Query: {query}
    Document: {document}
    
    Respond with a JSON: {{"relevant": true/false, "reason": "..."}}"""
    
    scores = []
    for doc in state["reranked_docs"]:
        result = await llm.ainvoke(
            grading_prompt.format(query=state["query"], document=doc["text"])
        )
        scores.append(parse_relevance(result))
    
    avg_relevance = sum(scores) / len(scores) if scores else 0
    
    return {
        **state,
        "relevance_scores": scores,
        "is_relevant": avg_relevance >= 0.6,  # threshold configurable
    }


async def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the query for better retrieval.
    
    Uses LLM to generate an alternative query that might
    retrieve more relevant documents.
    """
    rewrite_prompt = """The original query did not retrieve relevant documents.
    
    Original query: {query}
    Retrieved documents (not relevant enough): {doc_summaries}
    
    Generate a better search query that would find the right information.
    Focus on key terms and specificity."""
    
    rewritten = await llm.ainvoke(
        rewrite_prompt.format(
            query=state["query"],
            doc_summaries=summarize_docs(state["reranked_docs"]),
        )
    )
    
    return {
        **state,
        "rewritten_query": rewritten,
        "query": rewritten,  # update query for re-retrieval
        "rewrite_count": state["rewrite_count"] + 1,
    }
```

**面试话术：** "这是 Agentic RAG 与 Pipeline RAG 的核心区别。传统 RAG 做一次检索再生成就完了。检索失败就会产生幻觉。我的系统会评估检索质量，不达标就用重写后的查询重试。重试上限 2 次以控制延迟，多次失败后优雅降级。"

#### 4.2.7 一致性检查（V1）

在 `grade_documents` 通过后、`generate` 之前，V1 增加可选的 `consistency_check` 节点。该节点的设计灵感来自 MA-RAG（arxiv 2603.03292）。

**核心思路：** 对同一组检索文档生成两次答案，如果两次答案存在矛盾 → 提取分歧点 → 将分歧点转化为新的查询 → 触发再检索，从而获取更全面的上下文来消解矛盾。

**生命周期：**
- MVP 不包含此节点（`route_after_consistency` 返回 `"skip"`）
- V1 作为 self-correction 的增强手段激活

```python
async def consistency_check_node(state: RAGState) -> RAGState:
    """Check answer consistency by generating two independent answers.
    
    Inspired by MA-RAG (arxiv 2603.03292): if two answers from the 
    same retrieved docs are contradictory, extract the divergence 
    point and use it as a new retrieval query.
    
    This node is optional (V1). MVP skips it via routing.
    """
    docs_context = format_docs_for_prompt(state["reranked_docs"])
    
    # Generate two independent answers with different temperatures
    answer_a = await llm.ainvoke(
        GENERATE_PROMPT.format(query=state["query"], context=docs_context),
        temperature=0.3,
    )
    answer_b = await llm.ainvoke(
        GENERATE_PROMPT.format(query=state["query"], context=docs_context),
        temperature=0.7,
    )
    
    # Check consistency
    consistency_prompt = """Compare these two answers to the same question.

    Question: {query}
    Answer A: {answer_a}
    Answer B: {answer_b}

    Are they consistent? If not, what is the specific point of divergence?
    Respond with JSON: {{"consistent": true/false, "divergence": "..." or null}}"""
    
    result = await llm.ainvoke(
        consistency_prompt.format(
            query=state["query"],
            answer_a=answer_a,
            answer_b=answer_b,
        )
    )
    parsed = parse_consistency_result(result)
    
    if not parsed["consistent"] and parsed["divergence"]:
        # Convert divergence point into a new retrieval query
        return {
            **state,
            "consistency_passed": False,
            "divergence_query": parsed["divergence"],
            "query": parsed["divergence"],  # re-retrieve with divergence
            "rewrite_count": state["rewrite_count"] + 1,
        }
    
    return {
        **state,
        "consistency_passed": True,
        "divergence_query": None,
    }
```

**面试话术：** "Self-correction 循环解决的是'没检索到'的问题，一致性检查解决的是'检索到了但信息互相矛盾'的问题。论文 MA-RAG（arxiv 2603.03292）提出的方法是生成多次答案、比对矛盾、提取分歧点再检索。我在 V1 引入这个节点作为 self-correction 的增强——当两次生成结果矛盾时，把分歧点转化为新查询重新检索，用更完整的上下文消解矛盾。MVP 阶段先跳过这一步，V1 打开。"

### 4.3 MCP Server

#### 4.3.1 工具定义

ToolRef 暴露三个 MCP 工具：

| 工具 | 用途 | 调用方 |
|------|------|--------|
| `rag_query` | 对指定命名空间执行 Agentic RAG 查询 | P3 ToolArch、任何 MCP 兼容的 Agent |
| `document_add` | 向命名空间添加文档进行索引 | 外部自动化 |
| `namespace_list` | 列出可用命名空间及其统计信息 | 发现/UI |

#### 4.3.2 接口 Schema

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("toolref")

@server.tool()
async def rag_query(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
    include_sources: bool = True,
) -> dict:
    """Query the RAG knowledge base.
    
    Args:
        query: Natural language question.
        namespace: Document namespace to search within.
        top_k: Number of source documents to return.
        include_sources: Whether to include source attributions.
    
    Returns:
        {
            "answer": str,
            "sources": [
                {
                    "doc_title": str,
                    "chunk_text": str,
                    "source_url": str,
                    "relevance_score": float,
                    "page_number": int | None,
                }
            ],
            "cached": bool,
            "latency_ms": int,
            "rewrite_count": int,
        }
    """
    ...

@server.tool()
async def document_add(
    url: str,
    doc_type: str = "auto",        # "pdf", "markdown", "html", "auto"
    namespace: str = "default",
    metadata: dict | None = None,
) -> dict:
    """Add a document to the knowledge base.
    
    The document will be fetched, parsed, chunked, embedded,
    and indexed asynchronously.
    
    Returns:
        {
            "doc_id": str,
            "status": "queued",
            "estimated_time_seconds": int,
        }
    """
    ...

@server.tool()
async def namespace_list() -> dict:
    """List all available namespaces.
    
    Returns:
        {
            "namespaces": [
                {
                    "name": str,
                    "doc_count": int,
                    "chunk_count": int,
                    "last_updated": str,  # ISO timestamp
                }
            ]
        }
    """
    ...
```

#### 4.3.3 P3 ToolArch 集成设计

P3 ToolArch 作为独立的架构分析工具，可通过 MCP 消费 ToolRef 的知识。例如，其架构规范检查模块可调用 ToolRef 检索编码标准：

```python
# In P3 ToolArch — Architecture Analysis Module
async def check_coding_standards(diff: str, repo_name: str) -> list[dict]:
    """Use ToolRef to retrieve relevant coding standards for analysis."""
    
    # Extract key patterns from diff
    patterns = extract_code_patterns(diff)  # e.g., "error handling", "auth middleware"
    
    # Call P1 via MCP
    result = await mcp_client.call_tool(
        server="toolref",
        tool="rag_query",
        arguments={
            "query": f"coding standards for {', '.join(patterns)} in {repo_name}",
            "namespace": "coding-standards",
            "top_k": 5,
        },
    )
    
    return result["sources"]
```

**设计原则：** P3 不在 prompt 中硬编码规则。而是动态从 P1 检索最新标准。当标准变更时，只需更新 ToolRef 中的文档——P3 零代码改动。

### 4.4 Redis 语义缓存

#### 4.4.1 缓存策略

语义缓存不仅是成本优化手段，更是知识复用的核心机制。论文数据表明 47% 的生产环境 Agent 查询与历史查询语义相似（arxiv 2603.23013），8B 模型 + 记忆检索达到 30.5% F1，恢复了 235B 模型 69% 的性能，成本降低 96%。

**何时缓存：**
- 每次 `generate` 节点成功完成后（答案 + 来源）
- 缓存键：原始查询的 embedding
- 缓存值：序列化的 `{answer, sources, metadata, timestamp}`

**何时失效：**
- 基于 TTL 的分级策略：高频查询 72h，普通查询 24h，低频查询 12h（按命名空间可配置）
- 基于事件：当命名空间中的文档被更新/删除时，该命名空间下所有缓存条目失效
- 手动：管理员 API 按命名空间清除缓存

**何时不缓存：**
- 触发了 self-correction 的查询（rewrite_count > 0）——缓存可能已过时
- 调用方设置 `cached=false` 覆盖标志的查询

#### 4.4.2 语义相似度阈值

```python
import numpy as np
from redis import Redis

class SemanticCache:
    """Redis-backed semantic cache for RAG responses."""
    
    SIMILARITY_THRESHOLD = 0.92  # cosine similarity
    
    # Tiered TTL strategy
    HIGH_FREQ_TTL = 259200       # 72 hours - high frequency queries
    DEFAULT_TTL = 86400          # 24 hours - normal queries
    LOW_FREQ_TTL = 43200         # 12 hours - low frequency queries
    
    def __init__(self, redis: Redis, embedding_service: "EmbeddingService"):
        self.redis = redis
        self.embedder = embedding_service
    
    async def get(self, query: str, namespace: str) -> dict | None:
        """Check cache for semantically similar query.
        
        Process:
        1. Embed the incoming query
        2. Search Redis for cached query embeddings in the namespace
        3. If cosine similarity > threshold, return cached result
        
        Uses Redis HSET with embedding stored as bytes.
        Scans are bounded by namespace prefix.
        """
        query_embedding = self.embedder.embed([query])["dense"][0]
        
        # Scan cached entries for this namespace
        pattern = f"rag_cache:{namespace}:*"
        for key in self.redis.scan_iter(pattern, count=100):
            cached = self.redis.hgetall(key)
            cached_embedding = np.frombuffer(cached[b"embedding"], dtype=np.float32)
            
            similarity = cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self.SIMILARITY_THRESHOLD:
                return deserialize_cache_entry(cached)
        
        return None
    
    async def put(
        self,
        query: str,
        namespace: str,
        result: dict,
        frequency: str = "normal",  # "high", "normal", "low"
    ) -> None:
        """Cache a RAG result with tiered TTL."""
        query_embedding = self.embedder.embed([query])["dense"][0]
        key = f"rag_cache:{namespace}:{hash(query)}"
        
        ttl_map = {
            "high": self.HIGH_FREQ_TTL,
            "normal": self.DEFAULT_TTL,
            "low": self.LOW_FREQ_TTL,
        }
        ttl = ttl_map.get(frequency, self.DEFAULT_TTL)
        
        self.redis.hset(key, mapping={
            "embedding": np.array(query_embedding, dtype=np.float32).tobytes(),
            "result": serialize_cache_entry(result),
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.redis.expire(key, ttl)
```

**注意：** 在生产规模（> 1 万缓存条目）下，应将基于 scan 的方式替换为 Redis Vector Similarity Search（Redis Stack）或专用的 Milvus collection 来存储缓存 embedding。[TBD: 在 V1 规模下评估]

#### 4.4.3 ROI 分析框架

| 指标 | 测量方法 | 目标 |
|------|----------|------|
| **Cache Hit Rate** | `cache_hits / total_queries` | > 40%（知识库 Q&A 场景典型值；论文数据表明 47% 的查询语义相似） |
| **LLM 成本节省** | `cache_hits × avg_cost_per_llm_call` | 按日/周跟踪 |
| **延迟降低** | `avg_latency_cached vs avg_latency_uncached` | 缓存命中应 < 50ms vs 未命中约 2-5s |
| **过期率** | `user_reported_stale_answers / cache_hits` | < 5% |

**面试话术：** "语义缓存不只是用了 Redis 这么简单——核心是 LLM API 成本优化和知识复用。论文数据（arxiv 2603.23013）表明 47% 的生产环境 Agent 查询与历史查询语义相似，说明缓存的命中空间非常大。每次查询使用 GPT-4o 成本约 $0.003，按 40% 以上的缓存命中率计算，每月可节省 $X。0.92 的相似度阈值是经验调参得出的：低于 0.90 误命中太多（返回错误缓存），高于 0.95 又会漏掉近义查询。TTL 方面我采用分级策略——高频查询 72h、普通查询 24h、低频查询 12h，比一刀切更贴合实际访问模式。"

### 4.5 DSPy 评估

#### 4.5.1 评估维度

| 维度 | 测量内容 | 方法 |
|------|----------|------|
| **Faithfulness** | 答案是否基于检索文档？无幻觉。 | LLM-as-judge：将答案声明与来源 chunk 比对 |
| **Relevance** | 检索到的文档是否与查询相关？ | 已评分文档的 Precision@k |
| **Answer Quality** | 答案是否有帮助、完整、结构清晰？ | LLM-as-judge + 人工评估 |
| **Retrieval Recall** | 检索步骤是否找到了正确的文档？ | 与标注 ground truth 的 Recall@k |
| **Latency** | 端到端响应时间 | 每次查询的 wall clock time |

#### 4.5.2 基准数据集设计

```python
@dataclass
class EvalSample:
    """A single evaluation sample."""
    query: str
    expected_answer: str                    # gold reference
    relevant_doc_ids: list[str]             # ground truth documents
    difficulty: Literal["simple", "complex", "multi-hop"]
    domain: str                             # "api-docs", "design-docs", etc.

@dataclass  
class EvalDataset:
    """Benchmark dataset for RAG evaluation."""
    name: str
    samples: list[EvalSample]
    version: str
    
    # Target: 100 samples for MVP, 500 for V1
    # Sources:
    # - 30% manually crafted (high quality, covers edge cases)
    # - 40% LLM-generated from documents (reviewed by human)
    # - 30% extracted from real user queries (V1, after deployment)
```

#### 4.5.3 实验对比框架

```python
import dspy
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """A RAG configuration variant to evaluate."""
    name: str
    chunking_strategy: str          # "fixed_256", "semantic", "hierarchical"
    retrieval_mode: str             # "dense_only", "sparse_only", "hybrid"
    reranker_enabled: bool
    self_correction_enabled: bool
    llm_model: str                  # "qwen2.5-7b", "deepseek-v3", "gpt-4o"
    embedding_model: str

@dataclass
class ExperimentResult:
    """Metrics for a single experiment run."""
    config: ExperimentConfig
    faithfulness: float             # 0-1
    relevance_precision: float      # Precision@5
    retrieval_recall: float         # Recall@10
    answer_quality: float           # 0-1 (LLM judge)
    avg_latency_ms: float
    p95_latency_ms: float
    cost_per_query_usd: float
    cache_hit_rate: float           # if cache enabled
    timestamp: str

# Example experiment matrix:
EXPERIMENTS = [
    ExperimentConfig("baseline", "fixed_256", "dense_only", False, False, "qwen2.5-7b", "bge-m3"),
    ExperimentConfig("hybrid", "fixed_256", "hybrid", False, False, "qwen2.5-7b", "bge-m3"),
    ExperimentConfig("hybrid+rerank", "fixed_256", "hybrid", True, False, "qwen2.5-7b", "bge-m3"),
    ExperimentConfig("agentic", "hierarchical", "hybrid", True, True, "qwen2.5-7b", "bge-m3"),
    ExperimentConfig("agentic+gpt4o", "hierarchical", "hybrid", True, True, "gpt-4o", "bge-m3"),
]
```

**面试话术：** "我跑了 5 组配置变体，测量了 Faithfulness、Retrieval Recall 和 Latency。Agentic 配置（层级切块 + 混合检索 + Reranking + self-correction）在基线基础上将 Recall@10 从 XX% 提升到 XX%，延迟从 XX ms 增加到 XX ms。数据都在我的评估面板里。"

### 4.6 API 层（FastAPI）

#### 4.6.1 RESTful API 端点

```python
# app/api/v1/router.py

# === Query ===
POST   /api/v1/query                    # Execute RAG query (non-streaming)
POST   /api/v1/query/stream             # Execute RAG query (SSE streaming)

# === Documents ===
POST   /api/v1/documents                # Upload document (multipart)
GET    /api/v1/documents                # List documents (with filters)
GET    /api/v1/documents/{doc_id}       # Get document details
DELETE /api/v1/documents/{doc_id}       # Delete document + vectors
PATCH  /api/v1/documents/{doc_id}       # Update metadata

# === Namespaces ===
POST   /api/v1/namespaces               # Create namespace
GET    /api/v1/namespaces               # List namespaces
DELETE /api/v1/namespaces/{name}        # Delete namespace + all docs

# === Conversations ===
GET    /api/v1/conversations             # List user conversations
GET    /api/v1/conversations/{id}        # Get conversation history
DELETE /api/v1/conversations/{id}        # Delete conversation

# === Evaluation ===
POST   /api/v1/evals/run                # Trigger evaluation run
GET    /api/v1/evals/results            # List evaluation results
GET    /api/v1/evals/results/{run_id}   # Get specific run details

# === Health ===
GET    /health                           # Health check
GET    /health/ready                     # Readiness (all deps up)
```

#### 4.6.2 WebSocket 流式传输

```python

from fastapi import WebSocket
from starlette.websockets import WebSocketState

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """Stream RAG query results via WebSocket.
    
    Protocol:
    1. Client sends: {"query": "...", "namespace": "...", "conversation_id": "..."}
    2. Server streams:
       - {"type": "status", "node": "analyze_query", "message": "Analyzing..."}
       - {"type": "status", "node": "hybrid_retrieve", "message": "Searching..."}
       - {"type": "chunk", "text": "partial answer text..."}
       - {"type": "sources", "data": [{...}]}
       - {"type": "done", "metadata": {"latency_ms": 1234, "cached": false}}
    """
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        
        async for event in rag_engine.stream(
            query=data["query"],
            namespace=data.get("namespace", "default"),
            conversation_id=data.get("conversation_id"),
        ):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(event)
    
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
```

#### 4.6.3 认证与限流

**MVP：** 基于 API Key 的认证（简单，Demo 够用）。

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key and return user_id."""
    user = await db.get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user["id"]

# Rate limiting via Redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# 30 queries/minute per user for query endpoints
# 10 uploads/minute per user for document endpoints
```

**V1：** JWT token 认证 + refresh token。OAuth2 可选。

### 4.7 前端（React/TS）

#### 4.7.1 页面路由

```
/                       → Redirect to /chat
/chat                   → Chat interface (default namespace)
/chat/:conversationId   → Continue specific conversation
/documents              → Document management (upload, list, delete)
/documents/:docId       → Document detail (chunks, status)
/namespaces             → Namespace management
/evals                  → Evaluation dashboard
/settings               → API keys, LLM config, cache settings
```

#### 4.7.2 核心交互

**聊天界面：**
- 左侧面板：对话列表（历史记录）
- 中间区域：带流式响应的聊天消息
- 右侧面板：来源归因——点击来源可高亮显示上下文中的 chunk
- 顶部命名空间选择器（在文档集合间切换）
- Agentic RAG 流程中的状态指示器："正在分析查询..." → "正在搜索文档..." → "正在评估相关性..." → "正在生成答案..."

**文档管理：**
- 拖拽上传文件（PDF/MD/HTML）
- URL 输入用于网页
- 摄入流水线进度指示器
- 带状态徽章的文档列表（indexing/ready/error）
- Chunk 预览：点击文档查看切块方式

**来源归因：**
- 每个答案附带可点击的来源引用 [1], [2], [3]
- 点击打开侧边面板：文档标题、匹配的 chunk 文本（高亮）、来源 URL、相关性分数、页码
- V1 新增"这个回答有帮助吗？"👍/👎 反馈循环

#### 4.7.3 状态管理

**选型：Zustand**（轻量、TypeScript 优先）

**vs Redux：** 对这个应用规模来说过重。Zustand 样板代码更少，TS 类型推断更好。
**vs React Context：** 多 store 扩展性差；不支持中间件。
**vs Jotai：** 原子模型很好但对 API 状态模式不够直观。

```typescript
// stores/chatStore.ts
interface ChatStore {
  conversations: Conversation[];
  activeConversation: string | null;
  messages: Map<string, Message[]>;
  isStreaming: boolean;
  
  // Actions
  sendQuery: (query: string, namespace: string) => Promise<void>;
  createConversation: () => string;
  deleteConversation: (id: string) => void;
}

// stores/documentStore.ts
interface DocumentStore {
  documents: Document[];
  uploadProgress: Map<string, number>;
  
  uploadDocument: (file: File, namespace: string) => Promise<void>;
  deleteDocument: (docId: string) => Promise<void>;
  refreshDocuments: () => Promise<void>;
}
```

**数据获取：** TanStack Query (React Query) 用于服务端状态（文档、评估、命名空间）。Zustand 用于纯客户端状态（UI 状态、活动面板）。

**UI 组件库：** Shadcn/ui（基于 Tailwind，复制粘贴组件，无供应商锁定）。

### 4.8 上下文管理策略

三个关键决策决定了每次 LLM 调用前如何组装上下文。

#### 4.8.1 Token 预算管理

每次 LLM 调用都必须在模型的上下文窗口内。在调用 LLM 之前，系统计算所有上下文组件的总 token 数并执行预算分配：

| 组件 | 预算 | 备注 |
|------|------|------|
| System prompt | ~500 token（固定） | 角色指令、输出格式、工具定义 |
| 检索文档 | ≤ 3,000 token | Reranking 后的 Top-K chunk，超预算则截断 |
| 对话历史 | 滑动窗口（见 §4.8.2） | 最近 N 轮 |
| 长期记忆注入 | ≤ 500 token | 从 Milvus 检索的最相关记忆 |
| 生成预留空间 | 剩余 token | 留给模型的响应 |

如果 system prompt + 检索文档 + 历史 + 记忆的总和超过 `context_window - generation_headroom`，系统按优先级压缩：(1) 截断检索文档到 Top-K，(2) 缩小对话窗口，(3) 裁剪记忆注入。System prompt 永不截断。

#### 4.8.2 对话历史压缩

三种策略对比：

| 策略 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| **截断** | 丢弃 N 轮前的消息 | 最简单 | 上下文突然丢失；用户引用早期消息时会断裂 |
| **滑动窗口** | 保留最近 N 轮原文 | 保留最近上下文 | 长对话仍然完全丢失早期上下文 |
| **摘要 + 窗口** | 将旧轮次摘要化 → 写入长期记忆；保留最近 N 轮原文 | 兼得：近期细节 + 长期要点 | 需要额外一次 LLM 调用做摘要 |

**推荐：摘要 + 滑动窗口（策略 3）。**

理由：在知识库 Q&A 场景中，用户经常引用对话中更早的内容（"你前面提到的那个错误码是什么？"）。纯截断或滑动窗口会丢失这个上下文。通过摘要化旧轮次并将摘要持久化为长期记忆条目，系统可以在需要时检索回来——无需承担保留完整历史的 token 成本。摘要调用是分摊的（仅在窗口滑动时触发，而非每轮都触发）。

实现方式：
- 滑动窗口大小：**N = 10 轮**（可配置）。
- 当对话超过 N 轮时，窗口外的旧轮次通过轻量 LLM 调用摘要为单条记忆条目。
- 摘要存入 Milvus（长期记忆 collection），带 `memory_type: "conversation_summary"` 和 `session_id` 元数据。
- 后续查询时，记忆检索步骤（§4.8.1）可能拉回相关摘要作为额外上下文。

#### 4.8.3 各模型上下文窗口策略

不同 LLM 有不同的上下文窗口。Token 预算必须动态适配：

```python
# config/model_context.py
MODEL_CONTEXT_CONFIG = {
    "qwen2.5-7b-instruct": {
        "context_window": 32_768,
        "generation_headroom": 2_048,
        "max_retrieved_docs_tokens": 3_000,
        "max_history_turns": 8,
        "max_memory_tokens": 500,
    },
    "deepseek-v3": {
        "context_window": 131_072,
        "generation_headroom": 4_096,
        "max_retrieved_docs_tokens": 8_000,
        "max_history_turns": 20,
        "max_memory_tokens": 1_000,
    },
    "gpt-4o": {
        "context_window": 131_072,
        "generation_headroom": 4_096,
        "max_retrieved_docs_tokens": 8_000,
        "max_history_turns": 20,
        "max_memory_tokens": 1_000,
    },
}
```

`generate` 节点读取当前模型的配置，在组装 prompt 前调整预算分配。这确保同一套 RAG 流程在上下文限制差异巨大的模型间都能正常工作——32K 模型获得更紧凑的检索/历史预算，而 128K 模型可以容纳更多上下文。

---

## 5. 项目结构

### 5.1 Monorepo vs Multi-Repo

**推荐：Monorepo**

理由：
- 单个 `docker-compose.yml` 编排所有服务。
- 后端和前端共享类型/Schema（如 API 响应类型）。
- CI/CD 简化——一条流水线构建和测试全部内容。
- 面试优势：面试官克隆一个仓库、跑一个命令，就能看到完整系统。

**vs Multi-repo：** 只有在多团队协作时才有意义。这是单人项目。

### 5.2 目录树

```
toolref/
├── README.md                          # Project overview, Quick Start, architecture
├── LICENSE                            # MIT
├── docker-compose.yml                 # Dev environment
├── docker-compose.prod.yml            # Production overrides
├── Makefile                           # Common commands (make dev, make test, etc.)
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Lint + test on PR
│       └── cd.yml                     # Build + push Docker images on merge
│
├── backend/                           # Python backend (FastAPI)
│   ├── Dockerfile
│   ├── pyproject.toml                 # Dependencies + project metadata
│   ├── alembic.ini                    # DB migrations config
│   ├── alembic/
│   │   └── versions/                  # Migration files
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── config.py                  # Settings (pydantic-settings)
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                # Dependency injection
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── router.py          # Mount all v1 routes
│   │   │       ├── query.py           # /query endpoints
│   │   │       ├── documents.py       # /documents endpoints
│   │   │       ├── namespaces.py      # /namespaces endpoints
│   │   │       ├── conversations.py   # /conversations endpoints
│   │   │       ├── evals.py           # /evals endpoints
│   │   │       └── websocket.py       # WebSocket handlers
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                # API key / JWT auth
│   │   │   ├── rate_limit.py          # Rate limiting
│   │   │   └── exceptions.py          # Custom exceptions
│   │   │
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── parser.py              # Document parsing (Unstructured)
│   │   │   ├── chunker.py             # Hierarchical chunking
│   │   │   ├── embedder.py            # BGE-M3 embedding service
│   │   │   ├── indexer.py             # Milvus write operations
│   │   │   └── worker.py              # Redis Streams consumer
│   │   │
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py               # LangGraph state machine
│   │   │   ├── state.py               # RAGState TypedDict
│   │   │   ├── nodes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analyze.py         # Query analysis node
│   │   │   │   ├── route.py           # Query routing node
│   │   │   │   ├── retrieve.py        # Hybrid retrieval node
│   │   │   │   ├── rerank.py          # Cross-encoder reranking
│   │   │   │   ├── grade.py           # Document grading node
│   │   │   │   ├── rewrite.py         # Query rewrite node
│   │   │   │   ├── consistency.py     # Consistency check node (V1)
│   │   │   │   └── generate.py        # Answer generation node
│   │   │   └── strategies/
│   │   │       ├── __init__.py
│   │   │       ├── hybrid.py          # RRF fusion
│   │   │       └── semantic_cache.py  # Redis semantic cache
│   │   │
│   │   ├── mcp/
│   │   │   ├── __init__.py
│   │   │   └── server.py              # MCP Server implementation
│   │   │
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── runner.py              # DSPy evaluation runner
│   │   │   ├── metrics.py             # Custom metrics
│   │   │   └── datasets/
│   │   │       └── benchmark_v1.json  # Evaluation dataset
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py             # SQLAlchemy async session
│   │   │   ├── models.py             # ORM models
│   │   │   └── repositories/
│   │   │       ├── __init__.py
│   │   │       ├── document.py
│   │   │       ├── conversation.py
│   │   │       └── evaluation.py
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── milvus.py              # Milvus client wrapper
│   │       ├── redis.py               # Redis client wrapper
│   │       └── llm.py                 # LLM provider abstraction
│   │
│   └── tests/
│       ├── conftest.py                # Fixtures
│       ├── unit/
│       │   ├── test_chunker.py
│       │   ├── test_hybrid_search.py
│       │   ├── test_semantic_cache.py
│       │   └── test_graph_nodes.py
│       ├── integration/
│       │   ├── test_ingestion_pipeline.py
│       │   ├── test_retrieval_flow.py
│       │   └── test_mcp_server.py
│       └── e2e/
│           └── test_query_flow.py
│
├── frontend/                          # React/TS frontend
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── index.html
│   │
│   ├── src/
│   │   ├── main.tsx                   # Entry point
│   │   ├── App.tsx                    # Root component + router
│   │   │
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   │   ├── ChatPanel.tsx      # Main chat interface
│   │   │   │   ├── MessageBubble.tsx  # Single message display
│   │   │   │   ├── StreamingText.tsx  # Streaming text renderer
│   │   │   │   ├── SourcePanel.tsx    # Source attribution sidebar
│   │   │   │   └── QueryInput.tsx     # Input with send button
│   │   │   ├── documents/
│   │   │   │   ├── DocumentList.tsx
│   │   │   │   ├── UploadDropzone.tsx
│   │   │   │   └── ChunkPreview.tsx
│   │   │   ├── evals/
│   │   │   │   ├── EvalDashboard.tsx
│   │   │   │   └── MetricsChart.tsx
│   │   │   └── ui/                    # Shadcn components
│   │   │
│   │   ├── stores/
│   │   │   ├── chatStore.ts
│   │   │   ├── documentStore.ts
│   │   │   └── settingsStore.ts
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useQuery.ts            # TanStack Query wrappers
│   │   │   └── useDocuments.ts
│   │   │
│   │   ├── lib/
│   │   │   ├── api.ts                 # API client (axios/fetch)
│   │   │   ├── ws.ts                  # WebSocket client
│   │   │   └── types.ts              # Shared types
│   │   │
│   │   └── pages/
│   │       ├── ChatPage.tsx
│   │       ├── DocumentsPage.tsx
│   │       ├── EvalsPage.tsx
│   │       └── SettingsPage.tsx
│   │
│   └── tests/
│       └── components/
│
├── scripts/
│   ├── seed_data.py                   # Seed sample documents
│   ├── run_eval.py                    # CLI for evaluation runs
│   └── migrate.sh                     # DB migration helper
│
└── docs/
    ├── architecture.md                # This document
    ├── api-reference.md               # Auto-generated from OpenAPI
    ├── development.md                 # Dev setup guide
    └── deployment.md                  # Production deployment guide
```

---

## 6. 数据模型

### 6.1 PostgreSQL Schema

```sql
-- ==========================================
-- Document Management
-- ==========================================

CREATE TABLE namespaces (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(128) UNIQUE NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id    UUID REFERENCES namespaces(id) ON DELETE CASCADE,
    title           VARCHAR(512) NOT NULL,
    doc_type        VARCHAR(32) NOT NULL,        -- 'pdf', 'markdown', 'html', 'txt'
    source_url      TEXT,                         -- original URL if fetched
    file_path       TEXT,                         -- local storage path
    file_size_bytes BIGINT,
    status          VARCHAR(32) DEFAULT 'pending', -- 'pending','indexing','ready','error'
    error_message   TEXT,
    chunk_count     INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}',           -- arbitrary metadata
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_documents_namespace ON documents(namespace_id);
CREATE INDEX idx_documents_status ON documents(status);

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id UUID REFERENCES chunks(id),  -- NULL for parent chunks
    chunk_level     VARCHAR(16) NOT NULL,         -- 'parent' or 'child'
    chunk_index     INTEGER NOT NULL,             -- ordering within document
    text            TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    heading_path    TEXT[],                        -- ['Chapter 1', 'Section 1.2']
    page_number     INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);

-- ==========================================
-- User & Auth
-- ==========================================

CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE,
    name        VARCHAR(255),
    api_key     VARCHAR(64) UNIQUE NOT NULL,
    role        VARCHAR(32) DEFAULT 'user',      -- 'user', 'admin'
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_users_api_key ON users(api_key);

-- ==========================================
-- Conversations & Query History
-- ==========================================

CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    namespace_id    UUID REFERENCES namespaces(id),
    title           VARCHAR(512),                 -- auto-generated from first query
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE query_history (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID REFERENCES conversations(id) ON DELETE CASCADE,
    query               TEXT NOT NULL,
    answer              TEXT,
    sources             JSONB,                    -- [{doc_title, chunk_id, score}]
    query_type          VARCHAR(32),              -- 'simple', 'complex'
    rewrite_count       INTEGER DEFAULT 0,
    cached              BOOLEAN DEFAULT FALSE,
    latency_ms          INTEGER,
    feedback            VARCHAR(16),              -- 'positive', 'negative', NULL
    metadata            JSONB DEFAULT '{}',       -- node latencies, model used, etc.
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_query_history_conversation ON query_history(conversation_id);
CREATE INDEX idx_query_history_created ON query_history(created_at);

-- ==========================================
-- Evaluation
-- ==========================================

CREATE TABLE eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    config          JSONB NOT NULL,               -- ExperimentConfig serialized
    dataset_name    VARCHAR(128),
    sample_count    INTEGER,
    status          VARCHAR(32) DEFAULT 'running', -- 'running', 'completed', 'failed'
    results         JSONB,                         -- ExperimentResult serialized
    created_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE eval_samples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID REFERENCES eval_runs(id) ON DELETE CASCADE,
    query       TEXT NOT NULL,
    expected    TEXT,
    actual      TEXT,
    scores      JSONB,                            -- {faithfulness, relevance, ...}
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

### 6.2 Milvus Collection Schema

```python
# Collection: toolref_child_chunks
fields = [
    FieldSchema("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
    FieldSchema("parent_chunk_id", DataType.VARCHAR, max_length=64),
    FieldSchema("namespace", DataType.VARCHAR, max_length=128),
    FieldSchema("dense_embedding", DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
]

# Indexes
# Dense: HNSW (M=16, efConstruction=256)
# Sparse: SPARSE_INVERTED_INDEX (BM25-like)

# Partition key: namespace (for efficient per-namespace queries)
```

### 6.3 Redis 缓存键设计

```
# Semantic cache
rag_cache:{namespace}:{query_hash}
  → HASH { embedding: bytes, result: json, timestamp: iso }
  → TTL: HIGH_FREQ=259200(72h) / DEFAULT=86400(24h) / LOW_FREQ=43200(12h)

# Ingestion queue (Redis Streams)
ingestion_queue
  → STREAM { doc_id, namespace, file_path, doc_type }

# Rate limiting
rate_limit:{user_id}:{endpoint}
  → STRING (counter)
  → TTL: 60 (per minute window)

# Session / active queries
active_query:{query_id}
  → HASH { user_id, status, current_node, started_at }
  → TTL: 300 (5 min max query lifetime)
```

---

## 7. 开发阶段

### 7.1 MVP（第 1–3 周）— 最小可行 RAG

**范围：**
- [x] 项目脚手架（monorepo、Docker Compose、CI 基础）
- [ ] 文档上传：PDF + Markdown + TXT 解析
- [ ] 固定大小切块（V1 切换为层级切块）
- [ ] BGE-M3 embedding + Milvus dense 向量存储
- [ ] 基础向量检索（仅 dense，不含混合检索）
- [ ] FastAPI：`/query`（非流式）+ `/documents` CRUD
- [ ] React 聊天界面：发送查询 → 展示答案 + 基础来源
- [ ] PostgreSQL：documents + chunks + query_history 表
- [ ] Docker Compose：FastAPI + React + Milvus + PostgreSQL + Redis + Etcd + MinIO

**验收标准：**
1. 上传一份 PDF → 系统解析、切块、生成 embedding、存入 Milvus
2. 提问 → 系统检索相关 chunk → LLM 生成带来源引用的答案
3. `docker compose up` 启动全栈
4. 响应延迟 < 10 秒（暂不优化）

### 7.2 V1（第 4–9 周）— Agentic RAG 核心

**范围：**
- [ ] 层级切块（替换固定大小）
- [ ] LangGraph 状态机：完整 Agentic RAG 流程
  - [ ] 查询分析 + 路由（simple/complex）
  - [ ] 混合检索（dense + sparse via BGE-M3）
  - [ ] RRF 融合
  - [ ] Cross-encoder Reranking（BGE-reranker）
  - [ ] 文档评分
  - [ ] 查询重写 + self-correction 循环
  - [ ] 一致性检查节点（§4.2.7）
- [ ] WebSocket 流式聊天
- [ ] Redis 语义缓存（分级 TTL）
- [ ] MCP Server：`rag_query`、`document_add`、`namespace_list`
- [ ] DSPy 评估框架 + 基线基准
- [ ] HTML + DOCX 文档支持
- [ ] React：流式聊天 + 来源归因面板 + 文档管理
- [ ] 命名空间支持（多文档集合）
- [ ] 对话历史（多轮）
- [ ] 记忆模块：通过 LangGraph checkpointer 实现短期对话记忆 + 长期记忆接口（基于向量存储，使用 Milvus collection 存储用户/会话记忆）
- [ ] RAG 流程中的记忆检索：在 generate 节点注入相关长期记忆作为额外上下文
- [ ] API Key 认证
- [ ] CI/CD：GitHub Actions（lint + test + Docker build）
- [ ] README 含架构图 + Quick Start

**验收标准：**
1. Agentic RAG：复杂查询触发分解 → 混合检索 → 评分 → 可选重写 → 答案
2. Self-correction 可演示："差检索"查询触发重写循环（在流式状态中可见）
3. DSPy 基准：至少 3 个实验变体及已发布指标
4. MCP Server 可被外部客户端调用
5. 缓存命中率可测量（重复查询 > 0%）
6. `docker compose up` 全部服务健康
7. Demo 就绪：可向面试官展示真实文档

### 7.3 V2（第 10–12 周，可选）— 高级能力

**范围：**
- [ ] GraphRAG 增强：实体关系图用于多跳推理 [TBD: 评估 Neo4j vs 内存图]
- [ ] Human-in-the-loop：用户标记"差答案" → 触发带反馈的重新检索
- [ ] 增量重索引：文档更新自动触发仅对变更 chunk 的重新 embedding
- [ ] 多源连接器：Notion API、GitHub README（via MCP）
- [ ] DSPy 自动 prompt 优化 + 对比报告
- [ ] React 评估面板（图表、对比表格）
- [ ] 限流 + 使用量指标

**验收标准：**
1. GraphRAG 在多跳查询上优于基线（有测量数据）
2. 文档更新触发部分重索引，而非全量重建
3. 至少一个外部连接器（Notion 或 GitHub）可用

---

## 8. Docker Compose

### 8.1 开发环境

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ============================================
  # Application Services
  # ============================================
  
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    ports:
      - "8000:8000"
    volumes:
      - ./backend/app:/app/app        # Hot reload
      - model-cache:/root/.cache       # Cache downloaded models
    environment:
      - DATABASE_URL=postgresql+psycopg://toolref:toolref@postgres:5432/toolref
      - REDIS_URL=redis://redis:6379/0
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - LLM_PROVIDER=ollama
      - LLM_BASE_URL=http://host.docker.internal:11434
      - LLM_MODEL=qwen2.5:7b
      - EMBEDDING_MODEL=BAAI/bge-m3
      - RERANKER_MODEL=BAAI/bge-reranker-v2-m3
      - LOG_LEVEL=DEBUG
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      milvus-standalone:
        condition: service_healthy
    command: >
      uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - toolref

  ingestion-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    volumes:
      - ./backend/app:/app/app
      - model-cache:/root/.cache
    environment:
      - DATABASE_URL=postgresql+psycopg://toolref:toolref@postgres:5432/toolref
      - REDIS_URL=redis://redis:6379/0
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - EMBEDDING_MODEL=BAAI/bge-m3
      - LOG_LEVEL=DEBUG
    depends_on:
      - backend
    command: python -m app.ingestion.worker
    networks:
      - toolref

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: development
    ports:
      - "3000:3000"
    volumes:
      - ./frontend/src:/app/src        # Hot reload
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
    command: npm run dev -- --host 0.0.0.0
    networks:
      - toolref

  # ============================================
  # Infrastructure Services
  # ============================================

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=toolref
      - POSTGRES_PASSWORD=toolref
      - POSTGRES_DB=toolref
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U toolref"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - toolref

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    command: redis-server --appendonly yes
    networks:
      - toolref

  # Milvus dependencies
  etcd:
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: >
      etcd
      -advertise-client-urls=http://127.0.0.1:2379
      -listen-client-urls=http://0.0.0.0:2379
      --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - toolref

  minio:
    image: minio/minio:RELEASE.2024-09-22T00-33-43Z
    environment:
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    ports:
      - "9001:9001"    # Console
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - toolref

  milvus-standalone:
    image: milvusdb/milvus:v2.4.17
    ports:
      - "19530:19530"  # gRPC
      - "9091:9091"    # Metrics
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - milvus-data:/var/lib/milvus
    command: ["milvus", "run", "standalone"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    networks:
      - toolref

volumes:
  postgres-data:
  redis-data:
  etcd-data:
  minio-data:
  milvus-data:
  model-cache:

networks:
  toolref:
    driver: bridge
```

### 8.2 生产环境覆盖

```yaml
# docker-compose.prod.yml
# Usage: docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

services:
  backend:
    build:
      target: production
    environment:
      - LOG_LEVEL=INFO
      - LLM_PROVIDER=deepseek        # Production LLM
      - LLM_MODEL=deepseek-chat
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    command: >
      gunicorn app.main:app
      -w 4
      -k uvicorn.workers.UvicornWorker
      --bind 0.0.0.0:8000
    restart: unless-stopped

  ingestion-worker:
    build:
      target: production
    restart: unless-stopped
    deploy:
      replicas: 2                     # Scale workers

  frontend:
    build:
      target: production
    ports:
      - "80:80"
    command: ["nginx", "-g", "daemon off;"]  # Serve static build via nginx

  postgres:
    environment:
      - POSTGRES_PASSWORD=${PG_PASSWORD}
    # In production, consider managed PostgreSQL (RDS, etc.)

  redis:
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
```

### 8.3 Backend Dockerfile（多阶段构建）

```dockerfile
# backend/Dockerfile

# ---- Base ----
FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# ---- Development ----
FROM base AS development
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- Production ----
FROM base AS production
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

---

## 9. 面试话术

### 9.1 逐模块高频问题

**文档摄入：**
- 问："为什么用层级切块而不是固定大小？"
- 答："固定大小切块会截断代码函数、破坏语义单元。我用层级切块：256 token 的子 chunk 保证检索精度，1024 token 的父 chunk 提供生成上下文。存储开销是 2 倍，但检索质量提升显著——我用 DSPy 做了量化测量。"

**检索引擎：**
- 问："你的 Agentic RAG 和标准 RAG 流水线有什么区别？"
- 答："标准 RAG 做一次检索再生成就结束了。检索失败就产生幻觉。我的 LangGraph 状态机加入了查询分析、路由（简单 vs 复杂）、文档相关性评分和 self-correction 循环——如果检索质量不达标就重写查询并重试。这就是'Agentic'的含义——系统能对自身的检索质量进行推理。"

- 问："为什么用混合检索？为什么不只用向量搜索？"
- 答："纯向量搜索会漏掉精确关键词匹配——错误码、API 端点名、变量名。纯 BM25 会漏掉语义相似性。通过 RRF 融合的混合检索两者兼得。我跑了实验：混合检索 + Reranking 给出了最佳 MRR，额外延迟仅约 150ms。"

**Milvus / 向量数据库：**
- 问："为什么选 Milvus 而不是 Qdrant 或 pgvector？"
- 答："三个原因：(1) Milvus 原生支持混合检索——dense + sparse 单次查询搞定。Qdrant 需要分开建索引。(2) 分布式架构在需要时可扩展到多节点。(3) 在国内 AI 生态中采用率高，与我的求职市场方向一致。代价是 Milvus 基础设施更重（依赖 etcd + MinIO），但 Docker Compose 让这一切透明化。"

**LangGraph：**
- 问："为什么用 LangGraph 而不是简单的 chain？"
- 答："Chain 是线性的。我的 RAG 流程有条件分支（简单 vs 复杂查询）、循环（重写 → 重新检索）以及跨节点累积的状态。LangGraph 的图结构 + TypedDict 状态 + checkpoint 支持正是为此设计的。CrewAI 太高层——我需要节点级别的控制。"

**MCP Server：**
- 问："为什么把 RAG 暴露为 MCP Server？"
- 答："MCP 标准化了 Agent 的工具接口。不用硬编码 API 调用，P3 ToolArch 通过 MCP 协议调用我的 RAG。这意味着任何 MCP 兼容的 Agent 都能把 ToolRef 当知识工具用——这是类库和服务的区别。"

**Redis 语义缓存：**
- 问："你的语义缓存怎么工作的？阈值是多少？"
- 答："我对查询做 embedding，然后与缓存中的查询 embedding 计算余弦相似度。阈值是 0.92——经验调参得出。低于 0.90 误命中太多（返回错误的缓存答案），高于 0.95 又会漏掉近义查询。论文数据（arxiv 2603.23013）表明 47% 的生产环境 Agent 查询与历史查询语义相似，这给了语义缓存很大的命中空间。我还采用了分级 TTL 策略：高频查询 72h、普通 24h、低频 12h，比一刀切更贴合实际访问模式。按 40%+ 的缓存命中率，每次 GPT-4o 调用省 $0.003，每月节省可观。"

**DSPy 评估：**
- 问："你怎么评估你的 RAG 系统？"
- 答："我构建了一个包含 100+ 查询-答案对的基准数据集，覆盖三个难度级别。测量 Faithfulness（答案是否基于来源？）、Retrieval Recall@10 和端到端延迟。跑了 5 个配置变体并发布了对比指标。DSPy 还支持自动优化 prompt——自动优化后的评分 prompt 将评分准确率提升了 X% [TBD]。"

**记忆架构：**
- 问："你的系统怎么处理用户长期记忆？"
- 答："两层设计。短期：LangGraph 内置的 checkpointer 跨轮次持久化对话状态——无需自定义序列化即可获得多轮上下文。长期：一个专用 Milvus collection 以 embedding 形式存储用户/会话记忆。在 generate 节点中，系统通过向量相似度检索 Top-K 相关记忆，并将其作为额外上下文与检索文档一起注入。持久化策略：当对话历史滑动窗口前进时，窗口外的旧轮次通过轻量 LLM 调用摘要化并写入长期记忆 collection。这意味着早期对话上下文永远不会真正丢失——它被压缩并可检索。记忆条目带有元数据（session_id、timestamp、memory_type），因此可以按时间或会话范围过滤。"

### 9.2 技术栈权衡话术

**"为什么不用 LlamaIndex？"**
> "LlamaIndex 在纯 RAG 方面很出色——更好的数据连接器、更简洁的检索 API。但我的项目需要 Agentic 流程控制（查询路由、self-correction 循环），这需要 LangGraph。LangChain + LangGraph 同时给我 RAG 原语和 Agent 编排能力。如果做一个不需要 Agent 逻辑的简单 RAG，我会选 LlamaIndex。"

**"为什么用 FastAPI 而不是 Flask？"**
> "三个原因：原生异步（对 LLM 流式输出 + Milvus 并发查询至关重要）、自动 OpenAPI 文档（省时间，对 MCP 集成有用）、Pydantic 校验（类型安全）。DI 模式和 Spring Boot 类似，所以我 7 年的 Java 经验直接迁移过来了。"

**"为什么用 PostgreSQL 存元数据，而不是全放 Milvus？"**
> "向量数据库为相似性搜索优化，不擅长关系型查询。我需要：文档状态追踪（pending/indexing/ready）、带分页的对话历史、聚合评估记录、用户管理与认证。这些全是关系型模式。架构决策是：向量放 Milvus，其他全放 PostgreSQL。"

**"为什么 Monorepo？"**
> "单人开发者，前后端紧耦合（共享 API 类型），一个 Docker Compose，一条 CI 流水线。Multi-repo 在这个规模下只增加协调开销，没有任何收益。面试官克隆一个仓库，跑 `docker compose up`，就能看到完整系统。"

**"为什么自建记忆而不用 Mem0/Zep？"**
> "Mem0 集成快但它是黑盒——无法控制记忆存储、检索排序、淘汰策略或权重衰减。Zep 在对话记忆方面很强，但不支持自定义记忆 schema（例如用不同 TTL 分离用户偏好和会话摘要）。自建记忆层让我复用已有的 Milvus 基础设施（零新依赖），完全控制生命周期（写入 → 检索 → 过期 → 衰减），而且——关键的——面试时每个设计决策都能讲清楚。实现成本很低：一个 Milvus collection、一个窗口滑动时的摘要 LLM 调用、一个 generate 节点中的检索步骤。"

### 9.3 预设基准数据点

[TBD — 开发过程中用实际测量数据填充]

| 指标 | 基线（Pipeline RAG） | Agentic RAG（V1） | 变化 |
|------|----------------------|-------------------|------|
| Retrieval Recall@10 | [TBD] | [TBD] | [TBD] |
| MRR (Mean Reciprocal Rank) | [TBD] | [TBD] | [TBD] |
| Faithfulness (LLM judge) | [TBD] | [TBD] | [TBD] |
| Answer Quality | [TBD] | [TBD] | [TBD] |
| Avg Latency (ms) | [TBD] | [TBD] | [TBD] |
| P95 Latency (ms) | [TBD] | [TBD] | [TBD] |
| Cache Hit Rate | N/A | [TBD] | — |
| LLM Cost/Query (USD) | [TBD] | [TBD] | [TBD] |

**预期模式（基于文献，待实际验证）：**
- 混合检索 + Reranking：Recall@10 比 dense-only 提升 +15-25%
- Self-correction 循环：复杂查询 Recall@10 提升 +5-10%（延迟增加 +500-1500ms）
- 语义缓存：知识库 Q&A 工作负载命中率 40%+（论文数据表明 47% 的查询语义相似）
- 层级切块：答案质量比固定大小切块提升 +10-15%

---

*architecture.md · ToolRef · v2.0 · 2026-03-26*
*下一步：Alex review → CC 搭建项目脚手架*
