# ToolRef — References

> 持续更新的参考文献库。论文、竞品、开源项目、技术博客。
> 每条标注：关联模块 + 对我们的启发/警示 + 状态（待读/已读/已应用）

---

## 📄 论文

### RAG 核心

| 论文 | 关键发现 | 关联模块 | 对 ToolRef 的影响 | 状态 |
|------|---------|---------|------------------|------|
| [Knowledge Access Beats Model Size](https://arxiv.org/abs/2603.23013) (2026-03) | 8B+记忆干掉235B无记忆(30.5% vs 13.7% F1)，47%查询语义重复，成本降96% | 语义缓存 | 缓存从"优化"升级为"核心特性"；分级 TTL；面试弹药 | ✅ 已读 → 待应用 |
| [MA-RAG: Multi-Round Agentic RAG](https://arxiv.org/abs/2603.03292) (2026-02) | 把答案不一致当主动信号驱动新一轮检索，+6.8分 | Self-correction | V1 加 consistency_check 节点（双重自检） | ✅ 已读 → V1 应用 |
| [Reasoner-Executor-Synthesizer](https://arxiv.org/abs/2603.22367) (2026-03) | O(1)上下文窗口的 Agent 架构 | Context Management | 待读，可能影响上下文管理策略 | ⏳ 待读 |
| [CRAG: Corrective RAG](https://arxiv.org/abs/2401.15884) (2024-01) | 检索文档质量评估 + web search 回退 + 知识精炼 | LangGraph 流程 | 已参考设计 grade→rewrite 循环 | ✅ 已应用 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) (2023-10) | 模型自我反思何时检索、检索结果是否相关、生成是否有据 | Self-correction | 影响 grading prompt 设计 | ✅ 已读 |
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403) (2024-03) | 根据查询复杂度自适应选择 RAG 策略 | Query routing | 影响 simple/complex 路由逻辑 | ⏳ 待读 |

### 检索与向量

| 论文 | 关键发现 | 关联模块 | 对 ToolRef 的影响 | 状态 |
|------|---------|---------|------------------|------|
| [BGE-M3](https://arxiv.org/abs/2402.03216) (2024-02) | 单模型同时输出 dense+sparse+ColBERT 三种表示 | Embedding | 我们用 dense+sparse 做混合检索 | ✅ 已应用 |
| [RRF: Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (2009) | 简单参数无关的排名融合，持续优于加权组合 | Hybrid Retrieval | RRF k=60 已采用 | ✅ 已应用 |

### Context Engineering

| 论文 | 关键发现 | 关联模块 | 对 ToolRef 的影响 | 状态 |
|------|---------|---------|------------------|------|
| [ALARA for Agents](https://arxiv.org/abs/2603.XXXXX) (2026-03) | 最小权限上下文工程，可组合多 Agent 团队 | Namespace 隔离 | 验证了 namespace 隔离的设计方向 | ⏳ 待读 |
| [Conversation Tree Architecture](https://arxiv.org/abs/2603.XXXXX) (2026-03) | 结构化多分支上下文管理 | 对话历史 | 可能影响 conversation memory 设计 | ⏳ 待读 |

### Agent 记忆

| 论文 | 关键发现 | 关联模块 | 对 ToolRef 的影响 | 状态 |
|------|---------|---------|------------------|------|
| [MemCollab](https://arxiv.org/abs/2603.23234) (2026-03) | 跨 Agent 记忆协作，对比轨迹蒸馏 | Memory/MCP | namespace 间知识共享场景参考 | ⏳ 待读 |

### 评估

| 论文 | 关键发现 | 关联模块 | 对 ToolRef 的影响 | 状态 |
|------|---------|---------|------------------|------|
| [RAGAS](https://arxiv.org/abs/2309.15217) (2023-09) | RAG 评估框架：Faithfulness, Answer Relevancy, Context Precision/Recall | DSPy Eval | 评估维度参考 | ✅ 已读 |
| [ARES](https://arxiv.org/abs/2311.09476) (2023-11) | 自动化 RAG 评估，少标注数据 | DSPy Eval | 评估方法论参考 | ⏳ 待读 |

---

## 🔧 开源项目 — 竞品/参考

### 直接竞品（MCP RAG Server）

| 项目 | Stars | 特点 | 差距（相对 ToolRef） | 监控频率 |
|------|-------|------|---------------------|---------|
| [mcp-ragdocs](https://github.com/xxx) | ~500 | MCP RAG server, 基础向量检索 | 无 namespace, 无 reranking, 无 eval, 无 UI | 月度 |
| [mcp-local-rag](https://github.com/xxx) | ~300 | 本地文件 RAG via MCP | 无混合检索, 无 self-correction | 月度 |
| [knowledge-mcp](https://github.com/xxx) | ~200 | 简单知识库 MCP | Demo 级别 | 月度 |

### 上游依赖（重点监控）

| 项目 | Stars | 我们用的 | 监控重点 | 频率 |
|------|-------|---------|---------|------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | ~10K | 核心 RAG 流程引擎 | 版本更新、API 变更、新特性（checkpoint 优化等） | 周度 |
| [Milvus](https://github.com/milvus-io/milvus) | ~30K | 向量存储 | 2.5 版本特性、混合检索改进 | 月度 |
| [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) | ~8K | Embedding + Reranker | 新模型版本、benchmark 对比 | 月度 |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | ~10K | 文档解析 | PDF 解析改进、新格式支持 | 月度 |
| [DSPy](https://github.com/stanfordnlp/dspy) | ~20K | 评估+prompt 优化 | API 变更（DSPy 更新快）、新优化器 | 月度 |
| [MCP SDK](https://github.com/modelcontextprotocol/python-sdk) | ~5K | MCP Server 实现 | 协议版本更新、新能力 | 月度 |

### 同赛道参考（不是直接竞品，但可以学习）

| 项目 | Stars | 参考价值 | 状态 |
|------|-------|---------|------|
| [RAGFlow](https://github.com/infiniflow/ragflow) | ~76K | 企业级 RAG 平台架构，重型但完整 | 已分析 |
| [Mem0](https://github.com/mem0ai/mem0) | ~48K | Agent 记忆管理，不是 RAG 但 memory 层可参考 | 已分析 |
| [Dify](https://github.com/langgenius/dify) | ~80K | Low-code RAG 平台，UI 交互参考 | 了解 |
| [Anything LLM](https://github.com/Mintplex-Labs/anything-llm) | ~30K | 本地 RAG，简洁的文档管理 UI 可参考 | 待看 |

---

## 📝 技术博客/文章

| 文章 | 来源 | 关键内容 | 对 ToolRef 的影响 |
|------|------|---------|------------------|
| Context Engineering (Karpathy/Lütke, 2025) | Twitter/Blog | "管理 LLM 看到的一切"，聚焦300token > 散漫113K | 理论基础 — 已融入定位 |
| Agentic Engineering Patterns (Simon Willison, 2026-03) | Blog | Coding Agent 工作模式系统化 | Agent 交互模式参考 |
| MCP vs CLI token 消耗 (Scalekit, 2026-03) | HN/Blog | MCP 3 服务 55K token，CLI 替代方案 | MCP Server 设计要注意 token 效率 |

---

## ⚠️ 踩坑预警

| 风险 | 来源 | 预防措施 |
|------|------|---------|
| MCP token 消耗过大 | Scalekit 基准 + HN 讨论 | MCP 工具返回精简结构化数据，不返回大段文本 |
| LangGraph API 频繁变更 | 社区反馈 | pin 版本 `>=0.3,<0.4`，关注 changelog |
| DSPy 更新激进 | 社区反馈 | pin 版本，写 adapter 层隔离 |
| Milvus standalone 内存占用 | 实际部署 | 开发环境限制内存，生产环境监控 |
| BGE-M3 中文 sparse 效果 | 待验证 | MVP 阶段对比 dense-only vs hybrid 在中文文档上的表现 |
| 语义缓存误命中 | 论文 + 常识 | 0.92 阈值 + namespace 隔离缓存，V1 加人工反馈修正 |

---

## 📅 更新日志

- **2026-03-26**: 初始创建。纳入 Pulse 推送论文分析结果、3/25 竞品调研成果
- 下次更新：P0 骨架搭完后，补充实际使用中发现的新参考
