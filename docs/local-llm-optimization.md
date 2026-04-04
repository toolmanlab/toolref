# P1 ToolRef — Local LLM Inference Optimization Report

**Date:** 2026-03-31
**Hardware:** Mac Mini M4 Pro (14-core), 64GB Unified Memory
**Scope:** Model upgrade (qwen2.5:14b → Qwen3-30B-A3B) + inference backend optimization (Ollama → LM Studio MLX)

---

## 1. Background

P1 ToolRef runs a full agentic RAG pipeline locally. The LLM is called multiple times per query:
- `analyze_query` — classify intent, extract entities (JSON output)
- `decompose_query` — split complex queries into sub-queries (JSON output)
- `grade_documents` — relevance grading (JSON output, skipped when reranker score > 0.5)
- `generate` — synthesize final answer with citations (natural language)
- `consistency_check` — verify answer consistency (disabled in MVP)

End-to-end latency is dominated by LLM inference time, especially when the model is called 3-5 times per query.

## 2. Model Upgrade: qwen2.5:14b → Qwen3-30B-A3B

### Why Qwen3-30B-A3B

| Metric | qwen2.5:14b | Qwen3-30B-A3B |
|--------|-------------|----------------|
| Parameters | 14B (dense) | 30B total, 3B active (MoE) |
| ArenaHard | ~75 | 91.0 |
| Architecture | Dense Transformer | Sparse MoE (Mixture of Experts) |
| GGUF size | ~9 GB (Q4) | ~18 GB (Q4) |
| MLX 4-bit size | N/A | 17.19 GB |

Qwen3-30B-A3B uses Mixture-of-Experts: 30B total parameters, but only 3B active per token. This gives GPT-4o-class quality (ArenaHard 91.0 vs 85.3) with low compute per token.

### Thinking Mode Issue

**Critical discovery:** Qwen3 ships with a built-in chain-of-thought "thinking mode" enabled by default.

**Problem:** When thinking mode is active:
- The model spends tokens in `reasoning_content` (internal reasoning) before generating `content` (visible output)
- With `max_tokens: 200`, ALL tokens go to reasoning — `content` is empty
- P1's LangChain pipeline reads `content` only → empty responses, parse failures
- Each LLM call takes 5-10x longer due to reasoning overhead

**Solution:** Append `/nothink` suffix to user messages. Implemented as a transparent `_NoThinkWrapper` in `backend/app/retrieval/llm.py`:
- Controlled by `LLM_DISABLE_THINKING=true` in `.env`
- Automatically appends `/nothink` to every LLM call's last human message
- Zero changes to prompt templates — model-agnostic design preserved
- When switching to non-Qwen3 models, simply set `LLM_DISABLE_THINKING=false`

**Code changed:**
- `backend/app/config.py` — added `llm_disable_thinking: bool = False`
- `backend/app/retrieval/llm.py` — added `_NoThinkWrapper` class

## 3. Inference Backend: Ollama vs LM Studio

### Architecture Difference

| | Ollama | LM Studio (0.4.2+) |
|--|--------|---------------------|
| Backend engine | llama.cpp + GGUF | MLX native |
| GPU interface | Metal (via llama.cpp) | Metal (via MLX, Apple-native) |
| Memory handling | Copy to GPU via Metal | Zero-copy unified memory |
| Quantization | GGUF Q4_K_M | MLX 4-bit |
| Batching | None | Continuous batching |
| API | OpenAI-compatible (:11434) | OpenAI-compatible (:1234) |

MLX is Apple's machine learning framework designed specifically for Apple Silicon's unified memory architecture. Key advantage: zero-copy memory access — the GPU reads weight tensors directly from unified memory without copying, unlike llama.cpp which uses Metal's buffer management with copy overhead.

### P1 Configuration

**Ollama (.env):**
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:30b-a3b
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_DISABLE_THINKING=false  # Ollama's ChatOllama handles thinking differently
```

**LM Studio (.env):**
```env
LLM_PROVIDER=openai
LLM_MODEL=qwen/qwen3-30b-a3b
OPENAI_API_KEY=lm-studio
OPENAI_API_BASE=http://host.docker.internal:1234/v1
LLM_DISABLE_THINKING=true
```

Both use the same `host.docker.internal` bridge — P1 runs in Docker, inference runs on host.

## 4. Benchmark Results

### 4.1 Isolated LLM Calls (no RAG pipeline)

**Query:** "什么是 RAG？用三句话解释。"

| Backend | Mode | Time | Tokens | Speed | Content |
|---------|------|------|--------|-------|---------|
| LM Studio MLX | thinking (default) | 6.67s | 199 tok | ~30 tok/s | Empty (all in reasoning) |
| LM Studio MLX | /nothink | 1.93s | 85 tok | ~44 tok/s | Complete 3-sentence answer |
| Ollama GGUF | default | N/A | N/A | ~12-18 tok/s | N/A (not tested in isolation with same query) |

**Query:** "Analyze query... JSON" (short structured output, ~42 tokens)

| Backend | Condition | Time |
|---------|-----------|------|
| LM Studio MLX (Ollama loaded) | Memory contention | 23.6s |
| LM Studio MLX (Ollama unloaded) | Clean memory | 1.16s |

**Critical finding:** Running Ollama and LM Studio simultaneously with the same 18GB model causes severe memory pressure. Ollama was holding qwen3:30b-a3b (33GB GPU) + qwen2.5:32b (21GB GPU) = 54GB in GPU memory. Combined with LM Studio's 17GB, total exceeded 64GB unified memory, causing swap thrashing and 20x slowdown.

**Action:** When using LM Studio, stop Ollama models with `ollama stop <model>`.

### 4.2 P1 End-to-End RAG Pipeline

**Query:** "What are the best practices for few-shot prompting?" (uncached)

| Backend | Total Latency | Notes |
|---------|---------------|-------|
| Ollama (qwen3:30b-a3b, no /nothink) | 67,803 ms | Thinking mode active, consumes tokens on internal reasoning |
| LM Studio MLX (qwen3:30b-a3b, /nothink, Ollama still loaded) | 150,998 ms | Memory contention — Ollama holding 54GB GPU |

**Query:** "What techniques can reduce hallucination in language models?" (uncached, Ollama unloaded)

| Backend | Total Latency | Breakdown |
|---------|---------------|-----------|
| LM Studio MLX (/nothink, clean memory) | 14,143 ms | See node breakdown below |

**Node-level breakdown (LM Studio MLX, clean memory):**

| Pipeline Node | Latency |
|---------------|---------|
| analyze_query (LLM) | 1,515 ms |
| decompose_query (LLM) | 1,407 ms |
| hybrid_retrieve (Milvus) | 1,102 ms |
| rerank (BGE-reranker-v2-m3) | 5,793 ms |
| grade_documents (fast-path) | 0 ms |
| generate (LLM) | 3,989 ms |
| **Total** | **14,143 ms** |

### 4.3 Comparison Summary

| Configuration | Model | End-to-End | Improvement |
|--------------|-------|-----------|-------------|
| Ollama + qwen2.5:14b (baseline) | 14B dense | ~16s | — |
| Ollama + qwen3:30b-a3b | 30B MoE | ~68s | -4.25x (regression) |
| **LM Studio MLX + qwen3:30b-a3b + /nothink** | **30B MoE** | **~14s** | **+14% faster, much better quality** |

**Winner:** LM Studio MLX backend with Qwen3-30B-A3B (/nothink mode)
- Faster than baseline qwen2.5:14b (14s vs 16s)
- Significantly higher quality (ArenaHard 91.0 vs ~75)
- Same memory footprint category (~18GB vs ~9GB, both fit in 64GB)

## 5. Memory Management Rules

Based on the memory contention discovery:

1. **Never run Ollama and LM Studio with large models simultaneously**
   - Each loads the full model into unified memory
   - 64GB is NOT enough for two 18GB+ models + Docker + OS
2. **Before starting LM Studio, unload Ollama models:**
   ```bash
   ollama stop qwen3:30b-a3b
   ollama stop qwen2.5:32b  # or any other loaded model
   ollama ps  # verify empty
   ```
3. **Before starting Ollama, stop LM Studio:**
   ```bash
   lms server stop
   ```
4. **Monitor with:** `ollama ps` (Ollama) and `lms ps` (LM Studio)

## 6. Known Issues

### 6.1 LangChain + Qwen3 reasoning_content Field
When `LLM_DISABLE_THINKING=false` (or with a model that returns `reasoning_content`), LangChain's `ChatOpenAI` may not correctly extract `content` from the response. The `_NoThinkWrapper` bypasses this by eliminating reasoning entirely.

### 6.2 Generate Node Error with LangGraph State
One test query ("How does LangGraph handle state management?") hit a generate error — "I encountered an error generating the answer." This may be related to LangChain's handling of the `reasoning_content` field leaking through despite `/nothink`. Needs further investigation.

### 6.3 Reranker Cold Start
BGE-reranker-v2-m3 runs on CPU (PyTorch). First rerank call after container restart takes ~10s (model loading) vs ~6s for subsequent calls.

## 7. Final Configuration

**Production .env settings (recommended):**
```env
LLM_PROVIDER=openai
LLM_MODEL=qwen/qwen3-30b-a3b
OPENAI_API_KEY=lm-studio
OPENAI_API_BASE=http://host.docker.internal:1234/v1
LLM_DISABLE_THINKING=true
```

**Startup sequence:**
```bash
# 1. Ensure Ollama models are unloaded
ollama stop qwen3:30b-a3b 2>/dev/null
ollama stop qwen2.5:32b 2>/dev/null

# 2. Start LM Studio server + load model
lms server start
lms load qwen/qwen3-30b-a3b -y

# 3. Start P1
cd ~/projects/toolref && docker compose up -d
```

## 8. Future Optimization Opportunities

| Area | Current | Target | Impact |
|------|---------|--------|--------|
| Reranker device | CPU (~6s) | Metal/MPS | -3-4s per query |
| analyze_query skip | Always runs | Skip for simple queries | -1.5s |
| Embedding model | all-MiniLM-L6-v2 (384d) | BGE-M3 (1024d) | Better retrieval quality |
| LM Studio context length | Default | Tune for P1 prompts | Memory efficiency |
| Ollama FLASH_ATTENTION | Not tested | OLLAMA_FLASH_ATTENTION=1 | Faster if using Ollama fallback |
