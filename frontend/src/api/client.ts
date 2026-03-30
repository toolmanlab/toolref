/**
 * ToolRef API client.
 *
 * Thin wrapper around fetch providing typed request/response
 * helpers for each backend endpoint.  All methods are async and
 * throw `ApiError` on non-2xx responses.
 *
 * Streaming:
 *   `queryStream` opens a Server-Sent Events connection to
 *   `POST /api/v1/query/stream` and calls the provided callbacks
 *   for each chunk, on completion, and on error.
 */

import type {
  DocumentSummary,
  QueryRequest,
  QueryResponse,
  Source,
  UploadDocumentResponse,
} from "../types";

// ── Configuration ─────────────────────────────────────────────────────────────

const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ── Error ─────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Core fetch helper ─────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore JSON parse failure
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

// ── Health ────────────────────────────────────────────────────────────────────

/** GET /health — check all backend components. */
export async function getHealth(): Promise<{
  status: "healthy" | "degraded";
  components: Record<string, "up" | "down">;
}> {
  return request("/health");
}

// ── Query ─────────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/query — execute an Agentic RAG query (non-streaming).
 *
 * Returns the full answer in one JSON response.  Use `queryStream` for
 * the streaming / incremental-render experience.
 *
 * NOTE: the backend returns `latency_ms` (snake_case); we normalise it
 * to `latencyMs` here so the rest of the UI can use the camelCase type.
 */
export async function executeQuery(
  payload: QueryRequest,
): Promise<QueryResponse> {
  type RawResponse = Omit<QueryResponse, "latencyMs" | "rewriteCount"> & { latency_ms: number; rewrite_count: number };

  const raw = await request<RawResponse>("/api/v1/query", {
    method: "POST",
    body: JSON.stringify({
      query: payload.query,
      namespace: payload.namespace ?? "default",
      conversation_id: payload.conversationId ?? null,
      top_k: payload.topK ?? 5,
      use_cache: payload.useCache ?? true,
    }),
  });

  return {
    answer: raw.answer,
    sources: raw.sources,
    cached: raw.cached,
    latencyMs: raw.latency_ms,
    rewriteCount: raw.rewrite_count,
  };
}

// ── Streaming query ────────────────────────────────────────────────────────────

/** Metadata delivered in the SSE `done` event. */
export interface StreamDonePayload {
  sources: Source[];
  cached: boolean;
  latencyMs: number;
  rewriteCount: number;
}

/**
 * POST /api/v1/query/stream — execute an Agentic RAG query and receive
 * the answer incrementally via Server-Sent Events.
 *
 * @param query          - User's question text.
 * @param namespace      - Knowledge namespace (default: "default").
 * @param conversationId - Optional conversation context ID.
 * @param onChunk        - Called for each token/word chunk as it arrives.
 * @param onComplete     - Called once when the stream ends with full metadata.
 * @param onError        - Called if a network or API error occurs.
 * @param signal         - Optional AbortSignal to cancel the stream early.
 *
 * @example
 * const ctrl = new AbortController();
 * await queryStream("What is ToolRef?", "default", undefined,
 *   (chunk) => setContent(c => c + chunk),
 *   (meta)  => setMeta(meta),
 *   (err)   => console.error(err),
 *   ctrl.signal,
 * );
 */
export async function queryStream(
  query: string,
  namespace: string,
  conversationId: string | undefined,
  onChunk: (chunk: string) => void,
  onComplete: (meta: StreamDonePayload) => void,
  onError: (err: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/v1/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        namespace: namespace ?? "default",
        conversation_id: conversationId ?? null,
        top_k: 5,
        use_cache: true,
      }),
      signal,
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch { /* ignore */ }
    onError(new ApiError(response.status, detail));
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError(new Error("ReadableStream not supported"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by "\n\n"; each line is "data: <json>"
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? ""; // Keep any incomplete trailing event

      for (const part of parts) {
        for (const line of part.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;

          let event: Record<string, unknown>;
          try {
            event = JSON.parse(raw) as Record<string, unknown>;
          } catch {
            continue; // Skip malformed events
          }

          if (event.type === "chunk") {
            onChunk(event.content as string);
          } else if (event.type === "done") {
            // Map snake_case backend fields to camelCase frontend types
            const rawSources = (event.sources as Record<string, unknown>[]) ?? [];
            const mappedSources: Source[] = rawSources.map((s) => ({
              docTitle: (s.doc_title as string) ?? (s.docTitle as string) ?? "",
              chunkText: (s.chunk_text as string) ?? (s.chunkText as string) ?? "",
              url: (s.source_url as string) ?? (s.url as string) ?? "",
              score: (s.relevance_score as number) ?? (s.score as number) ?? 0,
            }));
            onComplete({
              sources: mappedSources,
              cached: Boolean(event.cached),
              latencyMs: (event.latency_ms as number) ?? 0,
              rewriteCount: (event.rewrite_count as number) ?? 0,
            });
          } else if (event.type === "error") {
            onError(new Error((event.message as string) ?? "Unknown stream error"));
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
  } finally {
    reader.releaseLock();
  }
}

// ── Documents ─────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/documents — upload a document for ingestion.
 *
 * @param file - File selected by the user.
 * @param namespace - Knowledge namespace.
 * @param title - Optional display title.
 */
export async function uploadDocument(
  file: File,
  namespace: string,
  title?: string,
): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("namespace", namespace);
  if (title) formData.append("title", title);

  const response = await fetch(`${API_BASE_URL}/api/v1/documents`, {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type; browser sets multipart boundary automatically.
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<UploadDocumentResponse>;
}

/**
 * GET /api/v1/documents — list documents in a namespace.
 *
 * @param namespace - Optional namespace filter.
 * @param page - Page number (1-based).
 * @param pageSize - Items per page.
 */
export async function listDocuments(
  namespace?: string,
  page = 1,
  pageSize = 20,
): Promise<{ total: number; items: DocumentSummary[] }> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (namespace) params.set("namespace", namespace);
  return request(`/api/v1/documents?${params}`);
}

/**
 * DELETE /api/v1/documents/:id — delete a document and all its data.
 *
 * @param docId - Document UUID.
 */
export async function deleteDocument(docId: string): Promise<{ detail: string }> {
  return request(`/api/v1/documents/${docId}`, { method: "DELETE" });
}
