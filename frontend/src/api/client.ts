/**
 * ToolRef API client.
 *
 * Thin wrapper around fetch providing typed request/response
 * helpers for each backend endpoint.  All methods are async and
 * throw `ApiError` on non-2xx responses.
 */

import type {
  DocumentSummary,
  QueryRequest,
  QueryResponse,
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
 * POST /api/v1/query — execute an Agentic RAG query.
 *
 * @param payload - Query parameters.
 * @returns The assistant answer, sources, and metadata.
 *
 * @example
 * const res = await executeQuery({ query: "What is ToolRef?", namespace: "demo" });
 * console.log(res.answer);
 */
export async function executeQuery(
  payload: QueryRequest,
): Promise<QueryResponse> {
  // TODO: replace with streaming (SSE / WebSocket) in V1
  return request<QueryResponse>("/api/v1/query", {
    method: "POST",
    body: JSON.stringify({
      query: payload.query,
      namespace: payload.namespace ?? "default",
      conversation_id: payload.conversationId ?? null,
      top_k: payload.topK ?? 5,
      use_cache: payload.useCache ?? true,
    }),
  });
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
