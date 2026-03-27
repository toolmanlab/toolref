/**
 * Shared TypeScript types for the ToolRef Chat UI.
 */

// ── Chat ──────────────────────────────────────────────────────────────────────

/** A single message in a conversation. */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  /** True if the answer was served from the semantic cache. */
  cached?: boolean;
  /** Round-trip latency in milliseconds. */
  latencyMs?: number;
  createdAt: Date;
}

/** A source citation returned with an assistant message. */
export interface Source {
  docTitle: string;
  chunkText: string;
  url?: string;
  score: number;
}

/** A conversation (list of messages under a shared ID). */
export interface Conversation {
  id: string;
  title: string;
  namespace: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// ── API ───────────────────────────────────────────────────────────────────────

/** POST /api/v1/query request body. */
export interface QueryRequest {
  query: string;
  namespace?: string;
  conversationId?: string;
  topK?: number;
  useCache?: boolean;
}

/** POST /api/v1/query response. */
export interface QueryResponse {
  answer: string;
  sources: Source[];
  cached: boolean;
  latencyMs: number;
  rewriteCount: number;
}

/** POST /api/v1/documents response. */
export interface UploadDocumentResponse {
  id: string;
  namespace: string;
  title: string;
  docType: string;
  status: "pending" | "processing" | "completed" | "failed";
  createdAt: string;
}

/** GET /api/v1/documents list item. */
export interface DocumentSummary {
  id: string;
  namespace: string;
  title: string;
  docType: string;
  status: "pending" | "processing" | "completed" | "failed";
  totalChunks: number;
  createdAt: string;
}

// ── UI state ──────────────────────────────────────────────────────────────────

export type LoadingState = "idle" | "loading" | "streaming" | "error";

export interface AppError {
  message: string;
  code?: string | number;
}
