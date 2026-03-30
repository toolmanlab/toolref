/**
 * MessageList — the main message stream for a conversation.
 *
 * Renders an ordered list of user / assistant messages, auto-scrolling
 * to the bottom whenever new messages are appended.  Optionally shows
 * a "thinking" skeleton when `isLoading` is true.
 *
 * @example
 * <MessageList messages={messages} isLoading={isStreaming} />
 */

import { useEffect, useRef } from "react";
import type { Message, Source } from "../types";

// ── Sub-components ────────────────────────────────────────────────────────────

interface SourceCardProps {
  source: Source;
  index: number;
}

/** Collapsible citation card shown beneath an assistant message. */
function SourceCard({ source, index }: SourceCardProps) {
  return (
    <details className="mt-1 rounded-lg border border-gray-700 bg-gray-900/60 px-3 py-2 text-xs text-gray-400">
      <summary className="cursor-pointer select-none hover:text-gray-200">
        [{index + 1}] {source.docTitle}{" "}
        <span className="text-gray-600">(score: {(source.score ?? 0).toFixed(2)})</span>
      </summary>
      <p className="mt-2 text-gray-400 leading-relaxed">{source.chunkText}</p>
    </details>
  );
}

interface MessageBubbleProps {
  message: Message;
}

/** A single chat bubble — user messages right-aligned, assistant left-aligned. */
function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] ${isUser ? "" : "w-full"}`}>
        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-brand-600 text-white ml-auto"
              : "bg-gray-800 text-gray-200"
          }`}
        >
          {message.content}
        </div>

        {/* Sources (assistant only) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            {message.sources.map((src, i) => (
              <SourceCard key={i} source={src} index={i} />
            ))}
          </div>
        )}

        {/* Metadata row (assistant only) */}
        {!isUser && (message.latencyMs !== undefined || message.cached) && (
          <p className="mt-1 text-xs text-gray-600">
            {message.latencyMs !== undefined && <span>{message.latencyMs} ms</span>}
            {message.cached && (
              <span className="ml-2 rounded bg-gray-700 px-1 py-0.5 text-gray-400">
                cached
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

/** Animated "…" skeleton shown while the assistant is generating. */
function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl bg-gray-800 px-4 py-3">
        <span className="inline-flex gap-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 animate-bounce rounded-full bg-gray-500"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </span>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface MessageListProps {
  /** Ordered list of messages to render. */
  messages: Message[];
  /**
   * When true, a "thinking" animation is appended after the last message.
   * Use while waiting for the assistant response.
   */
  isLoading?: boolean;
}

export default function MessageList({ messages, isLoading = false }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or loading state change.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col gap-4 px-4 py-6">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {isLoading && <ThinkingBubble />}

      {/* Scroll anchor */}
      <div ref={bottomRef} aria-hidden="true" />
    </div>
  );
}
