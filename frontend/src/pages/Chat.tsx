/**
 * Chat page — main view for the ToolRef conversation interface.
 *
 * Layout:
 *   ┌──────────────┬─────────────────────────────┐
 *   │              │         Header               │
 *   │ Conversation ├─────────────────────────────┤
 *   │    List      │       MessageList            │
 *   │   (sidebar)  │   (scrollable message area)  │
 *   │              ├─────────────────────────────┤
 *   │              │         ChatInput            │
 *   └──────────────┴─────────────────────────────┘
 *
 * Streaming flow:
 *   1. User submits → append user message, set state="loading"
 *   2. First SSE chunk arrives → append streaming assistant placeholder,
 *      set state="streaming"
 *   3. Each subsequent chunk → append text to the placeholder message
 *   4. `done` event → attach sources/metadata, set state="idle"
 *   5. Error → fill placeholder with error text, set state="error"
 *
 * The AbortController stored in `abortRef` lets us cancel an in-flight
 * stream when the user starts a new conversation or sends another message.
 */

import { useRef, useState } from "react";
import ChatInput from "../components/ChatInput";
import ConversationList from "../components/ConversationList";
import MessageList from "../components/MessageList";
import { queryStream } from "../api/client";
import type { Conversation, LoadingState, Message } from "../types";

// ── Helpers ───────────────────────────────────────────────────────────────────

function createConversation(namespace: string): Conversation {
  const now = new Date();
  return {
    id: crypto.randomUUID(),
    title: "New Conversation",
    namespace,
    messages: [
      {
        id: "welcome",
        role: "assistant",
        content:
          "Welcome to ToolRef! I'm your Agentic RAG assistant. Ask me anything about your knowledge base.",
        createdAt: now,
      },
    ],
    createdAt: now,
    updatedAt: now,
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Chat() {
  const [namespace, setNamespace] = useState("default");
  const [conversations, setConversations] = useState<Conversation[]>(() => [
    createConversation("default"),
  ]);
  const [activeId, setActiveId] = useState<string>(conversations[0].id);
  const [loadingState, setLoadingState] = useState<LoadingState>("idle");

  /** Ref holding the AbortController for any in-flight stream. */
  const abortRef = useRef<AbortController | null>(null);

  // Derived: active conversation
  const activeConversation = conversations.find((c) => c.id === activeId)!;
  const messages: Message[] = activeConversation?.messages ?? [];

  // ── Conversation helpers ──────────────────────────────────────────────────

  const handleNewConversation = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    const conv = createConversation(namespace);
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setLoadingState("idle");
  };

  const handleSelectConversation = (id: string) => {
    abortRef.current?.abort();
    abortRef.current = null;
    setActiveId(id);
    setLoadingState("idle");
  };

  const patchConversation = (id: string, patch: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: new Date() } : c)),
    );
  };

  /** Append a new message to a conversation. */
  const appendMessage = (convId: string, msg: Message) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, msg],
              // Use first user message as conversation title
              title:
                c.title === "New Conversation" && msg.role === "user"
                  ? msg.content.slice(0, 60)
                  : c.title,
              updatedAt: new Date(),
            }
          : c,
      ),
    );
  };

  /** Update a specific message (by id) inside a conversation. */
  const patchMessage = (convId: string, msgId: string, patch: Partial<Message>) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.id === msgId ? { ...m, ...patch } : m,
              ),
              updatedAt: new Date(),
            }
          : c,
      ),
    );
  };

  /** Append text to a streaming message's content. */
  const appendChunkToMessage = (convId: string, msgId: string, chunk: string) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.id === msgId ? { ...m, content: m.content + chunk } : m,
              ),
            }
          : c,
      ),
    );
  };

  // ── Send handler ──────────────────────────────────────────────────────────

  const handleSend = async (text: string) => {
    // Prevent sending while already generating
    if (loadingState !== "idle" && loadingState !== "error") return;

    // Cancel any prior in-flight request
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Snapshot the active conversation at the moment of submission to avoid
    // stale-closure issues when callbacks fire asynchronously.
    const convId = activeId;
    const convNamespace = activeConversation.namespace;

    // ── 1. Append user message ─────────────────────────────────────────
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      createdAt: new Date(),
    };
    appendMessage(convId, userMsg);
    setLoadingState("loading");

    // ── 2. Reserve a message ID for the assistant reply ────────────────
    //   The actual message is added on the first chunk so the ThinkingBubble
    //   is shown while the LangGraph pipeline is running.
    const assistantMsgId = crypto.randomUUID();
    let firstChunk = true;

    await queryStream(
      text,
      convNamespace,
      convId,

      // onChunk ─────────────────────────────────────────────────────────
      (chunk) => {
        if (firstChunk) {
          firstChunk = false;
          // Add the placeholder on the first token so ThinkingBubble hides
          const placeholder: Message = {
            id: assistantMsgId,
            role: "assistant",
            content: chunk,
            createdAt: new Date(),
          };
          appendMessage(convId, placeholder);
          setLoadingState("streaming");
        } else {
          appendChunkToMessage(convId, assistantMsgId, chunk);
        }
      },

      // onComplete ──────────────────────────────────────────────────────
      (meta) => {
        if (firstChunk) {
          // Edge case: done event arrived without any chunks (empty answer)
          firstChunk = false;
          appendMessage(convId, {
            id: assistantMsgId,
            role: "assistant",
            content: "",
            sources: meta.sources,
            cached: meta.cached,
            latencyMs: meta.latencyMs,
            createdAt: new Date(),
          });
        } else {
          patchMessage(convId, assistantMsgId, {
            sources: meta.sources,
            cached: meta.cached,
            latencyMs: meta.latencyMs,
          });
        }
        setLoadingState("idle");
        abortRef.current = null;
      },

      // onError ─────────────────────────────────────────────────────────
      (err) => {
        const errorContent = `Error: ${err.message}`;
        if (firstChunk) {
          firstChunk = false;
          appendMessage(convId, {
            id: assistantMsgId,
            role: "assistant",
            content: errorContent,
            createdAt: new Date(),
          });
        } else {
          patchMessage(convId, assistantMsgId, {
            content: errorContent,
          });
        }
        setLoadingState("error");
        abortRef.current = null;
      },

      ctrl.signal,
    );
  };

  // ── Render ────────────────────────────────────────────────────────────────

  const isBusy = loadingState === "loading" || loadingState === "streaming";

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* Sidebar */}
      <ConversationList
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
      />

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center gap-3 border-b border-gray-800 px-6 py-3">
          <span className="text-xl" aria-hidden="true">🔍</span>
          <h1 className="text-lg font-semibold text-gray-100">ToolRef</h1>
          <span className="rounded-full bg-brand-600/20 px-2 py-0.5 text-xs text-brand-500">
            Agentic RAG
          </span>
          {/* Namespace badge */}
          <span className="ml-2 text-xs text-gray-600">
            / {activeConversation?.namespace ?? namespace}
          </span>
          {/* Streaming indicator */}
          {loadingState === "streaming" && (
            <span className="ml-auto text-xs text-brand-500 animate-pulse">
              ● streaming
            </span>
          )}
          {loadingState === "loading" && (
            <span className="ml-auto text-xs text-gray-500 animate-pulse">
              ● thinking…
            </span>
          )}
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl">
            <MessageList
              messages={messages}
              isLoading={loadingState === "loading"}
            />
          </div>
        </main>

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          isDisabled={isBusy}
          namespace={namespace}
          onNamespaceChange={(ns) => {
            setNamespace(ns);
            patchConversation(activeId, { namespace: ns });
          }}
        />
      </div>
    </div>
  );
}
