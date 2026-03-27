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
 */

import { useState } from "react";
import ChatInput from "../components/ChatInput";
import ConversationList from "../components/ConversationList";
import MessageList from "../components/MessageList";
import { executeQuery } from "../api/client";
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

  // Derived: active conversation
  const activeConversation = conversations.find((c) => c.id === activeId)!;
  const messages: Message[] = activeConversation?.messages ?? [];

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleNewConversation = () => {
    const conv = createConversation(namespace);
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  };

  const patchConversation = (id: string, patch: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: new Date() } : c)),
    );
  };

  const appendMessage = (convId: string, msg: Message) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, msg],
              // Use first user message as title
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

  const handleSend = async (text: string) => {
    if (loadingState === "loading") return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      createdAt: new Date(),
    };

    appendMessage(activeId, userMsg);
    setLoadingState("loading");

    try {
      // TODO: replace with streaming (SSE) in V1
      const result = await executeQuery({
        query: text,
        namespace: activeConversation.namespace,
        conversationId: activeId,
      });

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        sources: result.sources,
        cached: result.cached,
        latencyMs: result.latencyMs,
        createdAt: new Date(),
      };

      appendMessage(activeId, assistantMsg);
    } catch (err) {
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          err instanceof Error
            ? `Error: ${err.message}`
            : "An unexpected error occurred. Please try again.",
        createdAt: new Date(),
      };
      appendMessage(activeId, errMsg);
    } finally {
      setLoadingState("idle");
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* Sidebar */}
      <ConversationList
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
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
          <span className="ml-2 text-xs text-gray-600">
            / {activeConversation?.namespace ?? namespace}
          </span>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl">
            <MessageList messages={messages} isLoading={loadingState === "loading"} />
          </div>
        </main>

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          isDisabled={loadingState === "loading"}
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
