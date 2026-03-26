import { useState } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Welcome to ToolRef! I'm your Agentic RAG assistant. Ask me anything about your knowledge base.",
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    // TODO: connect to backend /api/v1/query
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gray-950">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-gray-800 px-6 py-3">
        <span className="text-xl">🔍</span>
        <h1 className="text-lg font-semibold text-gray-100">ToolRef</h1>
        <span className="rounded-full bg-brand-600/20 px-2 py-0.5 text-xs text-brand-500">
          Agentic RAG
        </span>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-brand-600 text-white"
                    : "bg-gray-800 text-gray-200"
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* Input */}
      <footer className="border-t border-gray-800 px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-gray-100 placeholder-gray-500 outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="rounded-xl bg-brand-600 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </footer>
    </div>
  );
}
