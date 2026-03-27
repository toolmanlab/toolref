/**
 * ChatInput — the message composition bar at the bottom of the chat.
 *
 * Supports multi-line input, submit on Enter (Shift+Enter for newline),
 * and a disabled state while the assistant is responding.
 *
 * @example
 * <ChatInput
 *   onSend={(text) => handleSend(text)}
 *   isDisabled={isLoading}
 *   namespace={activeNamespace}
 *   onNamespaceChange={setNamespace}
 * />
 */

import { useRef, useState } from "react";

interface ChatInputProps {
  /** Called when the user submits a message. */
  onSend: (text: string) => void;
  /** Disables the input while the assistant is generating. */
  isDisabled?: boolean;
  /** Currently active namespace. */
  namespace: string;
  /** Called when the user changes the namespace. */
  onNamespaceChange?: (ns: string) => void;
}

export default function ChatInput({
  onSend,
  isDisabled = false,
  namespace,
  onNamespaceChange,
}: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = text.trim().length > 0 && !isDisabled;

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || isDisabled) return;
    onSend(trimmed);
    setText("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  /** Auto-grow textarea up to ~8 lines. */
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  return (
    <div className="border-t border-gray-800 bg-gray-950 px-4 py-3">
      {/* Namespace selector */}
      {onNamespaceChange && (
        <div className="mx-auto mb-2 flex max-w-3xl items-center gap-2">
          <label htmlFor="namespace-input" className="text-xs text-gray-500">
            Namespace:
          </label>
          <input
            id="namespace-input"
            type="text"
            value={namespace}
            onChange={(e) => onNamespaceChange(e.target.value)}
            placeholder="default"
            className="rounded-md border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-gray-300 outline-none focus:border-brand-500"
          />
        </div>
      )}

      {/* Input row */}
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={isDisabled}
          placeholder={isDisabled ? "Generating…" : "Ask a question… (Enter to send)"}
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-700 bg-gray-900 px-4 py-3
                     text-sm text-gray-100 placeholder-gray-500 outline-none
                     focus:border-brand-500 focus:ring-1 focus:ring-brand-500
                     disabled:cursor-not-allowed disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={!canSend}
          className="shrink-0 rounded-xl bg-brand-600 px-4 py-3 text-sm font-medium text-white
                     transition-colors hover:bg-brand-700
                     disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Send message"
        >
          Send
        </button>
      </div>

      <p className="mx-auto mt-1 max-w-3xl text-right text-xs text-gray-700">
        Shift + Enter for new line
      </p>
    </div>
  );
}
