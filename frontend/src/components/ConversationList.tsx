/**
 * ConversationList — sidebar panel showing past conversations.
 *
 * Renders a vertical list of conversation summaries.  Clicking an
 * item fires `onSelect`; the "New Chat" button fires `onNew`.
 *
 * @example
 * <ConversationList
 *   conversations={conversations}
 *   activeId={activeConversationId}
 *   onSelect={setActiveConversationId}
 *   onNew={handleNewConversation}
 * />
 */

import type { Conversation } from "../types";

interface ConversationListProps {
  /** All conversations to display. */
  conversations: Conversation[];
  /** ID of the currently active conversation (highlighted). */
  activeId: string | null;
  /** Called when the user clicks a conversation. */
  onSelect: (id: string) => void;
  /** Called when the user clicks "New Chat". */
  onNew: () => void;
}

/** Format a Date as a short relative label (e.g. "Today", "Yesterday", or a date string). */
function formatDate(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const oneDayMs = 86_400_000;

  if (diff < oneDayMs) return "Today";
  if (diff < 2 * oneDayMs) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
}: ConversationListProps) {
  return (
    <aside className="flex h-full w-64 flex-col border-r border-gray-800 bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <span className="text-sm font-semibold text-gray-300">Conversations</span>
        <button
          onClick={onNew}
          title="New chat"
          className="rounded-md p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white"
          aria-label="New chat"
        >
          {/* Plus icon */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="h-5 w-5"
          >
            <path d="M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z" />
          </svg>
        </button>
      </div>

      {/* List */}
      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        {conversations.length === 0 ? (
          <p className="px-2 py-4 text-center text-xs text-gray-500">
            No conversations yet.
            <br />
            Start by asking a question!
          </p>
        ) : (
          <ul className="space-y-1">
            {conversations.map((conv) => (
              <li key={conv.id}>
                <button
                  onClick={() => onSelect(conv.id)}
                  className={`w-full rounded-lg px-3 py-2 text-left transition-colors ${
                    conv.id === activeId
                      ? "bg-gray-700 text-white"
                      : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                  }`}
                >
                  <p className="truncate text-sm font-medium">{conv.title}</p>
                  <p className="mt-0.5 flex items-center gap-1 text-xs text-gray-500">
                    <span className="truncate">{conv.namespace}</span>
                    <span>·</span>
                    <span>{formatDate(conv.updatedAt)}</span>
                  </p>
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-800 px-4 py-3">
        <p className="text-xs text-gray-600">ToolRef — Agentic RAG</p>
      </div>
    </aside>
  );
}
