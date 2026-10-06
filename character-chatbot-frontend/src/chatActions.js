import React from "react";

const ACTION = /<action>([\s\S]*?)<\/action>/gi;

export function parseActionSegments(content) {
  const segments = [];
  let cursor = 0;
  for (const match of content.matchAll(ACTION)) {
    if (match.index > cursor) segments.push({ type: "dialogue", text: content.slice(cursor, match.index) });
    if (match[1].trim()) segments.push({ type: "action", text: match[1].trim() });
    cursor = match.index + match[0].length;
  }
  if (cursor < content.length || !segments.length) {
    // Incomplete markers are ordinary text until a complete pair arrives.
    segments.push({ type: "dialogue", text: content.slice(cursor).replace(/<\/?action>/gi, "") });
  }
  return segments.filter((segment) => segment.text);
}

export function ChatMessageContent({ content, role }) {
  if (role !== "CHARACTER") return React.createElement("p", null, content);
  return React.createElement(React.Fragment, null,
    ...parseActionSegments(content).map((segment, index) => React.createElement(
      segment.type === "action" ? "div" : "p",
      { key: index, className: segment.type === "action" ? "chat-action" : "chat-dialogue",
        ...(segment.type === "action" ? { role: "note" } : {}) },
      segment.text,
    )),
  );
}
