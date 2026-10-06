import React from "react";

const ACTION = /<action>([\s\S]*?)<\/action>/gi;

function repairKnownActionWrappers(content) {
  let text = content
    .replace(/\[\s*Character\s+action\s*:\s*([\s\S]*?)\]/gi, "<action>$1</action>")
    .replace(/\[\s*Character\s+action\s*:\s*([^\]]+)$/gi, "<action>$1</action>")
    .replace(/<\s*action\s*>/gi, "<action>")
    .replace(/<\s*\/\s*action\s*>/gi, "</action>");
  for (let i = 0; i < 3; i += 1) {
    const repaired = text.replace(/<action>\s*<action>([\s\S]*?)<\/action>\s*<\/action>/gi, "<action>$1</action>");
    if (repaired === text) break;
    text = repaired;
  }
  text = text.replace(/<\/\s*action(?![\w>])/gi, "</action>");
  const openings = (text.match(/<action>/gi) || []).length;
  const closings = (text.match(/<\/action>/gi) || []).length;
  if (openings > closings && /[.!?…]\s*$/.test(text)) text += "</action>";
  return text;
}

export function parseActionSegments(content) {
  const repaired = repairKnownActionWrappers(content || "");
  const segments = [];
  let cursor = 0;
  for (const match of repaired.matchAll(ACTION)) {
    if (match.index > cursor) segments.push({ type: "dialogue", text: repaired.slice(cursor, match.index) });
    if (match[1].trim()) segments.push({ type: "action", text: match[1].trim() });
    cursor = match.index + match[0].length;
  }
  if (cursor < repaired.length || !segments.length) {
    // An unrecoverable partial marker is safe text, never executable markup.
    segments.push({ type: "dialogue", text: repaired.slice(cursor).replace(/<\/?\s*action\s*>?/gi, "") });
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
