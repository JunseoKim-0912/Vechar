import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ChatMessageContent, parseActionSegments } from "../src/chatActions.js";

test("dialogue-only and multiple action blocks retain order and newlines", () => {
  assert.deepEqual(parseActionSegments("Hello."), [{ type: "dialogue", text: "Hello." }]);
  assert.deepEqual(parseActionSegments("Hi <action>nods,\nslowly.</action> Yes <action>smiles!</action>"), [
    { type: "dialogue", text: "Hi " },
    { type: "action", text: "nods,\nslowly." },
    { type: "dialogue", text: " Yes " },
    { type: "action", text: "smiles!" },
  ]);
});

test("malformed markers fall back safely and user-authored markers remain literal", () => {
  assert.deepEqual(parseActionSegments("Hello <action>unfinished"), [
    { type: "dialogue", text: "Hello unfinished" },
  ]);
  const user = renderToStaticMarkup(React.createElement(ChatMessageContent,
    { role: "USER", content: "<action>literal</action>" }));
  assert.match(user, /&lt;action&gt;literal&lt;\/action&gt;/);
  assert.doesNotMatch(user, /chat-action/);
});

test("assistant actions render as accessible, escaped semantic blocks", () => {
  const html = renderToStaticMarkup(React.createElement(ChatMessageContent, {
    role: "CHARACTER", content: "Hello. <action>lowers head\n<script>alert(1)</script></action> Goodbye.",
  }));
  assert.match(html, /class="chat-action" role="note"/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>/);
  assert.doesNotMatch(html, /&lt;action&gt;/);
  assert.match(html, /class="chat-dialogue"/);
  const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
  assert.match(css, /\.chat-bubble \.chat-action/);
  assert.match(css, /overflow-wrap: anywhere/);
  assert.match(css, /@media \(max-width: 640px\)/);
});
