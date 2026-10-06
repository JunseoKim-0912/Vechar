import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { DualAvatar, nextRoomTurn, RoomMessage, roomCreatePayload } from "../src/characterConversationUi.js";
import { resumeCharacterChat } from "../src/chatSessions.js";

const source = (path) => readFileSync(new URL(path, import.meta.url), "utf8");

test("room create requires distinct characters, custom name, and explicit language", () => {
  assert.deepEqual(roomCreatePayload("a", "b", "  A Meeting  ", "ko"), {
    character_a_id: "a", character_b_id: "b", name: "A Meeting", language: "ko",
  });
  assert.throws(() => roomCreatePayload("a", "a", "Meeting", "en"), /roomParticipantsDistinct/);
  assert.throws(() => roomCreatePayload("a", "b", "   ", "en"), /roomNameRequired/);
  assert.throws(() => roomCreatePayload("", "b", "Meeting", "en"), /roomParticipantsRequired/);
});

test("dual avatar renders two independent portraits or initial fallbacks", () => {
  const html = renderToStaticMarkup(React.createElement(DualAvatar, { participants: [
    { id: "a", name: "Ari", profile_image_url: "/ari.png" },
    { id: "b", name: "Bex", profile_image_url: null },
  ], assetUrl: (url) => `https://example.invalid${url}` }));
  assert.match(html, /src="https:\/\/example.invalid\/ari.png"/);
  assert.match(html, /dual-avatar-initial/);
  assert.match(html, />B<\/span>/);
});

test("room messages reuse semantic action renderer", () => {
  const html = renderToStaticMarkup(React.createElement(RoomMessage, {
    message: { content: "Hello. <action>nods</action>" },
    participant: { id: "a", name: "Ari", position: 0 }, position: 0,
  }));
  assert.match(html, /room-message-0/);
  assert.match(html, /Ari/);
  assert.match(html, /class="chat-action"/);
});

test("each Next posts one expected server turn; user chat asks backend to resume", async () => {
  const calls = [];
  const api = { post: async (...args) => { calls.push(args); return { ok: true }; } };
  await nextRoomTurn(api, { id: "room", turn_index: 8 });
  await resumeCharacterChat(api, "character");
  assert.deepEqual(calls, [
    ["/character-conversations/room/next", { expected_turn_index: 8 }],
    ["/chat/conversations", { character_id: "character", reuse_existing: true }],
  ]);
});

test("room pages wire list/create/continue/delete and durable reload/loading", () => {
  const list = source("../src/pages/CharacterListPage.jsx");
  const room = source("../src/pages/CharacterConversationPage.jsx");
  const detail = source("../src/pages/CharacterDetailPage.jsx");
  const chat = source("../src/pages/ChatPage.jsx");
  assert.match(list, /characterConversations\.title/);
  assert.match(list, /roomCreatePayload\(firstId, secondId, roomName, roomLanguage\)/);
  assert.match(list, /api\.post\("\/character-conversations", payload\)/);
  assert.match(list, /api\.delete\(`\/character-conversations\/\$\{roomId\}`\)/);
  assert.match(list, /characterConversations\.continue/);
  assert.match(room, /api\.get\(`\/character-conversations\/\$\{roomId\}\/messages`\)/);
  assert.match(room, /inflight\.current/);
  assert.match(room, /disabled=\{generating\}/);
  assert.match(room, /nextRoomTurn\(api, room\)/);
  assert.match(room, /characterConversations\.generationFailed/);
  assert.match(room, /await refreshRoom\(\)/);
  assert.match(detail, /resumeCharacterChat/);
  assert.match(chat, /resumeCharacterChat/);
});
