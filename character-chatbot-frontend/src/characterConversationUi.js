import React from "react";
import { ChatMessageContent } from "./chatActions.js";

export function roomCreatePayload(firstId, secondId, name, language) {
  if (!firstId || !secondId) throw new Error("roomParticipantsRequired");
  if (firstId === secondId) throw new Error("roomParticipantsDistinct");
  if (!name.trim()) throw new Error("roomNameRequired");
  return { character_a_id: firstId, character_b_id: secondId, name: name.trim(), language };
}

export function nextRoomTurn(apiClient, room) {
  return apiClient.post(`/character-conversations/${room.id}/next`, {
    expected_turn_index: room.turn_index,
  });
}

export function DualAvatar({ participants, assetUrl = (url) => url }) {
  return React.createElement("div", { className: "dual-avatar", "aria-label": participants.map((p) => p.name).join(" and ") },
    ...participants.map((participant) => React.createElement(
      "span", { className: "dual-avatar-person", key: participant.id },
      participant.profile_image_url
        ? React.createElement("img", { src: assetUrl(participant.profile_image_url), alt: participant.name })
        : React.createElement("span", { className: "dual-avatar-initial", "aria-hidden": "true" },
          participant.name?.[0] || "?"),
    )),
  );
}

export function RoomMessage({ message, participant, position, assetUrl = (url) => url }) {
  return React.createElement("div", { className: `room-message room-message-${position}` },
    React.createElement("div", { className: "room-message-speaker" },
      React.createElement(DualAvatar, { participants: [participant], assetUrl }),
      React.createElement("span", null, participant.name),
    ),
    React.createElement("div", { className: "chat-bubble chat-character" },
      React.createElement(ChatMessageContent, { role: "CHARACTER", content: message.content }),
    ),
  );
}
