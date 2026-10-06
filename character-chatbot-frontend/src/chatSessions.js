/** Resolve a durable user-character session. The backend owns get-or-create. */
export function resumeCharacterChat(apiClient, characterId) {
  return apiClient.post("/chat/conversations", {
    character_id: characterId,
    reuse_existing: true,
  });
}
