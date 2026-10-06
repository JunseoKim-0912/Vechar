import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, apiAssetUrl } from "../api";
import { DualAvatar, nextRoomTurn, RoomMessage } from "../characterConversationUi";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";

export default function CharacterConversationPage() {
  const { roomId } = useParams();
  const { t } = useLocale();
  const [room, setRoom] = useState(null);
  const [messages, setMessages] = useState([]);
  const [nameDraft, setNameDraft] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const inflight = useRef(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    let active = true;
    Promise.all([
      api.get(`/character-conversations/${roomId}`),
      api.get(`/character-conversations/${roomId}/messages`),
    ]).then(([loadedRoom, loadedMessages]) => {
      if (!active) return;
      setRoom(loadedRoom);
      setNameDraft(loadedRoom.name);
      setMessages(loadedMessages);
    }).catch((failure) => { if (active) setError(localizeError(failure, t)); });
    return () => { active = false; };
  }, [roomId, t]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function refreshRoom() {
    const [loadedRoom, loadedMessages] = await Promise.all([
      api.get(`/character-conversations/${roomId}`),
      api.get(`/character-conversations/${roomId}/messages`),
    ]);
    setRoom(loadedRoom);
    setMessages(loadedMessages);
  }

  async function handleNext() {
    if (!room || inflight.current) return;
    inflight.current = true;
    setGenerating(true);
    setError("");
    try {
      const result = await nextRoomTurn(api, room);
      setRoom(result.room);
      setMessages((previous) => [...previous, result.message]);
    } catch (failure) {
      if (failure.status === 409) {
        try { await refreshRoom(); } catch { /* Preserve the original conflict message. */ }
        setError(t("characterConversations.turnConflict"));
      } else {
        setError(localizeError(failure, t));
      }
    } finally {
      inflight.current = false;
      setGenerating(false);
    }
  }

  async function handleRename(event) {
    event.preventDefault();
    if (!nameDraft.trim() || renaming) return;
    setRenaming(true);
    setError("");
    try {
      setRoom(await api.patch(`/character-conversations/${roomId}`, { name: nameDraft.trim() }));
    } catch (failure) {
      setError(localizeError(failure, t));
    } finally {
      setRenaming(false);
    }
  }

  if (!room && error) return <p className="form-error">{error}</p>;
  if (!room) return <p className="loading">{t("common.loading")}</p>;

  const speaker = room.participants.find((person) => person.id === room.next_speaker_character_id);
  return <div className="room-page">
    <Link to="/characters">← {t("characters.title")}</Link>
    <header className="room-header">
      <DualAvatar participants={room.participants} assetUrl={apiAssetUrl} />
      <div>
        <h1>{room.name}</h1>
        <p>{room.participants.map((person) => person.name).join(" · ")}</p>
        <span className="room-language">{room.language === "ko" ? t("locale.korean") : t("locale.english")}</span>
      </div>
    </header>
    <form className="room-rename" onSubmit={handleRename}>
      <label htmlFor="room-name">{t("characterConversations.rename")}</label>
      <input id="room-name" value={nameDraft} onChange={(event) => setNameDraft(event.target.value)} maxLength={120} />
      <button type="submit" disabled={renaming || !nameDraft.trim()}>{t("common.apply")}</button>
    </form>
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="room-messages" aria-live="polite">
      {messages.length === 0 && <p className="empty-state">{t("characterConversations.noMessages")}</p>}
      {messages.map((message) => {
        const participant = room.participants.find((person) => person.id === message.speaker_character_id);
        if (!participant) return null;
        return <RoomMessage key={message.id} message={message} participant={participant}
          position={participant.position} assetUrl={apiAssetUrl} />;
      })}
      <div ref={bottomRef} />
    </div>
    <div className="room-next-bar">
      <span>{t("characterConversations.nextSpeaker", { name: speaker?.name || t("chat.characterFallback") })}</span>
      <button className="btn-primary" onClick={handleNext} disabled={generating}>
        {generating ? t("characterConversations.generating") : t("characterConversations.next")}
      </button>
    </div>
  </div>;
}
