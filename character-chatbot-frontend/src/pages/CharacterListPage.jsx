import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, apiAssetUrl } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";
import { DualAvatar, roomCreatePayload } from "../characterConversationUi";

export default function CharacterListPage() {
  const { locale, t } = useLocale();
  const [characters, setCharacters] = useState(null);
  const [rooms, setRooms] = useState([]);
  const [showRoomForm, setShowRoomForm] = useState(false);
  const [firstId, setFirstId] = useState("");
  const [secondId, setSecondId] = useState("");
  const [roomName, setRoomName] = useState("");
  const [roomLanguage, setRoomLanguage] = useState(locale);
  const [creatingRoom, setCreatingRoom] = useState(false);
  const [roomError, setRoomError] = useState("");
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const [importFailed, setImportFailed] = useState(false);
  const fileInputRef = useRef(null);

  async function load() {
    try {
      const [ownedCharacters, ownedRooms] = await Promise.all([
        api.get("/characters/"), api.get("/character-conversations"),
      ]);
      setCharacters(ownedCharacters);
      setRooms(ownedRooms);
    } catch (e) {
      setError(localizeError(e, t));
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(id) {
    if (!confirm(t("characters.confirmDelete"))) return;
    try {
      await api.delete(`/characters/${id}`);
      await load();
    } catch (e) {
      alert(localizeError(e, t));
    }
  }

  async function handleCreateRoom(event) {
    event.preventDefault();
    setRoomError("");
    let payload;
    try {
      payload = roomCreatePayload(firstId, secondId, roomName, roomLanguage);
    } catch (error) {
      setRoomError(t(`characterConversations.${error.message}`));
      return;
    }
    setCreatingRoom(true);
    try {
      const room = await api.post("/character-conversations", payload);
      setRooms((previous) => [room, ...previous]);
      setShowRoomForm(false);
      setFirstId(""); setSecondId(""); setRoomName("");
    } catch (error) {
      setRoomError(localizeError(error, t));
    } finally {
      setCreatingRoom(false);
    }
  }

  async function handleDeleteRoom(roomId) {
    if (!confirm(t("characterConversations.confirmDelete"))) return;
    try {
      await api.delete(`/character-conversations/${roomId}`);
      setRooms((previous) => previous.filter((room) => room.id !== roomId));
    } catch (error) {
      setRoomError(localizeError(error, t));
    }
  }

  async function handleImportFile(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // 같은 파일을 다시 골라도 onChange가 또 발생하도록 초기화
    if (!file) return;

    setImporting(true);
    setImportMessage("");
    setImportFailed(false);
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      await api.post("/characters/import", payload);
      setImportMessage(t("common.importComplete"));
      await load();
    } catch (err) {
      setImportFailed(true);
      setImportMessage(`${t("common.failed")}: ${localizeError(err, t)}`);
    } finally {
      setImporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!characters) return <p className="loading">{t("common.loading")}</p>;

  return (
    <div>
      <div className="page-header">
        <h1>{t("characters.title")}</h1>
        <div className="page-header-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json"
            onChange={handleImportFile}
            style={{ display: "none" }}
          />
          <button onClick={() => fileInputRef.current?.click()} disabled={importing}>
            {importing ? t("common.importing") : t("common.import")}
          </button>
          <Link to="/characters/new" className="btn-primary">{t("characters.new")}</Link>
        </div>
      </div>

      {importMessage && (
        <p className={importFailed ? "form-error" : "form-success"}>{importMessage}</p>
      )}

      {characters.length === 0 && (
        <p className="empty-state">{t("characters.empty")}</p>
      )}

      <div className="card-grid">
        {characters.map((c) => (
          <div key={c.id} className="entity-card">
            {c.profile_image_url ? (
              <img src={apiAssetUrl(c.profile_image_url)} alt={c.name} className="avatar" />
            ) : (
              <div className="avatar avatar-placeholder">{c.name[0]}</div>
            )}
            <h3>{c.name}</h3>
            <div className="entity-card-actions">
              <Link to={`/characters/${c.id}`}>{t("characters.details")}</Link>
              <button className="link-danger" onClick={() => handleDelete(c.id)}>{t("common.delete")}</button>
            </div>
          </div>
        ))}
      </div>

      <section className="conversation-section" aria-labelledby="character-conversations-title">
        <div className="page-header">
          <h2 id="character-conversations-title">{t("characterConversations.title")}</h2>
          <button type="button" onClick={() => {
            setRoomLanguage(locale);
            setShowRoomForm((shown) => !shown);
          }} disabled={characters.length < 2}>{t("characterConversations.new")}</button>
        </div>
        {roomError && <p className="form-error" role="alert">{roomError}</p>}
        {showRoomForm && (
          <form className="room-create-form" onSubmit={handleCreateRoom}>
            <label>{t("characterConversations.first")}
              <select value={firstId} onChange={(event) => setFirstId(event.target.value)} required>
                <option value="">{t("characterConversations.choose")}</option>
                {characters.map((character) => <option value={character.id} key={character.id}>{character.name}</option>)}
              </select>
            </label>
            <label>{t("characterConversations.second")}
              <select value={secondId} onChange={(event) => setSecondId(event.target.value)} required>
                <option value="">{t("characterConversations.choose")}</option>
                {characters.filter((character) => character.id !== firstId).map((character) =>
                  <option value={character.id} key={character.id}>{character.name}</option>)}
              </select>
            </label>
            <label>{t("characterConversations.name")}
              <input value={roomName} onChange={(event) => setRoomName(event.target.value)} maxLength={120} required />
            </label>
            <label>{t("characterConversations.language")}
              <select value={roomLanguage} onChange={(event) => setRoomLanguage(event.target.value)}>
                <option value="en">{t("locale.english")}</option>
                <option value="ko">{t("locale.korean")}</option>
              </select>
            </label>
            <button className="btn-primary" type="submit" disabled={creatingRoom}>
              {creatingRoom ? t("common.creating") : t("common.create")}
            </button>
          </form>
        )}
        {rooms.length === 0 && <p className="empty-state">{t("characterConversations.empty")}</p>}
        <div className="room-card-grid">
          {rooms.map((room) => <div className="room-card" key={room.id}>
            <DualAvatar participants={room.participants} assetUrl={apiAssetUrl} />
            <div className="room-card-copy">
              <h3>{room.name}</h3>
              <p>{room.participants.map((person) => person.name).join(" · ")}</p>
            </div>
            <div className="entity-card-actions">
              <Link to={`/character-conversations/${room.id}`}>{t("characterConversations.continue")}</Link>
              <button className="link-danger" onClick={() => handleDeleteRoom(room.id)}>{t("common.delete")}</button>
            </div>
          </div>)}
        </div>
      </section>
    </div>
  );
}
