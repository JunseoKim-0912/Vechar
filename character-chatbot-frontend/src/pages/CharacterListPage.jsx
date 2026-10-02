import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, apiAssetUrl } from "../api";

export default function CharacterListPage() {
  const [characters, setCharacters] = useState(null);
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const fileInputRef = useRef(null);

  async function load() {
    try {
      setCharacters(await api.get("/characters/"));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(id) {
    if (!confirm("이 캐릭터를 삭제할까요?")) return;
    try {
      await api.delete(`/characters/${id}`);
      setCharacters((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      alert(e.message);
    }
  }

  async function handleImportFile(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // 같은 파일을 다시 골라도 onChange가 또 발생하도록 초기화
    if (!file) return;

    setImporting(true);
    setImportMessage("");
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      await api.post("/characters/import", payload);
      setImportMessage("가져오기 완료!");
      await load();
    } catch (err) {
      setImportMessage(`실패: ${err.message}`);
    } finally {
      setImporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!characters) return <p className="loading">불러오는 중...</p>;

  return (
    <div>
      <div className="page-header">
        <h1>내 캐릭터</h1>
        <div className="page-header-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json"
            onChange={handleImportFile}
            style={{ display: "none" }}
          />
          <button onClick={() => fileInputRef.current?.click()} disabled={importing}>
            {importing ? "가져오는 중..." : "가져오기"}
          </button>
          <Link to="/characters/new" className="btn-primary">+ 새 캐릭터</Link>
        </div>
      </div>

      {importMessage && (
        <p className={importMessage.startsWith("실패") ? "form-error" : "form-success"}>{importMessage}</p>
      )}

      {characters.length === 0 && (
        <p className="empty-state">아직 만든 캐릭터가 없어요. 첫 캐릭터를 만들거나 파일을 가져와보세요.</p>
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
              <Link to={`/characters/${c.id}`}>자세히</Link>
              <button className="link-danger" onClick={() => handleDelete(c.id)}>삭제</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
