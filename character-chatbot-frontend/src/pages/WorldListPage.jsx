import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function WorldListPage() {
  const [worlds, setWorlds] = useState(null);
  const [error, setError] = useState("");
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const fileInputRef = useRef(null);

  async function load() {
    try {
      setWorlds(await api.get("/worlds/"));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate(e) {
    e.preventDefault();
    setCreating(true);
    try {
      await api.post("/worlds/", { name: newName });
      setNewName("");
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id) {
    if (!confirm("이 세계관을 삭제할까요? (연결된 캐릭터는 사라지지 않고 세계관만 없어집니다)")) return;
    try {
      await api.delete(`/worlds/${id}`);
      setWorlds((prev) => prev.filter((w) => w.id !== id));
    } catch (e) {
      alert(e.message);
    }
  }

  async function handleImportFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setImporting(true);
    setImportMessage("");
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      await api.post("/worlds/import", payload);
      setImportMessage("가져오기 완료!");
      await load();
    } catch (err) {
      setImportMessage(`실패: ${err.message}`);
    } finally {
      setImporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!worlds) return <p className="loading">불러오는 중...</p>;

  return (
    <div>
      <div className="page-header">
        <h1>세계관</h1>
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
        </div>
      </div>

      {importMessage && (
        <p className={importMessage.startsWith("실패") ? "form-error" : "form-success"}>{importMessage}</p>
      )}

      <form onSubmit={handleCreate} className="inline-form">
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="새 세계관 이름"
          required
        />
        <button type="submit" disabled={creating}>+ 만들기</button>
      </form>

      <div className="card-grid">
        {worlds.map((w) => (
          <div key={w.id} className="entity-card">
            <Link to={`/worlds/${w.id}`}>
              <h3>{w.name}</h3>
            </Link>
            {w.name !== "현실" && (
              <button className="link-danger" onClick={() => handleDelete(w.id)}>삭제</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}