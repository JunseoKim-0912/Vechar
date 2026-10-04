import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";

export default function WorldListPage() {
  const { t } = useLocale();
  const [worlds, setWorlds] = useState(null);
  const [error, setError] = useState("");
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const [importFailed, setImportFailed] = useState(false);
  const fileInputRef = useRef(null);

  async function load() {
    try {
      setWorlds(await api.get("/worlds/"));
    } catch (e) {
      setError(localizeError(e, t));
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
      setError(localizeError(e, t));
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id) {
    if (!confirm(t("worlds.confirmDelete"))) return;
    try {
      await api.delete(`/worlds/${id}`);
      setWorlds((prev) => prev.filter((w) => w.id !== id));
    } catch (e) {
      alert(localizeError(e, t));
    }
  }

  async function handleImportFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setImporting(true);
    setImportMessage("");
    setImportFailed(false);
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      await api.post("/worlds/import", payload);
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
  if (!worlds) return <p className="loading">{t("common.loading")}</p>;

  return (
    <div>
      <div className="page-header">
        <h1>{t("worlds.title")}</h1>
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
        </div>
      </div>

      {importMessage && (
        <p className={importFailed ? "form-error" : "form-success"}>{importMessage}</p>
      )}

      <form onSubmit={handleCreate} className="inline-form">
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder={t("worlds.newName")}
          required
        />
        <button type="submit" disabled={creating}>+ {creating ? t("common.creating") : t("common.create")}</button>
      </form>

      <div className="card-grid">
        {worlds.map((w) => (
          <div key={w.id} className="entity-card">
            <Link to={`/worlds/${w.id}`}>
              <h3>{w.name}</h3>
            </Link>
            {w.name !== "현실" && (
              <button className="link-danger" onClick={() => handleDelete(w.id)}>{t("common.delete")}</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
