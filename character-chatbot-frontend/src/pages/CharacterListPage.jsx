import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, apiAssetUrl } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";

export default function CharacterListPage() {
  const { t } = useLocale();
  const [characters, setCharacters] = useState(null);
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const [importFailed, setImportFailed] = useState(false);
  const fileInputRef = useRef(null);

  async function load() {
    try {
      setCharacters(await api.get("/characters/"));
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
      setCharacters((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      alert(localizeError(e, t));
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
    </div>
  );
}
