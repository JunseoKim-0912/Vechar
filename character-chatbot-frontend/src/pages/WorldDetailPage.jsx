import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, downloadJSON } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";

export default function WorldDetailPage() {
  const { t } = useLocale();
  const { id } = useParams();
  const navigate = useNavigate();
  const [world, setWorld] = useState(null);
  const [error, setError] = useState("");

  const [sourceType, setSourceType] = useState("NOVEL_EPISODE");
  const [seriesName, setSeriesName] = useState("");
  const [episodeNumber, setEpisodeNumber] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadFailed, setUploadFailed] = useState(false);

  const [editOp, setEditOp] = useState("add");
  const [editInstruction, setEditInstruction] = useState("");
  const [editing, setEditing] = useState(false);

  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");
  const [selectedCharacterName, setSelectedCharacterName] = useState("");
  const [extractingCharacter, setExtractingCharacter] = useState(false);

  async function load() {
    try {
      setWorld(await api.get(`/worlds/${id}`));
    } catch (e) {
      setError(localizeError(e, t));
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function handleUpload(e) {
    e.preventDefault();
    setUploading(true);
    setUploadMessage("");
    setUploadFailed(false);
    try {
      const form = new FormData();
      form.append("source_type", sourceType);
      if (sourceType === "NOVEL_EPISODE") {
        form.append("series_name", seriesName);
        form.append("episode_number", episodeNumber);
      }
      if (file) form.append("file", file);
      else form.append("text", text);

      await api.post(`/worlds/${id}/sources`, form, { isForm: true });
      setUploadMessage(t("common.trained"));
      setText("");
      setFile(null);
      await load();
    } catch (err) {
      setUploadFailed(true);
      setUploadMessage(`${t("common.failed")}: ${localizeError(err, t)}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleEdit(e) {
    e.preventDefault();
    setEditing(true);
    try {
      await api.post(`/worlds/${id}/edit`, { operation: editOp, instruction: editInstruction });
      setEditInstruction("");
      await load();
    } catch (e) {
      alert(localizeError(e, t));
    } finally {
      setEditing(false);
    }
  }

  async function handleSummary() {
    setSummaryLoading(true);
    try {
      const res = await api.get(`/worlds/${id}/summary`);
      setSummary(res.summary);
    } catch (e) {
      setSummary(`${t("common.error")}: ${localizeError(e, t)}`);
    } finally {
      setSummaryLoading(false);
    }
  }

  async function handleCompact() {
    if (!confirm(t("worlds.confirmCompact"))) return;
    try {
      await api.post(`/worlds/${id}/compact`);
      await load();
    } catch (e) {
      alert(localizeError(e, t));
    }
  }

  async function handleExtractCharacter(name) {
    if (extractingCharacter) return; // 이미 만드는 중이면 무시
    setExtractingCharacter(true);
    try {
      const character = await api.post(`/worlds/${id}/extract-character`, { name });
      navigate(`/characters/${character.id}`);
    } catch (e) {
      alert(localizeError(e, t));
    } finally {
      setExtractingCharacter(false);
    }
  }

  async function handleExport() {
    setExporting(true);
    setExportError("");
    try {
      const data = await api.get(`/worlds/${id}/export`);
      downloadJSON(data, `${world.name}_${t("worlds.exportFilenameSuffix")}.json`);
    } catch (e) {
      setExportError(localizeError(e, t));
    } finally {
      setExporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!world) return <p className="loading">{t("common.loading")}</p>;

  const profile = world.profile?.data;

  return (
    <div className="detail-page">
      <div className="page-header">
        <h1>{world.name}</h1>
        <button onClick={handleExport} disabled={exporting || !profile}>
          {exporting ? t("common.exporting") : t("common.export")}
        </button>
      </div>
      {exportError && <p className="form-error">{exportError}</p>}

      <section className="profile-summary">
        <h2>{t("worlds.knownInfo")}</h2>
        {profile ? (
          <>
            <p>{profile.world_summary || t("worlds.noSummary")}</p>

            {profile.key_facts?.length > 0 && (
              <ul>
                {profile.key_facts.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}

            {profile.mentioned_characters?.length > 0 && (
              <div className="suggested-characters">
                <h3>{t("worlds.discovered")}</h3>
                {profile.mentioned_characters.slice(0, 5).map((mc) => (
                  <button key={mc.name} onClick={() => handleExtractCharacter(mc.name)} disabled={extractingCharacter}>
                    {extractingCharacter ? t("common.creating") : t("characters.createFrom", { name: mc.name })}
                  </button>
                ))}
              </div>
            )}

            {profile.mentioned_characters?.length > 5 && (
              <div className="character-picker">
                <select
                  value={selectedCharacterName}
                  onChange={(e) => setSelectedCharacterName(e.target.value)}
                  disabled={extractingCharacter}
                >
                  <option value="">{t("worlds.chooseCharacter", { count: profile.mentioned_characters.length })}</option>
                  {profile.mentioned_characters.map((mc) => (
                    <option key={mc.name} value={mc.name}>
                      {mc.name}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => selectedCharacterName && handleExtractCharacter(selectedCharacterName)}
                  disabled={!selectedCharacterName || extractingCharacter}
                >
                  {extractingCharacter ? t("common.creating") : t("characters.createSelected")}
                </button>
              </div>
            )}

            <div className="world-actions">
              <button onClick={handleSummary} disabled={summaryLoading}>
                {summaryLoading ? t("common.loading") : t("worlds.viewSummary")}
              </button>
              <button onClick={handleCompact}>{t("worlds.compact")}</button>
            </div>
            {summary && <p className="world-summary-prose">{summary}</p>}
          </>
        ) : (
          <p className="empty-state">{t("worlds.empty")}</p>
        )}
      </section>

      <section>
        <h2>{t("worlds.uploadTitle")}</h2>
        <p className="form-help">{t("training.bilingualHelp")}</p>
        <form onSubmit={handleUpload} className="stacked-form">
          <label>
            {t("common.type")}
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="NOVEL_EPISODE">{t("worlds.episode")}</option>
              <option value="DESCRIPTION">{t("worlds.description")}</option>
            </select>
          </label>

          {sourceType === "NOVEL_EPISODE" && (
            <>
              <label>
                {t("worlds.seriesName")}
                <input value={seriesName} onChange={(e) => setSeriesName(e.target.value)} required />
              </label>
              <label>
                {t("worlds.episodeNumber")}
                <input
                  type="number"
                  min="1"
                  value={episodeNumber}
                  onChange={(e) => setEpisodeNumber(e.target.value)}
                  required
                />
              </label>
            </>
          )}

          <label>
            {t("training.file")}
            <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>

          <p className="or-divider">{t("common.or")}</p>

          <label>
            {t("training.text")}
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              maxLength={15000}
              disabled={Boolean(file)}
            />
          </label>

          {uploadMessage && (
            <p className={uploadFailed ? "form-error" : "form-success"}>{uploadMessage}</p>
          )}
          <button type="submit" disabled={uploading || (!file && !text.trim())}>
            {uploading ? t("common.training") : t("common.upload")}
          </button>
        </form>
      </section>

      <section>
        <h2>{t("worlds.editTitle")}</h2>
        <form onSubmit={handleEdit} className="stacked-form">
          <label>
            {t("worlds.operation")}
            <select value={editOp} onChange={(e) => setEditOp(e.target.value)}>
              <option value="add">{t("worlds.add")}</option>
              <option value="delete">{t("common.delete")}</option>
              <option value="modify">{t("worlds.modify")}</option>
            </select>
          </label>
          <label>
            {t("common.content")}
            <input value={editInstruction} onChange={(e) => setEditInstruction(e.target.value)} required />
          </label>
          <button type="submit" disabled={editing}>{editing ? t("common.applying") : t("common.apply")}</button>
        </form>
      </section>
    </div>
  );
}
