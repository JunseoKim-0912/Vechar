import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, downloadJSON } from "../api";
import { resumeCharacterChat } from "../chatSessions";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";
import { trainingRequestBody } from "../trainingSource";
import { startTrainingJobPolling, trainingProgress } from "../trainingJobs";

export default function CharacterDetailPage() {
  const { t } = useLocale();
  const { id } = useParams();
  const navigate = useNavigate();
  const [character, setCharacter] = useState(null);
  const [error, setError] = useState("");

  const [sourceType, setSourceType] = useState("STORY");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const fileInput = useRef(null);
  const [training, setTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState("");
  const [trainFailed, setTrainFailed] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobProgress, setJobProgress] = useState(null);

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");

  async function loadCharacter() {
    try {
      setCharacter(await api.get(`/characters/${id}`));
    } catch (e) {
      setError(localizeError(e, t));
    }
  }

  useEffect(() => {
    loadCharacter();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (!jobId) return undefined;
    return startTrainingJobPolling({
      api, jobId,
      onUpdate: (job) => {
        setTrainMessage(trainingProgress(job, t));
        setJobProgress(job.progress);
      },
      onComplete: () => {
        setTraining(false);
        setJobId(null);
        setTrainMessage(t("common.trained"));
        void loadCharacter();
      },
      onFailure: (job) => {
        setTraining(false);
        setJobId(null);
        setTrainFailed(true);
        setTrainMessage(`${t("training.failed")}: ${localizeError({ code: job.error_code }, t, "training")}`);
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, id, t]);

  async function handleTrain(e) {
    e.preventDefault();
    setTraining(true);
    setTrainMessage("");
    setTrainFailed(false);
    setJobProgress(null);
    try {
      const body = await trainingRequestBody({ sourceType, text, file });
      const job = await api.post(`/characters/${id}/training-sources`, body);
      setJobId(job.job_id);
      setTrainMessage(t("training.preparing"));
      setText("");
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    } catch (err) {
      setTraining(false);
      setTrainFailed(true);
      setTrainMessage(`${t("common.failed")}: ${localizeError(err, t, "training")}`);
    }
  }

  async function startChat() {
    try {
      const conversation = await resumeCharacterChat(api, id);
      navigate(`/chat/${id}`, { state: { conversationId: conversation.id } });
    } catch (e) {
      alert(localizeError(e, t));
    }
  }

  async function handleExport() {
    setExporting(true);
    setExportError("");
    try {
      const data = await api.get(`/characters/${id}/export`);
      downloadJSON(data, `${character.name}_${t("characters.exportFilenameSuffix")}.json`);
    } catch (e) {
      setExportError(localizeError(e, t));
    } finally {
      setExporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!character) return <p className="loading">{t("common.loading")}</p>;

  const profile = character.profile?.data;

  return (
    <div className="detail-page">
      <div className="page-header">
        <h1>{character.name}</h1>
        <div className="page-header-actions">
          <button onClick={handleExport} disabled={exporting || !profile}>
            {exporting ? t("common.exporting") : t("common.export")}
          </button>
          <button className="btn-primary" onClick={startChat}>{t("characters.startChat")}</button>
        </div>
      </div>
      {exportError && <p className="form-error">{exportError}</p>}
      {!profile && <p className="empty-state">{t("characters.profileRequired")}</p>}

      <section className="profile-summary">
        <h2>{t("characters.profile")}</h2>
        {profile ? (
          <dl>
            <dt>{t("characters.personality")}</dt>
            <dd>{profile.personality_summary || "-"}</dd>
            <dt>{t("characters.speechStyle")}</dt>
            <dd>{profile.speech_style || "-"}</dd>
            <dt>{t("characters.background")}</dt>
            <dd>{profile.background_facts?.join(" · ") || "-"}</dd>
          </dl>
        ) : (
          <p className="empty-state">{t("characters.noTraining")}</p>
        )}
      </section>

      <section>
        <h2>{t("characters.train")}</h2>
        <p className="form-help">{t("training.bilingualHelp")}</p>
        <form onSubmit={handleTrain} className="stacked-form">
          <label>
            {t("common.type")}
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="STORY">{t("characters.sourceStory")}</option>
              <option value="DIALOGUE">{t("characters.sourceDialogue")}</option>
              <option value="MANUAL_DESCRIPTION">{t("characters.sourceManual")}</option>
            </select>
          </label>

          <label>
            {t("training.file")}
            <input ref={fileInput} type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>

          <p className="or-divider">{t("common.or")}</p>

          <label>
            {t("training.text")}
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              disabled={Boolean(file)}
            />
          </label>

          {trainMessage && (
            <p className={trainFailed ? "form-error" : "form-success"}>{trainMessage}</p>
          )}
          {jobProgress !== null && <progress value={jobProgress} max="100" aria-label={t("training.progress")} />}
          <button type="submit" disabled={training || (!file && !text.trim())}>
            {training ? t("common.training") : t("characters.train")}
          </button>
        </form>
      </section>
    </div>
  );
}
