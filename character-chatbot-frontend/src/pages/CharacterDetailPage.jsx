import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, downloadJSON } from "../api";

export default function CharacterDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [character, setCharacter] = useState(null);
  const [error, setError] = useState("");

  const [sourceType, setSourceType] = useState("STORY");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState("");

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");

  async function loadCharacter() {
    try {
      setCharacter(await api.get(`/characters/${id}`));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    loadCharacter();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function handleTrain(e) {
    e.preventDefault();
    setTraining(true);
    setTrainMessage("");
    try {
      const form = new FormData();
      form.append("source_type", sourceType);
      if (file) form.append("file", file);
      else form.append("text", text);

      await api.post(`/characters/${id}/training-sources`, form, { isForm: true });
      setTrainMessage("학습 완료!");
      setText("");
      setFile(null);
      await loadCharacter();
    } catch (err) {
      setTrainMessage(`실패: ${err.message}`);
    } finally {
      setTraining(false);
    }
  }

  async function startChat() {
    try {
      const conversation = await api.post("/chat/conversations", { character_id: id });
      navigate(`/chat/${id}`, { state: { conversationId: conversation.id } });
    } catch (e) {
      alert(e.message);
    }
  }

  async function handleExport() {
    setExporting(true);
    setExportError("");
    try {
      const data = await api.get(`/characters/${id}/export`);
      downloadJSON(data, `${character.name}_캐릭터.json`);
    } catch (e) {
      setExportError(e.message);
    } finally {
      setExporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!character) return <p className="loading">불러오는 중...</p>;

  const profile = character.profile?.data;

  return (
    <div className="detail-page">
      <div className="page-header">
        <h1>{character.name}</h1>
        <div className="page-header-actions">
          <button onClick={handleExport} disabled={exporting || !profile}>
            {exporting ? "내보내는 중..." : "내보내기"}
          </button>
          <button className="btn-primary" onClick={startChat}>대화 시작하기</button>
        </div>
      </div>
      {exportError && <p className="form-error">{exportError}</p>}
      {!profile && <p className="empty-state">학습된 프로필이 있어야 내보낼 수 있어요.</p>}

      <section className="profile-summary">
        <h2>학습된 프로필</h2>
        {profile ? (
          <dl>
            <dt>성격</dt>
            <dd>{profile.personality_summary || "-"}</dd>
            <dt>말투</dt>
            <dd>{profile.speech_style || "-"}</dd>
            <dt>배경</dt>
            <dd>{profile.background_facts?.join(" · ") || "-"}</dd>
          </dl>
        ) : (
          <p className="empty-state">아직 학습된 내용이 없어요. 아래에서 텍스트를 추가해보세요.</p>
        )}
      </section>

      <section>
        <h2>학습시키기</h2>
        <form onSubmit={handleTrain} className="stacked-form">
          <label>
            종류
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="STORY">단편 소설</option>
              <option value="DIALOGUE">대화 기록</option>
              <option value="MANUAL_DESCRIPTION">직접 설명</option>
            </select>
          </label>

          <label>
            파일 업로드 (.txt)
            <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>

          <p className="or-divider">또는</p>

          <label>
            텍스트 직접 입력 (최대 15,000자)
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              maxLength={15000}
              disabled={Boolean(file)}
            />
          </label>

          {trainMessage && (
            <p className={trainMessage.startsWith("실패") ? "form-error" : "form-success"}>{trainMessage}</p>
          )}
          <button type="submit" disabled={training || (!file && !text.trim())}>
            {training ? "학습 중..." : "학습시키기"}
          </button>
        </form>
      </section>
    </div>
  );
}