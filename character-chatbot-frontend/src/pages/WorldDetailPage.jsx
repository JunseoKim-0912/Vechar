import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, downloadJSON } from "../api";

export default function WorldDetailPage() {
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
      setError(e.message);
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
      setUploadMessage("학습 완료!");
      setText("");
      setFile(null);
      await load();
    } catch (err) {
      setUploadMessage(`실패: ${err.message}`);
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
      alert(e.message);
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
      setSummary(`오류: ${e.message}`);
    } finally {
      setSummaryLoading(false);
    }
  }

  async function handleCompact() {
    if (!confirm("세계관 정보를 압축할까요? (중복 정리, 필요하면 되돌릴 수 있어요)")) return;
    try {
      await api.post(`/worlds/${id}/compact`);
      await load();
    } catch (e) {
      alert(e.message);
    }
  }

  async function handleExtractCharacter(name) {
    if (extractingCharacter) return; // 이미 만드는 중이면 무시
    setExtractingCharacter(true);
    try {
      const character = await api.post(`/worlds/${id}/extract-character`, { name });
      navigate(`/characters/${character.id}`);
    } catch (e) {
      alert(e.message);
    } finally {
      setExtractingCharacter(false);
    }
  }

  async function handleExport() {
    setExporting(true);
    setExportError("");
    try {
      const data = await api.get(`/worlds/${id}/export`);
      downloadJSON(data, `${world.name}_세계관.json`);
    } catch (e) {
      setExportError(e.message);
    } finally {
      setExporting(false);
    }
  }

  if (error) return <p className="form-error">{error}</p>;
  if (!world) return <p className="loading">불러오는 중...</p>;

  const profile = world.profile?.data;

  return (
    <div className="detail-page">
      <div className="page-header">
        <h1>{world.name}</h1>
        <button onClick={handleExport} disabled={exporting || !profile}>
          {exporting ? "내보내는 중..." : "내보내기"}
        </button>
      </div>
      {exportError && <p className="form-error">{exportError}</p>}

      <section className="profile-summary">
        <h2>알고 있는 정보</h2>
        {profile ? (
          <>
            <p>{profile.world_summary || "아직 요약이 없어요."}</p>

            {profile.key_facts?.length > 0 && (
              <ul>
                {profile.key_facts.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}

            {profile.mentioned_characters?.length > 0 && (
              <div className="suggested-characters">
                <h3>이 세계관에서 발견된 인물</h3>
                {profile.mentioned_characters.slice(0, 5).map((mc) => (
                  <button key={mc.name} onClick={() => handleExtractCharacter(mc.name)} disabled={extractingCharacter}>
                    {extractingCharacter ? "만드는 중..." : `${mc.name}(으)로 캐릭터 만들기`}
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
                  <option value="">전체 목록에서 고르기 ({profile.mentioned_characters.length}명)</option>
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
                  {extractingCharacter ? "만드는 중..." : "선택한 인물로 만들기"}
                </button>
              </div>
            )}

            <div className="world-actions">
              <button onClick={handleSummary} disabled={summaryLoading}>
                {summaryLoading ? "불러오는 중..." : "자연어로 요약 보기"}
              </button>
              <button onClick={handleCompact}>압축하기</button>
            </div>
            {summary && <p className="world-summary-prose">{summary}</p>}
          </>
        ) : (
          <p className="empty-state">아직 학습된 내용이 없어요.</p>
        )}
      </section>

      <section>
        <h2>화/설명 업로드</h2>
        <form onSubmit={handleUpload} className="stacked-form">
          <label>
            종류
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="NOVEL_EPISODE">소설 화</option>
              <option value="DESCRIPTION">세계관 설명</option>
            </select>
          </label>

          {sourceType === "NOVEL_EPISODE" && (
            <>
              <label>
                시리즈 이름
                <input value={seriesName} onChange={(e) => setSeriesName(e.target.value)} required />
              </label>
              <label>
                화 번호
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

          {uploadMessage && (
            <p className={uploadMessage.startsWith("실패") ? "form-error" : "form-success"}>{uploadMessage}</p>
          )}
          <button type="submit" disabled={uploading || (!file && !text.trim())}>
            {uploading ? "학습 중..." : "업로드"}
          </button>
        </form>
      </section>

      <section>
        <h2>수정하기</h2>
        <form onSubmit={handleEdit} className="stacked-form">
          <label>
            작업
            <select value={editOp} onChange={(e) => setEditOp(e.target.value)}>
              <option value="add">추가</option>
              <option value="delete">삭제</option>
              <option value="modify">수정</option>
            </select>
          </label>
          <label>
            내용
            <input value={editInstruction} onChange={(e) => setEditInstruction(e.target.value)} required />
          </label>
          <button type="submit" disabled={editing}>{editing ? "적용 중..." : "적용하기"}</button>
        </form>
      </section>
    </div>
  );
}