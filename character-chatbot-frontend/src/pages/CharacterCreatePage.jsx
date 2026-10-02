import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";

export default function CharacterCreatePage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [worlds, setWorlds] = useState([]);
  const [worldId, setWorldId] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.get("/worlds/").then(setWorlds).catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      let profileImageUrl;
      if (imageFile) {
        const form = new FormData();
        form.append("image", imageFile);
        const uploaded = await api.post("/upload", form, { isForm: true });
        profileImageUrl = uploaded.url;
      }

      const character = await api.post("/characters/", {
        name,
        profile_image_url: profileImageUrl,
        world_id: worldId || undefined,
      });

      navigate(`/characters/${character.id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="form-page">
      <h1>새 캐릭터 만들기</h1>
      <form onSubmit={handleSubmit} className="stacked-form">
        <label>
          이름
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={50} />
        </label>

        <label>
          프로필 사진 (선택)
          <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
        </label>

        <label>
          세계관 (선택 — 비워두면 "현실"에 배정됩니다)
          <select value={worldId} onChange={(e) => setWorldId(e.target.value)}>
            <option value="">현실 (기본)</option>
            {worlds.map((w) => (
              <option key={w.id} value={w.id}>{w.name}</option>
            ))}
          </select>
        </label>

        {error && <p className="form-error">{error}</p>}
        <button type="submit" disabled={submitting}>
          {submitting ? "만드는 중..." : "만들기"}
        </button>
      </form>
    </div>
  );
}