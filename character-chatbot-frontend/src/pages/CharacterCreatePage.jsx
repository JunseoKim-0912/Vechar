import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";

export default function CharacterCreatePage() {
  const { t } = useLocale();
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
      setError(localizeError(err, t));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="form-page">
      <h1>{t("characters.createTitle")}</h1>
      <form onSubmit={handleSubmit} className="stacked-form">
        <label>
          {t("characters.name")}
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={50} />
        </label>

        <label>
          {t("characters.image")}
          <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
        </label>

        <label>
          {t("characters.world")} ({t("characters.worldHint")})
          <select value={worldId} onChange={(e) => setWorldId(e.target.value)}>
            <option value="">{t("characters.reality")}</option>
            {worlds.map((w) => (
              <option key={w.id} value={w.id}>{w.name}</option>
            ))}
          </select>
        </label>

        {error && <p className="form-error">{error}</p>}
        <button type="submit" disabled={submitting}>
          {submitting ? t("common.creating") : t("common.create")}
        </button>
      </form>
    </div>
  );
}
