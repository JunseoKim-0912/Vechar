import { useState } from "react";
import { API_BASE_URL } from "./api";
import { useLocale } from "./context/LocaleContext";
import { localizeError } from "./i18n/errors";

export default function CharacterCreateForm({ token, onCreated }) {
  const { t } = useLocale();
  const [name, setName] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [status, setStatus] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setStatus(t("common.uploading"));

    let profileImageUrl;
    if (imageFile) {
      const form = new FormData();
      form.append("image", imageFile);
      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      const data = await res.json();
      if (!res.ok) {
        const error = { status: res.status, code: typeof data.detail === "object" ? data.detail?.code : undefined };
        return setStatus(`${t("common.failed")}: ${localizeError(error, t)}`);
      }
      profileImageUrl = data.url;
    }

    const res = await fetch(`${API_BASE_URL}/characters/`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ name, profile_image_url: profileImageUrl }),
    });
    const data = await res.json();
    if (!res.ok) {
      const error = { status: res.status, code: typeof data.detail === "object" ? data.detail?.code : undefined };
      return setStatus(`${t("common.failed")}: ${localizeError(error, t)}`);
    }
    setStatus(t("common.created"));
    onCreated?.(data);
  }

  return (
    <form onSubmit={handleSubmit}>
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t("characters.name")} required />
      <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
      <button type="submit">{t("characters.createTitle")}</button>
      <p>{status}</p>
    </form>
  );
}
