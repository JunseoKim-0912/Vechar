import { useState } from "react";

const API = "http://localhost:4000";

export default function CharacterCreateForm({ token, onCreated }) {
  const [name, setName] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [status, setStatus] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setStatus("업로드 중...");

    let profileImageUrl;
    if (imageFile) {
      const form = new FormData();
      form.append("image", imageFile);
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      const data = await res.json();
      if (!res.ok) return setStatus(`실패: ${data.detail}`);
      profileImageUrl = data.url;
    }

    const res = await fetch(`${API}/characters/`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ name, profile_image_url: profileImageUrl }),
    });
    const data = await res.json();
    if (!res.ok) return setStatus(`실패: ${data.detail}`);
    setStatus("생성 완료!");
    onCreated?.(data);
  }

  return (
    <form onSubmit={handleSubmit}>
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="캐릭터 이름" required />
      <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
      <button type="submit">캐릭터 만들기</button>
      <p>{status}</p>
    </form>
  );
}