export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE || "http://localhost:4000"
).replace(/\/+$/, "");

export function apiAssetUrl(url) {
  return url?.startsWith("/uploads/") ? `${API_BASE_URL}${url}` : url;
}

export function getToken() {
  return localStorage.getItem("token");
}

export function setToken(token) {
  if (token) localStorage.setItem("token", token);
  else localStorage.removeItem("token");
}

async function request(path, { method = "GET", body, isForm = false, auth = true } = {}) {
  const headers = {};
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  let fetchBody = body;
  if (body !== undefined && !isForm) {
    headers["Content-Type"] = "application/json";
    fetchBody = JSON.stringify(body);
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { method, headers, body: fetchBody });

  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!res.ok) {
    const detail = data?.detail;
    const error = new Error(typeof detail === "string" ? detail : `Request failed (${res.status})`);
    error.status = res.status;
    error.code = typeof detail === "object" ? detail?.code : undefined;
    throw error;
  }
  return data;
}

export const api = {
  get: (path) => request(path),
  post: (path, body, opts = {}) => request(path, { method: "POST", body, ...opts }),
  delete: (path) => request(path, { method: "DELETE" }),
};

// 내보내기 결과(JSON)를 브라우저에서 파일로 다운로드시킵니다.
export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
