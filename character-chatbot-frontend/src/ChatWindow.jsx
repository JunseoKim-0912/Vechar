import { useState } from "react";

const API = "http://localhost:4000";

export default function ChatWindow({ token, conversationId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function send() {
    if (!input.trim()) return;
    const userMsg = { role: "USER", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    const res = await fetch(`${API}/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ content: userMsg.content }),
    });
    const data = await res.json();
    setMessages((prev) => [...prev, { role: data.role, content: data.content }]);
  }

  function handleKeyDown(e) {
    // 한글 등 조합형 입력(IME) 중에 Enter를 누르면 keydown이 두 번 발생할 수 있음.
    // isComposing이 true인 동안(조합 중)은 무시하고, 조합이 끝난 뒤의 Enter만 전송으로 처리.
    if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      send();
    }
  }

  return (
    <div>
      <div>
        {messages.map((m, i) => (
          <p key={i} style={{ fontStyle: m.role === "SYSTEM_NOTE" ? "italic" : "normal" }}>
            <strong>{m.role === "USER" ? "나" : m.role === "SYSTEM_NOTE" ? "시스템" : "캐릭터"}:</strong>{" "}
            {m.content}
          </p>
        ))}
      </div>
      <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} />
      <button onClick={send}>보내기</button>
    </div>
  );
}