import { useEffect, useRef, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import { api } from "../api";

export default function ChatPage() {
  const { characterId } = useParams();
  const location = useLocation();
  const [character, setCharacter] = useState(null);
  const [conversationId, setConversationId] = useState(location.state?.conversationId || null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    api.get(`/characters/${characterId}`).then(setCharacter).catch((e) => setError(e.message));
  }, [characterId]);

  useEffect(() => {
    async function ensureConversation() {
      if (conversationId) {
        try {
          const existing = await api.get(`/chat/conversations/${conversationId}/messages`);
          setMessages(existing.map((m) => ({ role: m.role, content: m.content })));
        } catch {
          setConversationId(null); // 이전 대화를 못 찾으면 새로 시작
        }
        return;
      }
      const conv = await api.post("/chat/conversations", { character_id: characterId });
      setConversationId(conv.id);
    }
    ensureConversation().catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [characterId, conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || !conversationId || sending) return;
    const userMsg = { role: "USER", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);
    try {
      const data = await api.post(`/chat/conversations/${conversationId}/messages`, { content: userMsg.content });
      setMessages((prev) => [...prev, { role: data.role, content: data.content }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "SYSTEM_NOTE", content: `오류: ${err.message}` }]);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.nativeEvent.isComposing) send();
  }

  if (error) return <p className="form-error">{error}</p>;

  return (
    <div className="chat-page">
      <div className="chat-header">
        <Link to={`/characters/${characterId}`}>← {character?.name || "캐릭터"}</Link>
      </div>

      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble chat-${m.role.toLowerCase()}`}>
            {m.role !== "SYSTEM_NOTE" && (
              <span className="chat-sender">{m.role === "USER" ? "나" : character?.name}</span>
            )}
            <p>{m.content}</p>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`${character?.name || ""}에게 말 걸기... (/수정 으로 정정 가능)`}
          disabled={sending}
        />
        <button onClick={send} disabled={sending}>보내기</button>
      </div>
    </div>
  );
}