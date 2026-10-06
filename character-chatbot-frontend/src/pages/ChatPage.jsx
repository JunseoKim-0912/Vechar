import { useEffect, useRef, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import { api } from "../api";
import { useLocale } from "../context/LocaleContext";
import { localizeError } from "../i18n/errors";
import { ChatMessageContent } from "../chatActions";
import { resumeCharacterChat } from "../chatSessions";

export default function ChatPage() {
  const { locale, t } = useLocale();
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
    api.get(`/characters/${characterId}`).then(setCharacter).catch((e) => setError(localizeError(e, t)));
  }, [characterId, t]);

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
      const conv = await resumeCharacterChat(api, characterId);
      setConversationId(conv.id);
    }
    ensureConversation().catch((e) => setError(localizeError(e, t)));
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
      const data = await api.post(`/chat/conversations/${conversationId}/messages`, { content: userMsg.content, locale });
      setMessages((prev) => [...prev, { role: data.role, content: data.content }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "SYSTEM_NOTE", content: `${t("common.error")}: ${localizeError(err, t)}` }]);
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
        <Link to={`/characters/${characterId}`}>← {character?.name || t("chat.characterFallback")}</Link>
      </div>

      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble chat-${m.role.toLowerCase()}`}>
            {m.role !== "SYSTEM_NOTE" && (
              <span className="chat-sender">{m.role === "USER" ? t("chat.me") : character?.name}</span>
            )}
            <ChatMessageContent role={m.role} content={m.content} />
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t("chat.placeholder", { name: character?.name || t("chat.characterFallback") })}
          disabled={sending}
        />
        <button onClick={send} disabled={sending}>{t("chat.send")}</button>
      </div>
    </div>
  );
}
