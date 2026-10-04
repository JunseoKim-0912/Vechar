import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLocale } from "../context/LocaleContext";
import LanguageSelector from "../components/LanguageSelector";
import { localizeError } from "../i18n/errors";

export default function LoginPage() {
  const { login } = useAuth();
  const { t } = useLocale();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      navigate("/characters");
    } catch (err) {
      setError(localizeError(err, t, "login"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-language"><LanguageSelector /></div>
      <form className="auth-card" onSubmit={handleSubmit}>
        <h1>{t("auth.login")}</h1>
        <label>
          {t("common.email")}
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          {t("common.password")}
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        {error && <p className="form-error">{error}</p>}
        <button type="submit" disabled={loading}>{loading ? t("auth.loggingIn") : t("auth.login")}</button>
        <p className="auth-switch">
          {t("auth.noAccount")} <Link to="/signup">{t("auth.signup")}</Link>
        </p>
      </form>
    </div>
  );
}
