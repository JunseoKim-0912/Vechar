import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLocale } from "../context/LocaleContext";
import LanguageSelector from "../components/LanguageSelector";
import { localizeError } from "../i18n/errors";

export default function SignupPage() {
  const { signup } = useAuth();
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
      await signup(email, password);
      navigate("/characters");
    } catch (err) {
      setError(localizeError(err, t, "signup"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-language"><LanguageSelector /></div>
      <form className="auth-card" onSubmit={handleSubmit}>
        <h1>{t("auth.signup")}</h1>
        <label>
          {t("common.email")}
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          {t("auth.passwordHint")}
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />
        </label>
        {error && <p className="form-error">{error}</p>}
        <button type="submit" disabled={loading}>{loading ? t("auth.signingUp") : t("auth.signup")}</button>
        <p className="auth-switch">
          {t("auth.hasAccount")} <Link to="/login">{t("auth.login")}</Link>
        </p>
      </form>
    </div>
  );
}
