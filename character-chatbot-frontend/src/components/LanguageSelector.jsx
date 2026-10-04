import { useLocale } from "../context/LocaleContext";

export default function LanguageSelector() {
  const { locale, setLocale, t } = useLocale();
  return (
    <div className="language-selector" role="group" aria-label={t("locale.label")}>
      <button type="button" className={locale === "en" ? "active" : ""} onClick={() => setLocale("en")}>EN</button>
      <span aria-hidden="true">|</span>
      <button type="button" className={locale === "ko" ? "active" : ""} onClick={() => setLocale("ko")}>한국어</button>
    </div>
  );
}
