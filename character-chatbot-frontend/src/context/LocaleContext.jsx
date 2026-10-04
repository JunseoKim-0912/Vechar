import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { SUPPORTED_LOCALES, translate } from "../i18n/translations";
import { getStoredLocale, persistLocale } from "../i18n/locale";

const LocaleContext = createContext(null);

export function LocaleProvider({ children }) {
  const [locale, setLocaleState] = useState(() => getStoredLocale(localStorage));

  useEffect(() => {
    document.documentElement.lang = locale;
    persistLocale(localStorage, locale);
  }, [locale]);

  const setLocale = useCallback((nextLocale) => {
    if (SUPPORTED_LOCALES.includes(nextLocale)) setLocaleState(nextLocale);
  }, []);
  const t = useCallback((key, values) => translate(locale, key, values), [locale]);
  const value = useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

// Kept beside the provider so the locale API stays as small as the app's other context.
// eslint-disable-next-line react-refresh/only-export-components
export function useLocale() {
  const context = useContext(LocaleContext);
  if (!context) throw new Error("useLocale must be used within LocaleProvider");
  return context;
}
