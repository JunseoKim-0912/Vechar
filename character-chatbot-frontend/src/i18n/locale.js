import { DEFAULT_LOCALE, SUPPORTED_LOCALES } from "./translations.js";

export const LOCALE_STORAGE_KEY = "vechar_locale";

export function getStoredLocale(storage) {
  const stored = storage.getItem(LOCALE_STORAGE_KEY);
  return SUPPORTED_LOCALES.includes(stored) ? stored : DEFAULT_LOCALE;
}

export function persistLocale(storage, locale) {
  if (SUPPORTED_LOCALES.includes(locale)) storage.setItem(LOCALE_STORAGE_KEY, locale);
}
