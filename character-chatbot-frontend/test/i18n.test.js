import test from "node:test";
import assert from "node:assert/strict";

import { translations, translate } from "../src/i18n/translations.js";
import { getStoredLocale, persistLocale } from "../src/i18n/locale.js";
import { localizeError } from "../src/i18n/errors.js";

function keys(value, prefix = "") {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof child === "string" ? [path] : keys(child, path);
  });
}

test("English and Korean dictionaries have identical keys", () => {
  assert.deepEqual(keys(translations.en).sort(), keys(translations.ko).sort());
});

test("first load defaults to English and valid selections persist", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  assert.equal(getStoredLocale(storage), "en");
  persistLocale(storage, "ko");
  assert.equal(getStoredLocale(storage), "ko");
  persistLocale(storage, "en");
  assert.equal(getStoredLocale(storage), "en");
});

test("translations render both locales and interpolate values", () => {
  assert.equal(translate("en", "auth.login"), "Log in");
  assert.equal(translate("ko", "auth.login"), "로그인");
  assert.equal(translate("en", "chat.placeholder", { name: "Mina" }), "Message Mina... (use /수정 to correct the profile)");
});

test("common API failures use localized stable mappings", () => {
  const en = (key, values) => translate("en", key, values);
  const ko = (key, values) => translate("ko", key, values);
  assert.equal(localizeError({ code: "daily_limit_reached" }, en), translations.en.errors.dailyLimit);
  assert.equal(localizeError({ code: "monthly_limit_reached" }, ko), translations.ko.errors.monthlyLimit);
  assert.equal(localizeError({ status: 401 }, en), translations.en.errors.unauthorized);
  assert.equal(localizeError({ status: 403 }, ko), translations.ko.errors.forbidden);
  assert.equal(localizeError({ status: 404 }, ko), translations.ko.errors.notFound);
  assert.equal(localizeError({ status: 500 }, en), translations.en.errors.server);
});
