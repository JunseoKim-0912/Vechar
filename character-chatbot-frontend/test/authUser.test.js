import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { isAdminUser, normalizeCurrentUser } from "../src/authUser.js";
import { translate } from "../src/i18n/translations.js";

test("regular users do not receive the admin badge", () => {
  assert.equal(isAdminUser({ id: "1", email: "user@example.com", role: "user" }), false);
});

test("admins receive the localized admin badge", () => {
  const admin = normalizeCurrentUser({ id: "1", email: "admin@example.com", role: "admin" });
  assert.equal(isAdminUser(admin), true);
  assert.equal(translate("en", "nav.adminBadge"), "Admin");
  assert.equal(translate("ko", "nav.adminBadge"), "관리자");
});

test("missing or unknown roles safely default to a regular user", () => {
  assert.equal(normalizeCurrentUser({ id: "1", email: "legacy@example.com" }).role, "user");
  assert.equal(normalizeCurrentUser({ id: "2", email: "unknown@example.com", role: "owner" }).role, "user");
  assert.equal(isAdminUser({ id: "1", email: "legacy@example.com" }), false);
});

test("AuthContext restores the current user and Layout gates the badge on the DB-backed role", () => {
  const authContext = readFileSync(new URL("../src/context/AuthContext.jsx", import.meta.url), "utf8");
  const layout = readFileSync(new URL("../src/components/Layout.jsx", import.meta.url), "utf8");

  assert.match(authContext, /api\.get\("\/auth\/me"\)/);
  assert.match(authContext, /value = \{ token, user,/);
  assert.match(layout, /isAdminUser\(user\) &&/);
  assert.match(layout, /t\("nav\.adminBadge"\)/);
});
