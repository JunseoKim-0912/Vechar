import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { api } from "../src/api.js";
import { MAX_SOURCE_CHARS, trainingRequestBody } from "../src/trainingSource.js";
import { localizeError } from "../src/i18n/errors.js";
import { translate } from "../src/i18n/translations.js";

async function capturePost(path, body) {
  const previousFetch = globalThis.fetch;
  const previousStorage = globalThis.localStorage;
  let captured;
  globalThis.localStorage = { getItem: () => "test-token" };
  globalThis.fetch = async (url, options) => {
    captured = { url, ...options };
    return { ok: true, status: 201, text: async () => "{}" };
  };
  try {
    await api.post(path, body);
    return captured;
  } finally {
    globalThis.fetch = previousFetch;
    globalThis.localStorage = previousStorage;
  }
}

test("selected character and world .txt files reach multipart requests without JSON Content-Type", async () => {
  for (const [path, sourceType] of [
    ["/characters/1/training-sources", "STORY"],
    ["/worlds/1/sources", "DESCRIPTION"],
  ]) {
    const file = new File(["Training text"], "source.txt", { type: "text/plain" });
    const body = await trainingRequestBody({ sourceType, text: "ignored text", file });
    const request = await capturePost(path, body);
    assert.equal(request.body.get("file").name, "source.txt");
    assert.equal(await request.body.get("file").text(), "Training text");
    assert.equal(request.body.get("text"), null);
    assert.equal(request.headers.Authorization, "Bearer test-token");
    assert.equal(request.headers["Content-Type"], undefined);
  }
});

test("direct text continues as authenticated JSON with the full 300,000-character value", async () => {
  const text = "가".repeat(MAX_SOURCE_CHARS);
  const body = await trainingRequestBody({ sourceType: "DESCRIPTION", text });
  const request = await capturePost("/worlds/1/sources", body);
  assert.equal(request.headers["Content-Type"], "application/json");
  assert.equal(request.headers.Authorization, "Bearer test-token");
  assert.equal(JSON.parse(request.body).text, text);
  // JS textarea maxLength counts UTF-16 units, so validation uses code points instead.
  const emoji = "😀".repeat(MAX_SOURCE_CHARS);
  const emojiBody = await trainingRequestBody({ sourceType: "STORY", text: emoji });
  assert.equal(emojiBody.text, emoji);
});

test("client validation catches missing, non-txt, invalid UTF-8, empty, and 300,001 characters", async () => {
  const cases = [
    [{ sourceType: "STORY", text: "" }, "training_source_missing"],
    [{ sourceType: "STORY", text: "x".repeat(300_001) }, "source_too_long"],
    [{ sourceType: "STORY", file: new File([], "empty.txt") }, "empty_training_file"],
    [{ sourceType: "STORY", file: new File(["text"], "wrong.pdf") }, "invalid_source_file_type"],
    [{ sourceType: "STORY", file: new File([new Uint8Array([255])], "bad.txt") }, "source_invalid_utf8"],
    [{ sourceType: "STORY", file: new File(["x".repeat(300_001)], "large.txt") }, "source_too_long"],
  ];
  for (const [source, code] of cases) {
    await assert.rejects(trainingRequestBody(source), (error) => error.code === code);
  }
});

test("both training pages show the 300,000-character limit and localized upload errors", () => {
  assert.equal(translate("en", "training.text"), "Enter text directly (max 300,000 characters)");
  assert.equal(translate("ko", "training.text"), "텍스트 직접 입력 (최대 300,000자)");
  for (const page of ["CharacterDetailPage.jsx", "WorldDetailPage.jsx"]) {
    const source = readFileSync(new URL(`../src/pages/${page}`, import.meta.url), "utf8");
    assert.match(source, /trainingRequestBody\(/);
  }
  const ko = (key) => translate("ko", key);
  assert.equal(localizeError({ status: 400 }, ko, "training"), ko("errors.malformedTrainingRequest"));
  assert.equal(localizeError({ code: "source_invalid_utf8" }, ko, "training"), ko("errors.sourceInvalidUtf8"));
  assert.equal(localizeError({ status: 500 }, ko, "training"), ko("errors.server"));
});
