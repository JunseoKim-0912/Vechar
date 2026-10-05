import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { startTrainingJobPolling, trainingProgress } from "../src/trainingJobs.js";
import { translate } from "../src/i18n/translations.js";

const t = (key, values) => translate("en", key, values);

test("progress is localized for queue, extraction, synthesis and failure", () => {
  assert.match(trainingProgress({ status: "queued" }, t), /Preparing/);
  assert.match(trainingProgress({ status: "extracting", completed_chunks: 2, total_chunks: 7 }, t), /2 \/ 7/);
  assert.match(trainingProgress({ status: "synthesizing" }, t), /Combining/);
  assert.match(trainingProgress({ status: "failed" }, t), /failed/);
  assert.match(trainingProgress({ status: "queued" }, (key, values) => translate("ko", key, values)), /준비/);
});

test("polling stops on completion and reports job id", async () => {
  const calls = [];
  let done;
  const finished = new Promise((resolve) => { done = resolve; });
  const states = ["extracting", "completed"];
  const stop = startTrainingJobPolling({
    api: { get: async (path) => { calls.push(path); return { status: states.shift() }; } },
    jobId: "job-123", intervalMs: 1, onUpdate: () => {},
    onComplete: done, onFailure: () => assert.fail("must not fail"),
  });
  await finished;
  stop();
  assert.deepEqual(calls, ["/training-jobs/job-123", "/training-jobs/job-123"]);
});

test("transient network error retries; failure and cleanup stop polling", async () => {
  let calls = 0;
  let failed;
  const finished = new Promise((resolve) => { failed = resolve; });
  const stop = startTrainingJobPolling({
    api: { get: async () => { calls += 1; if (calls === 1) throw Error("offline"); return { status: "failed", error_code: "daily_limit_reached" }; } },
    jobId: "job", intervalMs: 1, onUpdate: () => {},
    onComplete: () => assert.fail("must not complete"), onFailure: failed,
  });
  const job = await finished;
  stop();
  assert.equal(calls, 2);
  assert.equal(job.error_code, "daily_limit_reached");
});

test("both training forms wire 202 job IDs to polling, refresh, and disabled submit", () => {
  for (const [page, pending, refresh] of [
    ["CharacterDetailPage.jsx", "training", "loadCharacter"],
    ["WorldDetailPage.jsx", "uploading", "load"],
  ]) {
    const source = readFileSync(new URL(`../src/pages/${page}`, import.meta.url), "utf8");
    assert.match(source, /setJobId\(job\.job_id\)/);
    assert.match(source, /startTrainingJobPolling\(/);
    assert.match(source, new RegExp(`void ${refresh}\\(\\)`));
    assert.match(source, new RegExp(`disabled=\\{${pending} \\|\\|`));
  }
});
