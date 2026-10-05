export function trainingProgress(job, t) {
  if (!job) return "";
  if (job.status === "queued" || job.status === "chunking") return t("training.preparing");
  if (job.status === "extracting") {
    return t("training.extracting", { completed: job.completed_chunks, total: job.total_chunks });
  }
  if (job.status === "synthesizing") return t("training.synthesizing");
  if (job.status === "completed") return t("common.trained");
  return t("training.failed");
}

export function startTrainingJobPolling({ api, jobId, onUpdate, onComplete, onFailure, intervalMs = 2500 }) {
  let stopped = false;
  let timer;
  async function poll() {
    if (stopped) return;
    try {
      const job = await api.get(`/training-jobs/${jobId}`);
      if (stopped) return;
      onUpdate(job);
      if (job.status === "completed") {
        stopped = true;
        onComplete(job);
        return;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        stopped = true;
        onFailure(job);
        return;
      }
    } catch {
      // A transient network error is not a failed training job.
    }
    if (!stopped) timer = setTimeout(poll, intervalMs);
  }
  void poll();
  return () => { stopped = true; clearTimeout(timer); };
}
