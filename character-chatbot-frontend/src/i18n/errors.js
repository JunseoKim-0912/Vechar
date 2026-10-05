const CODE_KEYS = {
  daily_limit_reached: "errors.dailyLimit",
  monthly_limit_reached: "errors.monthlyLimit",
  user_not_found: "errors.unauthorized",
  training_source_missing: "errors.trainingSourceMissing",
  empty_training_file: "errors.emptyTrainingFile",
  invalid_source_file_type: "errors.invalidSourceFileType",
  source_invalid_utf8: "errors.sourceInvalidUtf8",
  source_too_long: "errors.sourceTooLong",
  malformed_training_request: "errors.malformedTrainingRequest",
  training_context_too_large: "errors.trainingContextTooLarge",
  llm_input_too_large: "errors.trainingContextTooLarge",
  training_queue_unavailable: "errors.server",
  training_failed: "errors.server",
  chunk_failed: "errors.server",
  synthesis_failed: "errors.server",
  planning_failed: "errors.server",
  llm_request_rejected: "errors.server",
};

export function localizeError(error, t, context) {
  if (context === "login") return t("errors.loginFailed");
  if (context === "signup" && error?.status === 409) return t("errors.emailExists");
  if (context === "signup" && error?.status === 422) return t("errors.signupValidation");
  const codeKey = CODE_KEYS[error?.code];
  if (codeKey) return t(codeKey);
  if (context === "training" && error?.status === 400) return t("errors.malformedTrainingRequest");
  if (context === "training" && error?.status === 422) return t("errors.malformedTrainingRequest");
  if (error?.status === 401) return t("errors.unauthorized");
  if (error?.status === 403) return t("errors.forbidden");
  if (error?.status === 404) return t("errors.notFound");
  if (error?.status >= 500) return t("errors.server");
  return t("errors.generic");
}
