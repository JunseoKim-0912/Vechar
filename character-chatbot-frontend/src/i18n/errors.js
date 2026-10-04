const CODE_KEYS = {
  daily_limit_reached: "errors.dailyLimit",
  monthly_limit_reached: "errors.monthlyLimit",
  user_not_found: "errors.unauthorized",
};

export function localizeError(error, t, context) {
  if (context === "login") return t("errors.loginFailed");
  if (context === "signup" && error?.status === 409) return t("errors.emailExists");
  if (context === "signup" && error?.status === 422) return t("errors.signupValidation");
  const codeKey = CODE_KEYS[error?.code];
  if (codeKey) return t(codeKey);
  if (error?.status === 401) return t("errors.unauthorized");
  if (error?.status === 403) return t("errors.forbidden");
  if (error?.status === 404) return t("errors.notFound");
  if (error?.status >= 500) return t("errors.server");
  return t("errors.generic");
}
