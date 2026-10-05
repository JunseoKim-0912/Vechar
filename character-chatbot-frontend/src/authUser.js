export function normalizeCurrentUser(user) {
  if (!user || typeof user !== "object") return null;

  return {
    id: user.id,
    email: user.email,
    role: user.role === "admin" ? "admin" : "user",
  };
}

export function isAdminUser(user) {
  return normalizeCurrentUser(user)?.role === "admin";
}
