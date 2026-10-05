import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api, getToken, setToken as persistToken } from "../api";
import { normalizeCurrentUser } from "../authUser";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(getToken());
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (!token) return undefined;

    let cancelled = false;
    api.get("/auth/me")
      .then((currentUser) => {
        if (!cancelled) setUser(normalizeCurrentUser(currentUser));
      })
      .catch((error) => {
        if (!cancelled && error?.status === 401) {
          persistToken(null);
          setTokenState(null);
          setUser(null);
        }
      });

    return () => { cancelled = true; };
  }, [token]);

  const login = useCallback(async (email, password) => {
    const data = await api.post("/auth/login", { email, password });
    persistToken(data.token);
    setTokenState(data.token);
  }, []);

  const signup = useCallback(async (email, password) => {
    const data = await api.post("/auth/signup", { email, password });
    persistToken(data.token);
    setTokenState(data.token);
  }, []);

  const logout = useCallback(() => {
    persistToken(null);
    setTokenState(null);
    setUser(null);
  }, []);

  const value = { token, user, login, signup, logout, isAuthenticated: Boolean(token) };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// This hook intentionally shares the context module with its provider.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
