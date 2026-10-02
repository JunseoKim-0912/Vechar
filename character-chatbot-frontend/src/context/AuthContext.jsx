import { createContext, useContext, useState, useCallback } from "react";
import { api, getToken, setToken as persistToken } from "../api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(getToken());

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
  }, []);

  const value = { token, login, signup, logout, isAuthenticated: Boolean(token) };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}