import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Layout from "./components/Layout";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import CharacterListPage from "./pages/CharacterListPage";
import CharacterCreatePage from "./pages/CharacterCreatePage";
import CharacterDetailPage from "./pages/CharacterDetailPage";
import ChatPage from "./pages/ChatPage";
import CharacterConversationPage from "./pages/CharacterConversationPage";
import WorldListPage from "./pages/WorldListPage";
import WorldDetailPage from "./pages/WorldDetailPage";
import { useLocale } from "./context/LocaleContext";

function NotFoundPage() {
  const { t } = useLocale();
  return <div className="empty-state"><h1>{t("notFound.title")}</h1><Link to="/characters">{t("notFound.action")}</Link></div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />

          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route path="/" element={<Navigate to="/characters" replace />} />
              <Route path="/characters" element={<CharacterListPage />} />
              <Route path="/characters/new" element={<CharacterCreatePage />} />
              <Route path="/characters/:id" element={<CharacterDetailPage />} />
              <Route path="/chat/:characterId" element={<ChatPage />} />
              <Route path="/character-conversations/:roomId" element={<CharacterConversationPage />} />
              <Route path="/worlds" element={<WorldListPage />} />
              <Route path="/worlds/:id" element={<WorldDetailPage />} />
            </Route>
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
