import { Link, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLocale } from "../context/LocaleContext";
import LanguageSelector from "./LanguageSelector";

export default function Layout() {
  const { logout } = useAuth();
  const { t } = useLocale();

  return (
    <div className="app-shell">
      <nav className="topnav">
        <Link to="/characters" className="brand">VECHAR</Link>
        <div className="nav-links">
          <Link to="/characters">{t("nav.characters")}</Link>
          <Link to="/worlds">{t("nav.worlds")}</Link>
          <LanguageSelector />
          <button onClick={logout} className="nav-logout">{t("nav.logout")}</button>
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
