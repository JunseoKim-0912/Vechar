import { Link, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLocale } from "../context/LocaleContext";
import { isAdminUser } from "../authUser";
import LanguageSelector from "./LanguageSelector";

export default function Layout() {
  const { logout, user } = useAuth();
  const { t } = useLocale();

  return (
    <div className="app-shell">
      <nav className="topnav">
        <Link to="/characters" className="brand">VECHAR</Link>
        <div className="nav-links">
          <Link to="/characters">{t("nav.characters")}</Link>
          <Link to="/worlds">{t("nav.worlds")}</Link>
          {user?.email && (
            <span className="nav-account">
              <span>{user.email}</span>
              {isAdminUser(user) && <span className="admin-badge">{t("nav.adminBadge")}</span>}
            </span>
          )}
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
