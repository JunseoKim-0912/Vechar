import { Link, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Layout() {
  const { logout } = useAuth();

  return (
    <div className="app-shell">
      <nav className="topnav">
        <Link to="/characters" className="brand">VECHAR</Link>
        <div className="nav-links">
          <Link to="/characters">캐릭터</Link>
          <Link to="/worlds">세계관</Link>
          <button onClick={logout} className="nav-logout">로그아웃</button>
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}