"""
Login gate for the Arjun Streamlit app.

Credentials live in a CSV file with two columns:  id,password
(default: credentials.csv next to this file; override with CREDENTIALS_CSV in .env).
The file is re-read on every login attempt, so you can add/remove users
without restarting the app.
"""

from __future__ import annotations

import csv
import hmac
import os
import time
from pathlib import Path
from typing import Dict

import streamlit as st

from gyanpur_booth.palette import GOLD

ROOT = Path(__file__).resolve().parent
MAP_IMAGE = ROOT / "assets" / "gyanpur_location_map.png"
LOGO_IMAGE = ROOT / "assets" / "my_booth_agent_logo.png"

CONTACT_EMAIL = "aunsiddiqui@myboothagent.com"
MAX_ATTEMPTS = 5      # failed tries before a temporary lock
LOCK_SECONDS = 30


def credentials_path() -> Path:
    custom = os.environ.get("CREDENTIALS_CSV", "").strip()
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else ROOT / p
    return ROOT / "credentials.csv"


def load_credentials() -> Dict[str, str]:
    """Return {id: password} from the CSV. Missing/bad file -> empty dict."""
    path = credentials_path()
    creds: Dict[str, str] = {}
    if not path.exists():
        return creds
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return creds
            # tolerate header case/spacing differences: ID, Id, " password "
            fields = {(n or "").strip().lower(): n for n in reader.fieldnames}
            id_col, pw_col = fields.get("id"), fields.get("password")
            if not id_col or not pw_col:
                return creds
            for row in reader:
                uid = (row.get(id_col) or "").strip()
                pw = (row.get(pw_col) or "").strip()
                if uid and pw:
                    creds[uid] = pw
    except Exception:
        return {}
    return creds


def verify(user_id: str, password: str) -> bool:
    """True only if the (id, password) pair exists in the CSV."""
    user_id, password = (user_id or "").strip(), (password or "").strip()
    if not user_id or not password:
        return False
    stored = load_credentials().get(user_id)
    if stored is None:
        return False
    return hmac.compare_digest(stored.encode("utf-8"), password.encode("utf-8"))


_LOGIN_CSS = f"""
<style>
  .login-logo {{ display: flex; justify-content: center; margin: 0 0 1.6rem 0; }}
  .login-logo img {{ max-width: 260px; width: 100%; height: auto; }}
  .login-footer {{
    text-align: center; color: #fff; font-size: .88rem; font-weight: 300;
    margin-top: 2.2rem; padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,.35); line-height: 1.7;
  }}
  .login-footer p {{ margin: .3rem 0; }}
  .login-footer a {{ color: {GOLD}; text-decoration: none; font-weight: 500; }}
  .login-footer .rights {{ font-size: .78rem; margin-top: .9rem; opacity: .9; }}
  .login-title {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 .9rem 0; }}
</style>
"""


def _render_login() -> None:
    import base64

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    if LOGO_IMAGE.exists():
        logo_b64 = base64.b64encode(LOGO_IMAGE.read_bytes()).decode("ascii")
        st.markdown(
            f'<div class="login-logo"><img src="data:image/png;base64,{logo_b64}" '
            f'alt="Arjun"></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        """
<div class="brand">
  <p class="name">Arjun</p>
  <p class="sub">Electoral intelligence and data analysis for Gyanpur Vidhan Sabha</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1, 1], gap="large", vertical_alignment="center")

    with left:
        locked_until = st.session_state.get("_locked_until", 0.0)
        remaining = int(locked_until - time.time())
        locked = remaining > 0

        with st.form("login_form", clear_on_submit=False):
            st.markdown('<p class="login-title">Sign in</p>', unsafe_allow_html=True)
            user_id = st.text_input("Login ID", key="login_id")
            password = st.text_input("Password", type="password", key="login_pw")
            submitted = st.form_submit_button(
                "Log in", type="primary", use_container_width=True, disabled=locked
            )

        if locked:
            st.error(f"Too many failed attempts. Try again in {remaining}s.")
        elif submitted:
            if not credentials_path().exists():
                st.error(f"Credentials file not found: {credentials_path().name}")
            elif verify(user_id, password):
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = user_id.strip()
                st.session_state["_failed_attempts"] = 0
                st.rerun()
            else:
                fails = st.session_state.get("_failed_attempts", 0) + 1
                st.session_state["_failed_attempts"] = fails
                if fails >= MAX_ATTEMPTS:
                    st.session_state["_locked_until"] = time.time() + LOCK_SECONDS
                    st.session_state["_failed_attempts"] = 0
                    st.error(f"Too many failed attempts. Locked for {LOCK_SECONDS}s.")
                else:
                    st.error("Invalid ID or password.")

    with right:
        if MAP_IMAGE.exists():
            cap = "Gyanpur (highlighted) in Uttar Pradesh"
            try:
                st.image(str(MAP_IMAGE), caption=cap, width="stretch")  # new Streamlit
            except TypeError:
                st.image(str(MAP_IMAGE), caption=cap, use_container_width=True)  # old

    st.markdown(
        f"""
<div class="login-footer">
  <p>Arjun is a data-driven election intelligence platform currently available for Gyanpur.</p>
  <p>For access, contact <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>.</p>
  <p class="rights">© 2026 Make a Soft India. All rights reserved.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def require_login() -> bool:
    """Show the login page until the user authenticates.

    Usage (right after st.set_page_config):
        if not require_login():
            st.stop()
    """
    if st.session_state.get("authenticated"):
        return True
    _render_login()
    return False


def logout_button() -> None:
    """Small 'Log out' button; clears the session."""
    if st.button("Log out", use_container_width=True, key="logout_btn"):
        for k in ("authenticated", "user_id", "messages", "pending_prompt"):
            st.session_state.pop(k, None)
        st.rerun()
