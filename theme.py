"""Shared visual theme (black and gold) for the Streamlit app and login page."""

from __future__ import annotations

import streamlit as st

from gyanpur_booth.palette import (
    BLACK, CHARCOAL, CHARCOAL_DEEP, FONT_STACK, GOLD, GOLD_DEEP, MUTED, WHITE,
)

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {{
  --black: {BLACK}; --charcoal: {CHARCOAL}; --deep: {CHARCOAL_DEEP};
  --gold: {GOLD}; --gold-deep: {GOLD_DEEP}; --white: {WHITE}; --muted: {MUTED};
}}

/* ---- base ---------------------------------------------------------- */
.stApp {{ background: var(--black); color: var(--white); }}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div,
.stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2,
.stApp h3, .stApp h4 {{
  font-family: {FONT_STACK};
}}
/* keep icon fonts intact */
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important;
}}
.stApp {{ -webkit-font-smoothing: antialiased; letter-spacing: 0.005em; }}

/* ---- chrome removal ---------------------------------------------------- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stSidebarNav"],
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}

.main .block-container, [data-testid="stMainBlockContainer"] {{
  max-width: 960px; padding-top: 2rem; padding-bottom: 6rem;
}}

/* ---- headings ---------------------------------------------------------- */
.stApp h1, .stApp h2, .stApp h3 {{
  font-weight: 600; letter-spacing: -0.02em; color: var(--white);
}}

/* ---- brand / hero ------------------------------------------------------ */
.brand {{
  background: var(--charcoal); border-radius: 14px;
  padding: 1.35rem 1.6rem; border-top: 3px solid var(--gold);
  box-shadow: 0 10px 30px rgba(0,0,0,.6), 0 0 0 1px rgba(243,189,59,.08);
}}
.brand .name {{
  font-size: 1.55rem; font-weight: 600; letter-spacing: -0.02em; margin: 0;
}}
.brand .sub {{
  margin: .3rem 0 0 0; font-size: .92rem; font-weight: 300; color: var(--muted);
}}
.brand .meta {{
  margin: .8rem 0 0 0; font-size: .74rem; font-weight: 500;
  letter-spacing: .14em; text-transform: uppercase; color: var(--gold);
}}

.section-label {{
  font-size: .72rem; font-weight: 600; letter-spacing: .16em;
  text-transform: uppercase; color: var(--white); margin: 1.4rem 0 .5rem 0;
  opacity: .95;
}}

/* ---- buttons ----------------------------------------------------------- */
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"] {{
  background: var(--charcoal); color: var(--white);
  border: 1px solid rgba(255,255,255,.14); border-radius: 10px;
  font-weight: 400; font-size: .9rem; text-align: left;
  height: auto; white-space: normal; padding: .6rem .9rem;
  transition: border-color .15s ease, background .15s ease;
}}
.stApp button[kind="secondary"]:hover, .stApp [data-testid="stBaseButton-secondary"]:hover {{
  border-color: var(--gold); background: var(--deep); color: var(--white);
}}
.stApp button[kind="secondary"] p, .stApp [data-testid="stBaseButton-secondary"] p {{
  color: var(--white);
}}
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"],
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(180deg, var(--gold), var(--gold-deep)); color: var(--black); border: 0; border-radius: 10px;
  font-weight: 600; letter-spacing: .02em;
}}
.stApp button[kind="primary"] p, .stApp button[kind="primaryFormSubmit"] p,
.stApp [data-testid="stBaseButton-primary"] p,
.stApp [data-testid="stBaseButton-primaryFormSubmit"] p {{ color: var(--black); }}
.stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover {{
  filter: brightness(1.06);
}}
.topbar button {{ font-size: .82rem !important; padding: .45rem .6rem !important; text-align: center !important; }}

/* ---- voice ------------------------------------------------------------- */
.voice-card {{
  background: var(--charcoal); border: 1px solid rgba(243,189,59,.18);
  border-radius: 14px; padding: 1rem 1.2rem; margin: .2rem 0 .7rem 0;
}}
.voice-title {{ color: var(--white); font-weight: 600; font-size: .98rem; }}
.voice-help {{ color: var(--muted); font-size: .82rem; line-height: 1.55; margin-top: .25rem; }}
[data-testid="stAudioInput"] {{
  background: var(--charcoal); border-radius: 12px;
  border: 1px solid rgba(255,255,255,.12); padding: .2rem;
}}

/* ---- inputs ------------------------------------------------------------ */
.stApp [data-testid="stTextInputRootElement"], .stApp [data-baseweb="input"],
.stApp [data-baseweb="base-input"] {{
  background: var(--deep) !important; border-radius: 10px;
  border: 1px solid rgba(255,255,255,.22) !important;
}}
.stApp [data-testid="stTextInputRootElement"]:focus-within {{
  border-color: var(--gold) !important;
}}
.stApp input {{ color: var(--white) !important; background: transparent !important; }}
.stApp label p {{ color: var(--white); font-weight: 500; font-size: .85rem; }}
[data-testid="stForm"] {{
  background: var(--charcoal); border: 0; border-radius: 14px;
  padding: 1.4rem 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,.6), 0 0 0 1px rgba(243,189,59,.08);
}}

/* ---- chat -------------------------------------------------------------- */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] > div:first-child:has(img, svg) {{ display: none !important; }}
[data-testid="stChatMessage"] {{
  background: var(--charcoal); border-radius: 12px; padding: 1rem 1.3rem;
  margin-bottom: .8rem; border: 1px solid rgba(255,255,255,.06);
}}
.role {{
  font-size: .68rem; font-weight: 600; letter-spacing: .16em;
  text-transform: uppercase; color: var(--gold); margin-bottom: .35rem;
}}
.plain {{
  font-size: .97rem; font-weight: 400; line-height: 1.7; color: var(--white);
  overflow-wrap: anywhere;
}}
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {{ background: var(--black) !important; }}
[data-testid="stChatInput"] {{
  background: var(--charcoal); border-radius: 12px;
  border: 1px solid rgba(255,255,255,.14);
}}
[data-testid="stChatInput"] textarea {{ color: var(--white) !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted); }}

/* ---- misc -------------------------------------------------------------- */
[data-testid="stSpinner"] p, .stSpinner p {{ color: var(--white); }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
[data-testid="stImage"] img {{ border-radius: 8px; }}
[data-testid="stImageCaption"], .stApp [data-testid="stCaptionContainer"] {{
  color: var(--white); font-size: .78rem; font-weight: 300; text-align: center;
}}
.stApp a {{ color: var(--gold); }}
</style>
"""


def apply() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
