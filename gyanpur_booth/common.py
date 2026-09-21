#!/usr/bin/env python3
"""
Shared infrastructure: Gemini LLM client, SQL safety, SQLite helpers.
No business-logic hardcoding — only security and transport.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from the project's .env file. override=True so an
# empty/stale value already present in the process environment (e.g. set to ""
# by the Streamlit launcher) cannot mask the real value from .env.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Google Gemini (generateContent REST API)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "") or "gemini-2.5-flash"
GEMINI_URL = (
    os.environ.get("GEMINI_URL", "")
    or "https://generativelanguage.googleapis.com/v1beta/models"
).rstrip("/")
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "120"))
LLM_MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "3"))

_RETRYABLE_HTTP = (429, 500, 502, 503, 504)


class ToolFailure(Exception):
    """
    Raised by a specialist agent (voter / history / portfolio) when it could
    not produce an answer.

    message : short, plain-language, safe to show the end user (no SQL).
    detail  : technical info for logs and the planner only.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _retry_delay(err: "urllib.error.HTTPError", attempt: int) -> float:
    try:
        return min(30.0, float(err.headers.get("retry-after")))
    except (TypeError, ValueError, AttributeError):
        return min(30.0, 2.0 * (2 ** (attempt - 1)))  # 2s, 4s, ...


def _to_gemini_payload(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Convert OpenAI-style chat messages into a Gemini generateContent body."""
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        text = str(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
            continue
        g_role = "model" if role == "assistant" else "user"
        # Merge consecutive same-role turns (Gemini expects alternation).
        if contents and contents[-1]["role"] == g_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": g_role, "parts": [{"text": text}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }
    return payload


_FALLBACK_MODEL = "gemini-2.5-flash"


def _sanitize_model(model: str) -> str:
    """Strip a 'models/' prefix and refuse Live/audio-streaming models.

    Live models (e.g. '*-live', '*-native-audio*') only work over the
    bidiGenerateContent WebSocket API and always fail on generateContent.
    """
    name = (model or "").strip()
    if name.startswith("models/"):
        name = name[len("models/"):]
    low = name.lower()
    if not name:
        return _FALLBACK_MODEL
    if "live" in low or "native-audio" in low or "realtime" in low:
        print(
            f"[llm] WARNING: model {name!r} is a Live/streaming model and does not "
            f"support generateContent. Falling back to {_FALLBACK_MODEL!r}. "
            f"Set GEMINI_MODEL in .env to a standard text model."
        )
        return _FALLBACK_MODEL
    return name


class GeminiClient:
    """Thin client for the Google Gemini generateContent API."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        timeout: int = GEMINI_TIMEOUT,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = _sanitize_model(model or DEFAULT_GEMINI_MODEL)
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            return (
                "<<LLM_ERROR: GEMINI_API_KEY is not set. "
                "Ensure it is defined in your .env file or exported.>>"
            )
        payload = _to_gemini_payload(messages, temperature, max_tokens)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GEMINI_URL}/{self.model}:generateContent",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "User-Agent": "GyanpurBoothAgent/2.1",
            },
            method="POST",
        )
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                parsed = json.loads(raw.decode("utf-8"))
                candidates = parsed.get("candidates") or []
                if not candidates:
                    reason = (parsed.get("promptFeedback") or {}).get("blockReason", "")
                    return f"<<LLM_ERROR: empty candidates from Gemini {reason}>>".replace(" >>", ">>")
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(
                    str(p.get("text", "")) for p in parts if not p.get("thought")
                ).strip()
                if not text:
                    finish = candidates[0].get("finishReason", "unknown")
                    return f"<<LLM_ERROR: empty response from Gemini (finishReason={finish})>>"
                return text
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:1500]
                except Exception:
                    pass
                err = (
                    f"<<LLM_ERROR: HTTP {e.code} {e.reason}. "
                    f"Model={self.model!r}. Body={body}>>"
                )
                if e.code in _RETRYABLE_HTTP and attempt < LLM_MAX_ATTEMPTS:
                    wait = _retry_delay(e, attempt)
                    print(f"[llm] HTTP {e.code} (rate limit / server); "
                          f"retry {attempt}/{LLM_MAX_ATTEMPTS - 1} in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print(f"[llm] {err[:1600]}")
                return err
            except Exception as e:
                print(f"[llm] <<LLM_ERROR: {e}>>")
                return f"<<LLM_ERROR: {e}>>"
        return "<<LLM_ERROR: retries exhausted>>"


# ---------------------------------------------------------------------------
# JSON / text helpers
# ---------------------------------------------------------------------------

def safe_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    for candidate in (
        text,
        re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip(),
    ):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def compact_json(value: Any, max_chars: int = 10000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def extract_sql(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if m:
        ok, cleaned = validate_select(m.group(1))
        if ok:
            return cleaned
    m = re.search(r"(?is)\b(select|with)\b", text)
    if m:
        ok, cleaned = validate_select(text[m.start() :])
        if ok:
            return cleaned
    return None


# ---------------------------------------------------------------------------
# SQL safety (security only)
# ---------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe identifier: {name}")
    return f'"{name}"'


def validate_select(sql: str) -> Tuple[bool, str]:
    if not sql:
        return False, "Empty SQL."
    cleaned = sql.strip().rstrip(";").strip()
    low = cleaned.lower().lstrip("(").strip()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "Only SELECT/WITH allowed."
    forbidden = (
        "insert ", "update ", "delete ", "drop ", "alter ", "create ",
        "replace ", "attach ", "detach ", "pragma ", "vacuum ", "truncate ",
    )
    full = cleaned.lower()
    for kw in forbidden:
        if kw in full:
            return False, f"Forbidden operation: {kw.strip()}"
    if ";" in cleaned:
        return False, "Multiple statements not allowed."
    return True, cleaned


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

class SQLiteDB:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def tables(self) -> List[str]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def columns(self, table: str) -> List[str]:
        tq = quote_ident(table)
        with self.connect() as c:
            rows = c.execute(f"PRAGMA table_info({tq})").fetchall()
        return [r["name"] for r in rows]

    def schema_text(self) -> str:
        lines = []
        for t in self.tables():
            cols = self.columns(t)
            lines.append(f"TABLE {t}: {', '.join(cols)}")
        return "\n".join(lines) if lines else "(no tables)"

    def distinct(self, table: str, column: str, limit: int = 400) -> List[str]:
        tq, cq = quote_ident(table), quote_ident(column)
        with self.connect() as c:
            rows = c.execute(
                f"""
                SELECT DISTINCT {cq} FROM {tq}
                WHERE {cq} IS NOT NULL AND TRIM(CAST({cq} AS TEXT)) != ''
                ORDER BY {cq} LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(r[0]).strip() for r in rows if str(r[0]).strip()]

    def sample_values(self, table: str, column: str, limit: int = 40) -> List[str]:
        vals = self.distinct(table, column, limit=limit)
        return vals

    def execute(self, sql: str, max_rows: int = 80) -> List[Dict[str, Any]]:
        ok, cleaned = validate_select(sql)
        if not ok:
            raise ValueError(cleaned)
        with self.connect() as c:
            cur = c.execute(cleaned)
            rows = cur.fetchmany(max_rows)
            return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# ReAct memory
# ---------------------------------------------------------------------------

@dataclass
class ReactStep:
    step: int
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class ReactState:
    question: str
    steps: List[ReactStep] = field(default_factory=list)

    def memory(self, max_chars: int = 14000) -> str:
        parts = []
        for s in self.steps:
            parts.append(
                f"Step {s.step}\n"
                f"Thought: {s.thought}\n"
                f"Action: {s.action}\n"
                f"Input: {s.action_input}\n"
                f"Observation: {s.observation[:4000]}"
            )
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            return "[older steps truncated]\n" + text[-max_chars:]
        return text