"""Turn LLM output into clean plain text (no markdown, no emphasis marks)."""

from __future__ import annotations

import re


def to_plain_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).replace("\r\n", "\n")

    # code fences / inline code
    t = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", t)
    t = t.replace("`", "")

    # markdown tables -> plain rows
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith("|") or (s.count("|") >= 2 and s.endswith("|")):
            if re.fullmatch(r"[\s|:\-]+", s):
                continue  # separator row
            cells = [c.strip() for c in s.strip("|").split("|")]
            line = "  |  ".join(c for c in cells if c)
        lines.append(line)
    t = "\n".join(lines)

    # horizontal rules
    t = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", "", t)
    # headings and blockquotes
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s{0,3}>\s?", "", t)
    # bullets: "* item" / "• item" -> "- item"
    t = re.sub(r"(?m)^(\s*)[*•]\s+", r"\1- ", t)
    # links [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", t)
    # bold / italic markers
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)
    t = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"\1", t)
    t = re.sub(r"\*(?!\s)(.+?)(?<!\s)\*", r"\1", t)
    # any stray asterisks left over
    t = t.replace("*", "")

    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
