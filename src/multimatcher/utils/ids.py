from __future__ import annotations
import re
from typing import Optional

_WS_RE = re.compile(r"[\s\u200b\u00a0]+")

def element_id(source_name: str, element_name: str) -> str:
    """Canonical id across pipeline: {source_name}/{element_name} (lowercased)."""
    return f"{str(source_name).strip().lower()}/{str(element_name).strip().lower()}"

def canon_token(token: Optional[str]) -> str:
    """
    Canonicalize token for evaluation / parsing:
    - normalize whitespace & unicode slashes
    - if '/' exists: keep as-is
    - else convert ONLY first '.' into '/' (source/element split)
    """
    if token is None:
        return ""
    s = str(token).strip().lower()
    if not s:
        return s

    if _WS_RE.search(s):
        s = _WS_RE.sub("", s)

    s = s.replace("\u2215", "/")  # division slash -> slash

    if "/" in s:
        return s

    dot = s.find(".")
    if dot != -1:
        return s[:dot] + "/" + s[dot + 1 :]
    return s