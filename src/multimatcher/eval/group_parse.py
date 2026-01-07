from __future__ import annotations
from typing import List, Set
import re

_WS_RE = re.compile(r"[\s\u200b\u00a0]+")

def _extract_bracketed_groups(s: str) -> List[str]:
    if not s:
        return []
    if "\ufeff" in s:
        s = s.replace("\ufeff", "")
    blocks = []
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    blocks.append(s[start : i + 1].strip())
                    start = -1
    return blocks

def _canon(elem: str) -> str:
    if elem is None:
        return ""
    s = str(elem).lower()
    if _WS_RE.search(s):
        s = _WS_RE.sub("", s)
    s = s.replace("\u2215", "/")
    if "/" in s:
        return s
    dot_pos = s.find(".")
    if dot_pos != -1:
        return s[:dot_pos] + "/" + s[dot_pos + 1 :]
    return s

def _iter_flat_tokens(inner: str):
    n = len(inner)
    i = 0
    buf = []
    in_quote = None
    escape = False

    while i < n:
        ch = inner[i]
        if escape:
            buf.append(ch); escape = False; i += 1; continue
        if ch == "\\":
            buf.append(ch); escape = True; i += 1; continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            buf.append(ch); in_quote = ch; i += 1; continue

        if ch == "[":
            j = i + 1; d = 1
            sub_buf = []
            in_quote2 = None; escape2 = False
            while j < n:
                cj = inner[j]
                if escape2:
                    sub_buf.append(cj); escape2 = False; j += 1; continue
                if cj == "\\":
                    sub_buf.append(cj); escape2 = True; j += 1; continue
                if in_quote2:
                    sub_buf.append(cj)
                    if cj == in_quote2:
                        in_quote2 = None
                    j += 1; continue
                if cj in ("'", '"'):
                    sub_buf.append(cj); in_quote2 = cj; j += 1; continue
                if cj == "[":
                    d += 1; sub_buf.append(cj); j += 1; continue
                if cj == "]":
                    d -= 1
                    if d == 0:
                        sub_inner = "".join(sub_buf).strip()
                        if sub_inner:
                            for t in _iter_flat_tokens(sub_inner):
                                yield t
                        break
                    else:
                        sub_buf.append(cj); j += 1; continue
                else:
                    sub_buf.append(cj); j += 1
            i = j + 1
            continue

        if ch == ",":
            token = "".join(buf).strip()
            if token:
                yield token
            buf = []
            i += 1
            continue

        buf.append(ch); i += 1

    final = "".join(buf).strip()
    if final:
        yield final

def clean_schema_groups_from_strings(schema_groups_str: List[str]) -> List[List[str]]:
    parsed: List[List[str]] = []

    for raw in schema_groups_str:
        if raw is None:
            continue
        s_strip = str(raw).strip()
        if not s_strip or s_strip.lower() == "none":
            continue

        for blk in _extract_bracketed_groups(s_strip):
            inner = blk[1:-1].strip()
            if not inner:
                continue
            items = [_canon(tok) for tok in _iter_flat_tokens(inner) if tok.strip()]
            if len(items) >= 2:
                parsed.append(items)

    seen: Set[frozenset] = set()
    result: List[List[str]] = []
    for group in parsed:
        sources = {elem.split("/", 1)[0] for elem in group if "/" in elem}
        if len(sources) > 1:
            key = frozenset(group)
            if key not in seen:
                seen.add(key)
                result.append(sorted(key))
    return result