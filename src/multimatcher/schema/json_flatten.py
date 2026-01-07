from __future__ import annotations
from typing import Dict, Any
import json
from collections import defaultdict

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict into dot-notation keys.
    - Handles JSON-encoded list strings
    - Skips empty lists
    - Merges list-of-dict fields by key
    """
    items: Dict[str, Any] = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        v = parsed
                except json.JSONDecodeError:
                    pass

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))

        elif isinstance(v, list):
            if not v:
                continue
            elif all(isinstance(elem, dict) for elem in v):
                merged: Dict[str, list] = defaultdict(list)
                for elem in v:
                    flat_elem = flatten_dict(elem, parent_key="", sep=sep)
                    for subkey, subval in flat_elem.items():
                        merged[subkey].append(subval)
                for subkey, subvals in merged.items():
                    combined_key = f"{new_key}{sep}{subkey}"
                    items[combined_key] = subvals[0] if len(subvals) == 1 else subvals
            else:
                items[new_key] = v

        else:
            items[new_key] = v

    return items