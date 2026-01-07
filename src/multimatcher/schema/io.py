from __future__ import annotations
from typing import List
import json
import pandas as pd

def load_json_lines(filepath: str) -> List[dict]:
    """Read JSON Lines -> list[dict]."""
    with open(filepath, "r", encoding="utf-8") as f:
        out = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

def read_csv_clean(
    file_path: str,
    encodings: List[str] = ["utf-8", "cp949", "latin1"],
    **kwargs,
) -> pd.DataFrame:
    """Try multiple encodings; drop 'Unnamed:' columns."""
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, **kwargs)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            return df
        except UnicodeDecodeError:
            continue

    # fallback
    df = pd.read_csv(file_path, encoding="latin1", engine="python", **kwargs)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    return df