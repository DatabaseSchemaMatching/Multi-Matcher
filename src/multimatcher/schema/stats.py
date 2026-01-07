from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import json
import random
import numpy as np
import pandas as pd
from pandas.api import types as pdt

def stat_compute(series: pd.Series) -> Optional[Dict[str, Optional[Union[int, float, str, bool]]]]:
    if pd.api.types.is_numeric_dtype(series):
        null_count = series.isna().sum()
        count_val = series.count()
        percentage_unique = round(series.nunique() / count_val * 100, 1) if count_val > 0 else None

        return {
            "count": count_val,
            "min": round(series.min(), 1) if count_val > 0 else None,
            "max": round(series.max(), 1) if count_val > 0 else None,
            "mean": round(series.mean(), 1) if count_val > 0 else None,
            "median": round(series.median(), 1) if count_val > 0 else None,
            "std": round(series.std(), 1) if count_val > 0 else None,
            "var": round(series.var(), 1) if count_val > 0 else None,
            "percentage_unique_value": percentage_unique,
            "possible_primary_key": bool(null_count == 0 and (percentage_unique == 100)),
        }
    return None

def get_data_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pdt.is_string_dtype(dtype):
        return "string"
    if pdt.is_integer_dtype(dtype):
        return "integer"
    if pdt.is_float_dtype(dtype):
        return "float"
    if pdt.is_bool_dtype(dtype):
        return "boolean"
    if pdt.is_datetime64_any_dtype(dtype):
        return "datetime"
    return str(dtype)

def extract_unique_values(col_vals: pd.Series) -> List[Any]:
    """Supports JSON-encoded list strings; flattens list values if any exist."""
    def try_parse(x: Any) -> Any:
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return x

    def flatten(x: Any):
        if isinstance(x, list):
            for elem in x:
                yield from flatten(elem)
        else:
            yield x

    parsed = col_vals.map(try_parse)

    if parsed.apply(lambda x: isinstance(x, list)).any():
        seen = set()
        uniques: List[Any] = []
        for item in parsed:
            for val in flatten(item):
                if pd.isna(val):
                    continue
                if val not in seen:
                    seen.add(val)
                    uniques.append(val)
        return uniques

    return parsed.unique().tolist()

def sample_up_to_k(values: List[Any], k: int = 5) -> List[Any]:
    return values if len(values) <= k else random.sample(values, k=k)

def to_python_types(obj: Any) -> Any:
    """Convert NumPy scalars recursively to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    return obj