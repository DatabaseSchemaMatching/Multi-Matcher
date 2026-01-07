from __future__ import annotations
from typing import Iterable, List, Set
import ast
import pandas as pd
from itertools import combinations
from .group_parse import clean_schema_groups_from_strings

def evaluate_schema_grouping(schema_groups: List[str], group_path: str) -> None:
    def canon(token: str) -> str:
        s = token.strip().lower()
        if not s:
            return s
        if "/" in s:
            return s
        dot = s.find(".")
        return (s[:dot] + "/" + s[dot + 1 :]) if dot != -1 else s

    df_gt = pd.read_csv(group_path)
    if "group" not in df_gt.columns:
        raise ValueError("CSV needs 'group' column")

    def parse_group_cell(cell: str) -> List[str]:
        s = str(cell).strip()
        if not s or s.lower() == "none":
            return []
        try:
            items = ast.literal_eval(s)
        except Exception:
            inner = s.lstrip("[").rstrip("]")
            items = [x for x in inner.split(",") if x.strip()]
        return [canon(x) for x in items]

    gt_unique = set()
    for cell in df_gt["group"].dropna():
        grp = parse_group_cell(cell)
        if len(grp) >= 2:
            gt_unique.add(frozenset(grp))

    gt_groups: List[Set[str]] = [set(g) for g in gt_unique]
    gt_elements: Set[str] = {e for g in gt_unique for e in g}

    raw_pred = clean_schema_groups_from_strings(schema_groups)
    pred_groups: List[Set[str]] = [set(canon(e) for e in grp) for grp in raw_pred if len(grp) >= 2]
    pred_elements: Set[str] = {e for g in pred_groups for e in g}

    universe: Set[str] = gt_elements | pred_elements
    n = len(universe)
    total_pairs = n * (n - 1) // 2

    def group_pairs(groups: Iterable[Set[str]]) -> Set[frozenset]:
        pairs: Set[frozenset] = set()
        for g in groups:
            for a, b in combinations(sorted(g), 2):
                pairs.add(frozenset((a, b)))
        return pairs

    gt_pairs = group_pairs(gt_groups)
    pred_pairs = group_pairs(pred_groups)

    TP = len(pred_pairs & gt_pairs)
    FP_set = pred_pairs - gt_pairs
    FN_set = gt_pairs - pred_pairs
    FP = len(FP_set)
    FN = len(FN_set)
    TN = max(total_pairs - TP - FP - FN, 0)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("=== Pair-based evaluation (co-membership) ===")
    print(f"#elements (universe) = {n}")
    print(f"Total pairs          = {total_pairs}")
    print(f"TP_pairs             = {TP}")
    print(f"FP_pairs             = {FP}")
    print(f"FN_pairs             = {FN}")
    print(f"TN_pairs             = {TN}")
    print(f"Precision            = {precision:.6f}")
    print(f"Recall               = {recall:.6f}")
    print(f"F1-score             = {f1:.6f}")

    if FP > 0:
        print("\n[FP pairs]:")
        for p in sorted(FP_set):
            print(f"{tuple(p)}")
    if FN > 0:
        print("\n[FN pairs]:")
        for p in sorted(FN_set):
            print(f"{tuple(p)}")