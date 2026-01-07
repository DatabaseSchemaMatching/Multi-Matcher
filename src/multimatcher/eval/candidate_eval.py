from __future__ import annotations
from typing import Any, List
import ast
import pandas as pd

def evaluate_candidates(filtered: List[List[dict]], grouping_candidates_path: str) -> pd.DataFrame:
    """
    filtered: list of list of {'Candidate': str, 'Cosine Similarity': float}
      - assumes filtered[i][0]['Candidate'] is the query id (self)
    """
    pred_rows = []
    for f in filtered:
        if not f:
            continue
        query = f[0]["Candidate"]
        query_norm = str(query).strip().lower()
        cands = {str(item["Candidate"]).strip().lower() for item in f[1:]}
        pred_rows.append((query_norm, cands))
    df_pred = pd.DataFrame(pred_rows, columns=["query", "candidates"])

    df_gt = pd.read_csv(grouping_candidates_path)
    df_gt["query"] = df_gt["schema_element"].str.strip().str.lower()

    def parse_candidates(cell: Any) -> set:
        if isinstance(cell, str):
            cell = cell.strip()
            try:
                items = ast.literal_eval(cell)
            except Exception:
                items = [x.strip() for x in cell.split(",") if x.strip()]
            return {str(x).strip().lower() for x in items}
        if isinstance(cell, (list, set)):
            return {str(x).strip().lower() for x in cell}
        return set()

    df_gt["truth_set"] = df_gt["grouping_candidates"].apply(parse_candidates)

    df_eval = pd.merge(df_gt[["query", "truth_set"]], df_pred, on="query", how="left")
    df_eval["candidates"] = df_eval["candidates"].apply(lambda x: x if isinstance(x, set) else set())
    df_eval["candidates"] = df_eval.apply(lambda r: r["candidates"] & r["truth_set"], axis=1)

    df_eval["missing"] = df_eval.apply(
        lambda r: True if not (r["truth_set"] - r["candidates"]) else sorted(r["truth_set"] - r["candidates"]),
        axis=1,
    )
    df_eval["ground_truth"] = df_eval["truth_set"].apply(lambda s: sorted(s))

    df_final = df_eval[df_eval["query"].notna()].copy()
    df_final.index = range(1, len(df_final) + 1)
    return df_final[["query", "missing", "ground_truth"]]