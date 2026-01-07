from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .kneedle import kneedle

def build_sim_matrices(
    cosine_results: List[Dict[str, Any]],
    all_meta: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
      sim_matrix: (N, N, 4) = [from_id, rank(1-based), similarity, to_id]
      similarity_matrix: (N, N) float
    NOTE: from_id uses all_meta[i] (no regex parsing).
    """
    N = len(cosine_results)
    if N != len(all_meta):
        raise ValueError(f"Length mismatch: cosine_results={N}, all_meta={len(all_meta)}")

    sim_matrix = np.empty((N, N, 4), dtype=object)

    for i, result in enumerate(cosine_results):
        fm = all_meta[i]
        from_id = f"{fm['source_name']}/{fm['element_name']}".lower()

        candidates = result.get("candidates", [])
        for j, cand in enumerate(candidates[:N]):
            tm = cand.get("metadata", {})
            to_id = f"{tm['source_name']}/{tm['element_name']}".lower()
            sim = float(cand["similarity"])

            sim_matrix[i, j, 0] = from_id
            sim_matrix[i, j, 1] = j + 1
            sim_matrix[i, j, 2] = sim
            sim_matrix[i, j, 3] = to_id

    similarity_matrix = sim_matrix[:, :, 2].astype(float)
    return sim_matrix, similarity_matrix

def compute_thresholds(similarity_matrix: np.ndarray, S: float = 1.0, D: float = 0.85) -> np.ndarray:
    N = similarity_matrix.shape[0]
    thresholds = np.empty(N, dtype=float)
    for i in range(N):
        kp = kneedle(similarity_matrix[i], S=S, D=D)
        thresholds[i] = float(kp) if kp is not None else 0.0
    return thresholds

def apply_thresholds(
    sim_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """
    Returns:
      filtered: list[list[{'Candidate': id, 'Cosine Similarity': sim}, ...]]
      real_filter: filtered with first element removed (to drop self)
    """
    N = similarity_matrix.shape[0]
    mask = similarity_matrix >= thresholds[:, None]

    filtered: List[List[dict]] = []
    for i in range(N):
        valid_idxs = np.where(mask[i])[0]
        order = valid_idxs[np.argsort(similarity_matrix[i, valid_idxs])[::-1]]
        to_ids = sim_matrix[i, order, 3]
        sims = similarity_matrix[i, order]

        filtered.append(
            [{"Candidate": str(to_id), "Cosine Similarity": float(sim)} for to_id, sim in zip(to_ids, sims)]
        )

    real_filter = [f[1:] for f in filtered]
    return filtered, real_filter