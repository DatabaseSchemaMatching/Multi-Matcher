from __future__ import annotations
from typing import Any, Dict, List
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

def compute_pairwise_cosine_similarity(
    queries: List[str],
    metadata: List[Dict[str, Any]],
    openai_api_key: str,
    embedding_model: str,
    vectordb_path: str,
    collection_name: str,
) -> List[Dict[str, Any]]:
    """
    Compute all-pairs cosine similarity via Chroma (cosine distance -> similarity).
    Returns:
      [{"query": str, "candidates": [{"similarity": float, "metadata": dict, "document": str}, ...]}, ...]
    """
    assert len(queries) == len(metadata), "queries and metadata must have same length"

    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    Path(vectordb_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=vectordb_path)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model,
    )

    def _safe_delete(name: str):
        try:
            client.delete_collection(name=name)
        except Exception:
            pass

    def _create_collection_with_fallback(name: str):
        try:
            _safe_delete(name)
            return client.create_collection(
                name=name,
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            _safe_delete(name)
            return client.create_collection(
                name=name,
                embedding_function=openai_ef,
            )

    collection = _create_collection_with_fallback(collection_name)

    ids = [f"{m.get('source_name','?')}-{m.get('element_name','?')}" for m in metadata]
    collection.add(documents=queries, metadatas=metadata, ids=ids)

    n = len(queries)
    k = min(n, collection.count())

    out = collection.query(
        query_texts=queries,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = out.get("documents", [])
    metas_list = out.get("metadatas", [])
    distances_list = out.get("distances", [])

    all_results: List[Dict[str, Any]] = []
    for q, docs, metas, dists in zip(queries, docs_list, metas_list, distances_list):
        cands = []
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            cands.append({"similarity": sim, "metadata": meta, "document": doc})
        cands.sort(key=lambda x: x["similarity"], reverse=True)
        all_results.append({"query": q, "candidates": cands})

    return all_results