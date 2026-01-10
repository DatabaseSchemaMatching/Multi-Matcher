# scripts/run_dataset.py
from __future__ import annotations

import sys
from pathlib import Path

# B) packaging 없이 바로 실행: add repo_root/src to sys.path (minimal 3 lines)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

from multimatcher.datasets.registry import load_dataset
from multimatcher.schema.build import render_prompt_from_context
from multimatcher.retrieval.chroma_cosine import compute_pairwise_cosine_similarity
from multimatcher.filtering.thresholding import (
    build_sim_matrices,
    compute_thresholds,
    apply_thresholds,
)

from multimatcher.llm.registry import get_model_spec
from multimatcher.llm.factory import build_chat_model
from multimatcher.llm.prompts import REASONING_CANDIDATES_SYSTEM_MESSAGE
from multimatcher.llm.grouping import run_grouping

# NEW: desired outputs (cleaned groups + evaluation report)
from multimatcher.eval.group_parse import clean_schema_groups_from_strings
from multimatcher.eval.grouping_eval import evaluate_schema_grouping


# Load .env from repo root (or current working directory) if present
load_dotenv()

# -----------------------------
# Kneedle constants
# -----------------------------
KNEEDLE_S: float = 1.0  # fixed (advanced sensitivity); keep constant for reproducibility


def _get_env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    raise RuntimeError(
        "Missing environment variable. Tried: "
        f"{', '.join(names)}. "
        "Tip: copy .env.example -> .env and fill keys."
    )


def _repo_root() -> Path:
    """
    Resolve repository root as:
      repo_root = parent of 'scripts/' directory (this file lives in scripts/)
    This makes relative --data-root stable regardless of current working directory.
    """
    return Path(__file__).resolve().parents[1]


def _resolve_data_root(cli_data_root: Optional[str]) -> Path:
    """
    Decide dataset root directory with precedence:
      1) --data-root (if provided)
      2) MULTIMATCHER_DATA_ROOT env (if set)
      3) repo_root / "data" (fallback)

    If chosen path is relative, interpret it as relative to repo root (NOT CWD).
    """
    root = cli_data_root or os.getenv("MULTIMATCHER_DATA_ROOT") or "data"
    p = Path(root)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


def build_llm_reasoning_inputs(all_schema_contexts, real_filter) -> List[str]:
    """
    Notebook Cell 13 로직을 함수화:
      - Query: SchemaContext -> prompt string
      - Candidates: real_filter[i]의 Candidate id를 SchemaContext로 찾아 prompt string
      - 포맷: 'Query:{...}<->Candidates:{cand1|cand2|...}'
      - Candidates가 비면 Candidates:None 으로 명시
    """
    id_to_ctx: Dict[str, object] = {
        f"{ctx.source_name}/{ctx.element_name}": ctx for ctx in all_schema_contexts
    }

    llm_inputs: List[str] = []
    for i, query_ctx in enumerate(all_schema_contexts):
        query_text = render_prompt_from_context(query_ctx)

        cand_chunks: List[str] = []
        for entry in real_filter[i]:
            cand_id = entry.get("Candidate")
            if cand_id and cand_id in id_to_ctx:
                cand_chunks.append(render_prompt_from_context(id_to_ctx[cand_id]))

        cand_str = "|".join(cand_chunks) if cand_chunks else "None"
        llm_inputs.append(f"Query:{query_text}<->Candidates:{cand_str}")

    return llm_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="m2bench-ecommerce | m2bench-healthcare | unibench | m2e-unibench",
    )
    ap.add_argument(
        "--llm",
        required=True,
        help=(
            "gpt-5 | gpt-5-mini | gpt-oss-120b | gpt-oss-20b | "
            "gemini-2.5-pro | gemini-2.5-flash | "
            "claude-sonnet-4.5 | claude-haiku-4.5 | "
            "qwen3-max | qwen3-next-80b"
        ),
    )
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--vectordb-path", default=None)

    # data root (optional)
    ap.add_argument(
        "--data-root",
        default=None,
        help=(
            "Dataset root directory. If relative, it is resolved relative to repo root. "
            "If omitted, uses env MULTIMATCHER_DATA_ROOT; otherwise falls back to ./data under repo."
        ),
    )

    # (선택) LLM 파라미터 오버라이드. 지정 안 하면 registry 기본값 사용.
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=None)

    # ✅ Kneedle: expose only D (retention knob)
    ap.add_argument(
        "--kneedle-d",
        type=float,
        default=0.85,
        help=(
            "Post-knee scaling factor D (0< D <=1). "
            "Lower D keeps more candidates (higher recall, higher cost). "
            "Higher D prunes more (lower cost, higher recall risk)."
        ),
    )

    args = ap.parse_args()

    # Basic validation for D
    if not (0.0 < args.kneedle_d <= 1.0):
        raise ValueError(f"--kneedle-d must be in (0, 1]. Got: {args.kneedle_d}")

    # -----------------------------
    # 0) Resolve data root + Load dataset bundle
    # -----------------------------
    data_root = _resolve_data_root(args.data_root)

    if not data_root.exists():
        raise RuntimeError(
            f"Data root does not exist: {data_root}\n"
            "Fix by either:\n"
            "  - passing --data-root <path>\n"
            "  - setting MULTIMATCHER_DATA_ROOT in .env\n"
            "  - or placing datasets under repo_root/data\n"
        )

    # pass data_root to dataset loader
    bundle = load_dataset(args.dataset, data_root=str(data_root))
    all_schema_contexts = bundle.all_schema_contexts

    # -----------------------------
    # 1) Stage 1 -> text + meta
    # -----------------------------
    all_texts = [render_prompt_from_context(ctx) for ctx in all_schema_contexts]
    all_meta = [
        {
            "source_type": ctx.source_type,
            "source_name": ctx.source_name,
            "element_type": ctx.element_type,
            "element_name": ctx.element_name,
        }
        for ctx in all_schema_contexts
    ]

    vectordb_path = args.vectordb_path or os.path.join(bundle.spec.gt_dir, "vectordb")

    # Embedding key는 과거 변수명/새 변수명 둘 다 허용
    embedding_api_key = _get_env_any("OPENAI_EMBEDDING_API_KEY", "OPENAI_Embedding_API_KEY")

    # -----------------------------
    # 2) Stage 2 retrieval (cosine)
    # -----------------------------
    cosine_results = compute_pairwise_cosine_similarity(
        queries=all_texts,
        metadata=all_meta,
        openai_api_key=embedding_api_key,
        embedding_model=args.embedding_model,
        vectordb_path=vectordb_path,
        collection_name="candidates",
    )

    # regex 없이 from_id는 all_meta 기반으로
    sim_matrix, similarity_matrix = build_sim_matrices(cosine_results, all_meta)

    # -----------------------------
    # 3) Stage 2 filtering (Kneedle)
    # -----------------------------
    thresholds = compute_thresholds(similarity_matrix, S=KNEEDLE_S, D=args.kneedle_d)
    _, real_filter = apply_thresholds(sim_matrix, similarity_matrix, thresholds)

    # Defensive checks
    if len(real_filter) != len(all_schema_contexts):
        raise RuntimeError(
            f"Length mismatch: real_filter={len(real_filter)} vs contexts={len(all_schema_contexts)}"
        )

    # -----------------------------
    # 4) Stage 3 LLM grouping
    # -----------------------------
    llm_inputs = build_llm_reasoning_inputs(all_schema_contexts, real_filter)

    model_spec = get_model_spec(args.llm)
    chat = build_chat_model(
        model_spec,
        temperature=args.temperature,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )

    schema_groups_raw = run_grouping(
        chat_model=chat,
        llm_reasoning_inputs=llm_inputs,
        system_prompt=REASONING_CANDIDATES_SYSTEM_MESSAGE,
    )

    # -----------------------------
    # 5) Print summary
    # -----------------------------
    print(f"[DATA_ROOT] {data_root}")
    print(f"[DATASET] {bundle.spec.name}  contexts={len(all_schema_contexts)}")
    print(f"[LLM] alias={args.llm}  provider={model_spec.provider}  model={model_spec.model}")
    print(f"[EMBED] model={args.embedding_model}")
    print("vectordb_path:", vectordb_path)
    print("grouping_candidates_path:", bundle.grouping_candidates_path)
    print("group_path:", bundle.group_path)
    print(f"kneedle: S={KNEEDLE_S} (fixed), D={args.kneedle_d}")
    print(f"schema_groups_raw: {len(schema_groups_raw)} items")

    # -----------------------------
    # Desired output #1: cleaned grouping results
    # -----------------------------
    cleaned_groups = clean_schema_groups_from_strings(schema_groups_raw)
    print("\n=== Cleaned grouping results ===")
    print(f"#groups = {len(cleaned_groups)}")
    for i, g in enumerate(cleaned_groups, start=1):
        print(f"[Group {i}] {g}")

    # -----------------------------
    # Desired output #2: evaluation report (+ FP/FN lists)
    # -----------------------------
    print("\n=== Evaluation ===")
    evaluate_schema_grouping(schema_groups=schema_groups_raw, group_path=bundle.group_path)


if __name__ == "__main__":
    main()
