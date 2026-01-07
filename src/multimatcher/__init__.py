# src/multimatcher/__init__.py
__version__ = "0.1.0"

# Public API (stable + safe to import without optional deps)
from .datasets import load_dataset
from .schema.build import render_prompt_from_context
from .retrieval.chroma_cosine import compute_pairwise_cosine_similarity
from .filtering.thresholding import build_sim_matrices

__all__ = [
    "__version__",
    "load_dataset",
    "render_prompt_from_context",
    "compute_pairwise_cosine_similarity",
    "build_sim_matrices",
]