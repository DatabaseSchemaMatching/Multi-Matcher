# src/multimatcher/datasets/__init__.py
# src/multimatcher/datasets/__init__.py
from .registry import load_dataset, get_dataset_spec, DatasetBundle, DatasetSpec

__all__ = [
    "load_dataset",
    "get_dataset_spec",
    "DatasetBundle",
    "DatasetSpec",
]