# src/multimatcher/datasets/registry.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from multimatcher.schema.build import schema_generation, graph_edge_generation
from multimatcher.schema.models import SchemaContext, GraphEdge


# -----------------------------
# Specs
# -----------------------------
@dataclass(frozen=True)
class GraphEdgeSpec:
    graph_dir: str
    filename: str
    direction: str
    source_node: str
    target_node: str


@dataclass(frozen=True)
class DatasetSpec:
    """
    DatasetSpec stores dataset-relative structure. It should not hardcode
    machine-specific absolute paths.

    - single-dataset: table_dir/document_dir/graph_dir are subdirs under base_dir
    - cross-dataset: loader can special-case and use multiple subdirs
    """
    name: str

    # base directory for this dataset (absolute, resolved in get_dataset_spec)
    base_dir: str

    # dirs (relative to base_dir, or None if special-case loader uses multiple dirs)
    table_dir: Optional[str]
    document_dir: Optional[str]
    graph_dir: Optional[str]
    gt_dir: str  # usually base_dir

    # files
    table_files: Sequence[str]
    document_files: Sequence[str]
    graph_node_files: Sequence[str]  # node/property files

    # graph edges
    graph_edges: Sequence[GraphEdgeSpec]

    # graph node -> which edges to attach (by edge filename)
    graph_node_edge_map: Dict[str, Sequence[str]]

    # evaluation
    grouping_candidates_csv: str = "grouping_candidates.csv"
    group_csv: str = "group.csv"


@dataclass(frozen=True)
class DatasetBundle:
    spec: DatasetSpec
    all_schema_contexts: List[SchemaContext]
    grouping_candidates_path: str
    group_path: str


# -----------------------------
# Helpers
# -----------------------------
def _join(dirpath: str, fname: str) -> str:
    return os.path.join(dirpath, fname)


def _require_dir(path: str, hint: str) -> None:
    if not os.path.isdir(path):
        raise RuntimeError(f"Directory not found: {path}\nHint: {hint}")


def _require_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}\nHint: {hint}")


def _build_graph_edges(spec: DatasetSpec) -> Dict[str, GraphEdge]:
    """
    Returns mapping: edge_filename -> GraphEdge
    """
    edge_objs: Dict[str, GraphEdge] = {}
    for e in spec.graph_edges:
        edge_objs[e.filename] = graph_edge_generation(
            path=e.graph_dir,
            fname=e.filename,
            direction=e.direction,
            source_node=e.source_node,
            target_node=e.target_node,
        )
    return edge_objs


def _build_schema_contexts(spec: DatasetSpec) -> List[SchemaContext]:
    # 1) graph edges 먼저 생성
    edge_map = _build_graph_edges(spec)

    contexts: List[SchemaContext] = []

    # 2) tables
    if spec.table_dir:
        for f in spec.table_files:
            contexts += schema_generation(spec.table_dir, f, "table")

    # 3) documents
    if spec.document_dir:
        for f in spec.document_files:
            contexts += schema_generation(spec.document_dir, f, "document")

    # 4) graph nodes (node/property files)
    if spec.graph_dir:
        for node_file in spec.graph_node_files:
            edge_files = spec.graph_node_edge_map.get(node_file, [])
            edges_to_attach = [edge_map[ef] for ef in edge_files if ef in edge_map]
            contexts += schema_generation(
                spec.graph_dir,
                node_file,
                "graph",
                edges_to_attach if edges_to_attach else None,
            )

    return contexts


# -----------------------------
# Public API
# -----------------------------
def get_dataset_spec(name: str, data_root: str) -> DatasetSpec:
    """
    Convert a dataset alias to a DatasetSpec using a user-provided data_root.

    data_root example:
      - C:\\Users\\...\\dataset
      - /home/user/datasets
      - (repo-relative resolved already in run_dataset.py)
    """
    key = name.strip().lower()
    root = os.path.abspath(data_root)

    def base_of(folder: str) -> str:
        return _join(root, folder)

    if key in ("m2bench-ecommerce", "m2bench_ecommerce", "ecommerce", "m2e"):
        base = base_of("M2Bench_Ecommerce")
        return DatasetSpec(
            name="M2Bench E-commerce",
            base_dir=base,
            table_dir=_join(base, "table"),
            document_dir=_join(base, "document"),
            graph_dir=_join(base, "graph"),
            gt_dir=base,
            table_files=("customer.csv", "product.csv", "brand.csv"),
            document_files=("order.json", "review.json"),
            graph_node_files=("person.csv", "hashtag.csv"),
            graph_edges=(
                GraphEdgeSpec(_join(base, "graph"), "person_interestedIn_tag.csv", "directed", "person", "hashtag"),
                GraphEdgeSpec(_join(base, "graph"), "person_follows_person.csv", "directed", "person", "person"),
            ),
            graph_node_edge_map={
                "person.csv": ("person_interestedIn_tag.csv", "person_follows_person.csv"),
                "hashtag.csv": ("person_interestedIn_tag.csv",),
            },
        )

    if key in ("m2bench-healthcare", "m2bench_healthcare", "healthcare", "m2h"):
        base = base_of("M2Bench_Healthcare")
        return DatasetSpec(
            name="M2Bench Healthcare",
            base_dir=base,
            table_dir=_join(base, "table"),
            document_dir=_join(base, "document"),
            graph_dir=_join(base, "graph"),
            gt_dir=base,
            table_files=("diagnosis.csv", "patient.csv", "prescription.csv"),
            document_files=("drug.json",),
            graph_node_files=("diseases.csv",),
            graph_edges=(
                GraphEdgeSpec(_join(base, "graph"), "is_a.csv", "directed", "source_id", "destination_id"),
            ),
            graph_node_edge_map={
                "diseases.csv": ("is_a.csv",),
            },
        )

    if key in ("unibench", "uni"):
        base = base_of("Unibench")
        return DatasetSpec(
            name="UniBench",
            base_dir=base,
            table_dir=_join(base, "table"),
            document_dir=_join(base, "document"),
            graph_dir=_join(base, "graph"),
            gt_dir=base,
            table_files=("customer.csv", "feedback.csv", "product.csv", "vendor.csv"),
            document_files=("order.json",),
            graph_node_files=("post.csv",),
            graph_edges=(),
            graph_node_edge_map={
                "post.csv": (),
            },
        )

    if key in ("m2e-unibench", "m2e_unibench", "cross", "cross-dataset", "m2e<->uni"):
        base = base_of("M2E_Unibench")
        # special-case loader (multiple dirs)
        return DatasetSpec(
            name="M2Bench E-commerce <-> UniBench",
            base_dir=base,
            table_dir=None,
            document_dir=None,
            graph_dir=None,
            gt_dir=base,
            table_files=(),
            document_files=(),
            graph_node_files=(),
            graph_edges=(),
            graph_node_edge_map={},
        )

    raise ValueError(f"Unknown dataset name: {name}")


def load_dataset(name: str, data_root: str) -> DatasetBundle:
    """
    Load dataset by name using the provided data_root.
    run_dataset.py passes data_root resolved relative to repo root,
    so callers can avoid absolute paths.
    """
    spec = get_dataset_spec(name, data_root=data_root)

    # Basic existence checks for nicer errors
    _require_dir(
        spec.gt_dir,
        hint=f"Check your data_root and dataset folder name under it (dataset={name}, data_root={data_root}).",
    )
    _require_file(_join(spec.gt_dir, spec.grouping_candidates_csv), hint="Missing grouping_candidates.csv in gt_dir.")
    _require_file(_join(spec.gt_dir, spec.group_csv), hint="Missing group.csv in gt_dir.")

    # GT paths
    grouping_candidates_path = _join(spec.gt_dir, spec.grouping_candidates_csv)
    group_path = _join(spec.gt_dir, spec.group_csv)

    # Special case: cross-dataset
    if spec.name.lower() == "m2bench e-commerce <-> unibench":
        base = spec.gt_dir

        # dirs
        table_dir1 = _join(base, "table1")
        doc_dir1 = _join(base, "document1")
        graph_dir1 = _join(base, "graph1")

        table_dir2 = _join(base, "table2")
        doc_dir2 = _join(base, "document2")
        graph_dir2 = _join(base, "graph2")

        # existence checks
        for d in (table_dir1, doc_dir1, graph_dir1, table_dir2, doc_dir2, graph_dir2):
            _require_dir(d, hint="Cross-dataset expects table1/document1/graph1 and table2/document2/graph2.")

        # files
        table_files1 = ("customer1.csv", "product1.csv", "brand1.csv")
        doc_files1 = ("order1.json", "review1.json")
        graph_nodes1 = ("person1.csv", "hashtag1.csv")

        table_files2 = ("customer2.csv", "feedback2.csv", "product2.csv", "vendor2.csv")
        doc_files2 = ("order2.json",)
        graph_nodes2 = ("post2.csv",)

        # edges (graph1 only, per your notebook)
        e_interested = graph_edge_generation(graph_dir1, "person_interestedin_tag1.csv", "directed", "person", "hashtag")
        e_follows = graph_edge_generation(graph_dir1, "person_follows_person1.csv", "directed", "person", "person")

        # contexts
        contexts: List[SchemaContext] = []

        # side 1
        for f in table_files1:
            contexts += schema_generation(table_dir1, f, "table")
        for f in doc_files1:
            contexts += schema_generation(doc_dir1, f, "document")
        contexts += schema_generation(graph_dir1, "person1.csv", "graph", [e_interested, e_follows])
        contexts += schema_generation(graph_dir1, "hashtag1.csv", "graph", [e_interested])

        # side 2
        for f in table_files2:
            contexts += schema_generation(table_dir2, f, "table")
        for f in doc_files2:
            contexts += schema_generation(doc_dir2, f, "document")
        for f in graph_nodes2:
            contexts += schema_generation(graph_dir2, f, "graph")

        return DatasetBundle(
            spec=spec,
            all_schema_contexts=contexts,
            grouping_candidates_path=grouping_candidates_path,
            group_path=group_path,
        )

    # default: single dataset
    # existence checks for single dataset dirs
    if spec.table_dir:
        _require_dir(spec.table_dir, hint="Missing table/ directory under dataset folder.")
        for f in spec.table_files:
            _require_file(_join(spec.table_dir, f), hint="Check table_files list vs actual filenames.")

    if spec.document_dir:
        _require_dir(spec.document_dir, hint="Missing document/ directory under dataset folder.")
        for f in spec.document_files:
            _require_file(_join(spec.document_dir, f), hint="Check document_files list vs actual filenames.")

    if spec.graph_dir:
        _require_dir(spec.graph_dir, hint="Missing graph/ directory under dataset folder.")
        for f in spec.graph_node_files:
            _require_file(_join(spec.graph_dir, f), hint="Check graph_node_files list vs actual filenames.")
        for e in spec.graph_edges:
            _require_file(_join(e.graph_dir, e.filename), hint="Check graph_edges list vs actual filenames.")

    contexts = _build_schema_contexts(spec)
    return DatasetBundle(
        spec=spec,
        all_schema_contexts=contexts,
        grouping_candidates_path=grouping_candidates_path,
        group_path=group_path,
    )