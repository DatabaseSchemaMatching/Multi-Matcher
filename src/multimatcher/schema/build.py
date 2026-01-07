from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate

from .models import GraphEdge, SchemaContext
from .io import read_csv_clean, load_json_lines
from .json_flatten import flatten_dict
from .stats import get_data_type, stat_compute, extract_unique_values, sample_up_to_k, to_python_types

schema_description_prompt = PromptTemplate.from_template(
    "source_type:{source_type},source_name:{source_name},element_type:{element_type},"
    "element_name:{element_name},data_type:{data_type},sample_values:{sample_values},"
    "stat_summary:{stat_summary},graph_edges:{graph_edges}"
)

def graph_edge_generation(
    path: str,
    fname: str,
    direction: str,
    source_node: str,
    target_node: str,
) -> GraphEdge:
    direction = direction.lower()
    source_node = source_node.lower()
    target_node = target_node.lower()

    file_path = os.path.join(path, fname)
    data_name = fname.rsplit(".", 1)[0]

    df = read_csv_clean(
        file_path,
        encodings=["utf-8", "cp949", "latin1"],
        engine="python",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="warn",
    ).infer_objects()

    edge_property: Dict[str, List[Any]] = {}
    for col in df.columns:
        unique_vals = extract_unique_values(df[col].dropna())
        edge_property[col] = sample_up_to_k(unique_vals, k=5)

    return GraphEdge(
        edge_name=data_name,
        source=source_node,
        target=target_node,
        direction=direction,
        edge_properties=edge_property,
    )

def schema_generation(
    path: str,
    fname: str,
    model: str,
    graph_edges_input: Optional[List[GraphEdge]] = None,
) -> List[SchemaContext]:
    model = model.lower()
    file_path = os.path.join(path, fname)
    data_name = fname.rsplit(".", 1)[0]

    if model == "document":
        df = pd.DataFrame([flatten_dict(rec) for rec in load_json_lines(file_path)])
        element_type = "field"
        source_type = "document"
        graph_edges = None

    elif model in {"table", "graph"}:
        df = read_csv_clean(
            file_path,
            encodings=["utf-8", "cp949", "latin1"],
            engine="python",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="warn",
        ).infer_objects()

        source_type = model
        element_type = "column" if model == "table" else "property"
        graph_edges = [] if model == "graph" else None
        if model == "graph" and graph_edges_input:
            graph_edges = graph_edges_input
    else:
        raise ValueError("Invalid model type. Must be 'table', 'document', or 'graph'.")

    contexts: List[SchemaContext] = []
    for col in df.columns:
        unique_vals = extract_unique_values(df[col].dropna())
        samples = sample_up_to_k(unique_vals, k=5)

        ctx = SchemaContext(
            source_type=source_type,
            source_name=data_name,
            element_type=element_type,
            element_name=col,
            data_type=get_data_type(df[col]),
            sample_values=samples,
            stat_summary=stat_compute(df[col]) if pd.api.types.is_numeric_dtype(df[col]) else {},
            graph_edges=graph_edges,
        )
        contexts.append(ctx)

    return contexts

def render_prompt_from_context(ctx: SchemaContext) -> str:
    payload = ctx.model_dump(exclude_none=True)

    payload["sample_values"] = json.dumps(to_python_types(payload.get("sample_values", [])), ensure_ascii=False)
    payload["stat_summary"]  = json.dumps(to_python_types(payload.get("stat_summary", {})), ensure_ascii=False)
    payload["graph_edges"]   = json.dumps(to_python_types(payload.get("graph_edges", [])), ensure_ascii=False)

    return schema_description_prompt.format(**payload)