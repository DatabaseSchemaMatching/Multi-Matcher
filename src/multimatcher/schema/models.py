from __future__ import annotations
from typing import Optional, List, Dict, Union, Literal
from pydantic import BaseModel, Field

class GraphEdge(BaseModel):
    edge_name: str = Field(..., description="Edge label/name")
    source: str = Field(..., description="Source node")
    target: str = Field(..., description="Target node")
    direction: Optional[str] = Field(None, description="directed/undirected")
    edge_properties: Optional[dict] = Field(None, description="Additional edge properties (samples)")

class SchemaContext(BaseModel):
    source_type: Literal["table", "document", "graph"] = Field(...)
    source_name: str = Field(...)

    element_type: Literal["column", "field", "property"] = Field(...)
    element_name: str = Field(...)

    data_type: Optional[str] = Field(None)
    sample_values: Optional[List[Union[str, int, float]]] = Field(None)
    stat_summary: Optional[Dict] = Field(None)

    graph_edges: Optional[List[GraphEdge]] = Field(None)