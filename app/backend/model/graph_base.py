# app/backend/model/graph_base.py
"""
Shared graph-ontology scaffolding for Pydantic v2 domain modules.

Every generated domain module must import from here:

    from .graph_base import Edge, ListEdge, GraphNodeEnum, NodeModel

Design:
- `GraphNodeEnum` marks an enum as a graph node type. Its `node_prefix()`
  strips the trailing "Enum", so enum members map to stable node IDs:
      f"{node_prefix()}__{member_value}"
- `Edge(...)` / `ListEdge(...)` attach `json_schema_extra={"edge_label": ...}`
  to a Field, so the JSON Schema carries the relation name for downstream
  graph builders and prompt generators.
- `NodeModel` is the base for every node model. Subclasses declare
  `model_config = ConfigDict(graph_id_fields=["<pk>"])` and inherit
  `node_id()`, `to_node()`, `to_edges()`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Edge helpers
# =============================================================================

def Edge(
    label: str,
    default: Any = None,
    required: bool = False,
    **kwargs: Any,
):
    """Single-valued graph edge.

    `label` becomes ``json_schema_extra["edge_label"]`` so the relation name
    survives serialization into JSON Schema / LLM prompts.
    """
    if required:
        return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
    return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


def ListEdge(
    label: str,
    default_factory: Any = list,
    **kwargs: Any,
):
    """Multi-valued graph edge (list of node references)."""
    return Field(
        default_factory=default_factory,
        json_schema_extra={"edge_label": label},
        **kwargs,
    )


# =============================================================================
# Node-producing enum base
# =============================================================================

class GraphNodeEnum(str, Enum):
    """Enum whose members are graph node values."""

    @classmethod
    def node_prefix(cls) -> str:
        """`MasterStageEnum` -> `MasterStage` (strips the trailing "Enum")."""
        return cls.__name__.removesuffix("Enum")

    @classmethod
    def member_descriptions(cls) -> dict[str, str]:
        """Override in subclasses: {member_value: 中文说明}."""
        return {}

    @classmethod
    def description_for(cls, value: str) -> Optional[str]:
        """Look up the Chinese description for a member value across subclasses."""
        for sub in GraphNodeEnum.__subclasses__():
            desc = sub.member_descriptions().get(value)
            if desc:
                return desc
        return None

    def node_id(self) -> str:
        """Stable node ID, e.g. ``MasterStage__MasterStage_S1_RPCVerification``."""
        return f"{type(self).node_prefix()}__{self.value}"


# =============================================================================
# Node model base
# =============================================================================

class NodeModel(BaseModel):
    """Base for every node model in a generated ontology.

    Subclasses MUST set::

        model_config = ConfigDict(graph_id_fields=["<pk_field>"])

    The first field is conventionally the primary key, typed as the node's
    ``*Enum``.
    """

    model_config = ConfigDict(graph_id_fields=[])

    # ------------------------------------------------------------------ helpers

    @classmethod
    def graph_id_field(cls) -> str:
        fields = cls.model_config.get("graph_id_fields") or []
        if not fields:
            raise ValueError(f"{cls.__name__} has no graph_id_fields configured")
        return fields[0]

    def node_label(self) -> str:
        """Node type label, e.g. ``"MasterStage"``."""
        return type(self).__name__

    def node_id(self) -> str:
        """Stable node ID built from the primary-key field."""
        value = getattr(self, self.graph_id_field())
        if isinstance(value, GraphNodeEnum):
            return value.node_id()
        return f"{self.node_label()}__{value}"

    # ------------------------------------------------------------------ export

    def to_node(self) -> dict[str, Any]:
        """Return a plain dict describing this node."""
        return {
            "id": self.node_id(),
            "label": self.node_label(),
            "properties": self.model_dump(mode="json"),
        }

    def to_edges(self) -> list[dict[str, Any]]:
        """Yield ``{"from", "to", "label"}`` dicts for every Edge/ListEdge field."""
        out: list[dict[str, Any]] = []
        src = self.node_id()
        for fname, finfo in type(self).model_fields.items():
            extra = finfo.json_schema_extra or {}
            label = extra.get("edge_label")
            if not label:
                continue
            value = getattr(self, fname)
            targets = value if isinstance(value, list) else [value]
            for t in targets:
                if t is None:
                    continue
                tid = t.node_id() if isinstance(t, GraphNodeEnum) else str(t)
                out.append({"from": src, "to": tid, "label": label})
        return out