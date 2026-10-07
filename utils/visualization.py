"""utils/visualization.py — Convert graph / retrieval payloads to ECharts.

Pure functions: no FastAPI, no global state.  Given the same input they
always produce the same output, which makes them unit-testable without
spinning up the app.
"""
from __future__ import annotations

import ast
import json
import os
from typing import Dict, List

from utils.logger import logger


# ===========================================================================
# Graph visualisation
# ===========================================================================
def prepare_graph_visualization(graph_path: str) -> Dict:
    """Load a graph JSON file and convert it to ECharts-friendly form."""
    empty = {"nodes": [], "links": [], "categories": [], "stats": {}}
    try:
        if not os.path.exists(graph_path):
            return empty
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        if isinstance(graph_data, list):
            return convert_graphrag_format(graph_data)
        if isinstance(graph_data, dict) and "nodes" in graph_data:
            return convert_standard_format(graph_data)
        return empty
    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return empty


def _make_category_styles(names) -> List[Dict]:
    names = list(names)
    return [
        {
            "name": cat,
            "itemStyle": {
                "color": f"hsl({i * 360 / max(len(names), 1)}, 70%, 60%)"
            },
        }
        for i, cat in enumerate(names)
    ]


def _register_node(nodes_dict: Dict, node_props: Dict, fallback_label: str) -> str:
    """Register a node keyed by `properties.name`; return the key (or '')."""
    name = node_props.get("name", "") if node_props else ""
    if not name:
        return ""
    if name not in nodes_dict:
        category = node_props.get("class") or node_props.get(
            "schema_type", fallback_label
        )
        nodes_dict[name] = {
            "id": name,
            "name": str(name)[:200],
            "category": category,
            "symbolSize": 25,
            "properties": node_props,
        }
    return name


def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format."""
    nodes_dict: Dict[str, Dict] = {}
    links: List[Dict] = []

    for item in graph_data:
        if not isinstance(item, dict):
            continue

        start_node = item.get("start_node") or {}
        end_node = item.get("end_node") or {}
        relation = item.get("relation", "related_to")

        start_id = _register_node(
            nodes_dict,
            start_node.get("properties", {}) or {},
            start_node.get("label", "entity"),
        )
        end_id = _register_node(
            nodes_dict,
            end_node.get("properties", {}) or {},
            end_node.get("label", "entity"),
        )
        if start_id and end_id:
            links.append(
                {"source": start_id, "target": end_id, "name": relation, "value": 1}
            )

    nodes = list(nodes_dict.values())
    categories = _make_category_styles({n["category"] for n in nodes})
    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert {nodes: [], edges: []} to ECharts format."""
    raw_nodes = graph_data.get("nodes", []) or []
    raw_edges = graph_data.get("edges", []) or []

    node_types = {n.get("type", "entity") for n in raw_nodes}
    categories = _make_category_styles(node_types)

    nodes = []
    for node in raw_nodes:
        attrs = node.get("attributes", []) or []
        nodes.append({
            "id": node.get("id", ""),
            "name": str(node.get("name", node.get("id", "")))[:200],
            "category": node.get("type", "entity"),
            "value": len(attrs),
            "symbolSize": min(max(len(attrs) * 3 + 15, 15), 40),
            "attributes": attrs,
        })

    links = [
        {
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "name": e.get("relation", "related_to"),
            "value": e.get("weight", 1),
        }
        for e in raw_edges
    ]

    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(raw_nodes),
            "total_edges": len(raw_edges),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


# ===========================================================================
# Retrieval-side visualisations
# ===========================================================================
def prepare_subquery_visualization(
    sub_questions: List[Dict], reasoning_steps: List[Dict]
) -> Dict:
    nodes = [{
        "id": "original",
        "name": "Original Question",
        "category": "question",
        "symbolSize": 40,
    }]
    links: List[Dict] = []
    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append({
            "id": sub_id,
            "name": (sub_q.get("sub-question", "") or "")[:200] + "...",
            "category": "sub_question",
            "symbolSize": 30,
        })
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})
    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}},
        ],
    }


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    """Build a tiny graph from the *formatted* triple strings.

    The retrieval layer formats triples as ``"[(h, r, t)]"`` (a Python
    literal); we recover the tuple with ``ast.literal_eval`` and lay it out
    for ECharts.
    """
    nodes: List[Dict] = []
    links: List[Dict] = []
    seen: set = set()

    for triple in triples[:10]:
        if not (isinstance(triple, str)
                and triple.startswith("[") and triple.endswith("]")):
            continue
        try:
            parts = ast.literal_eval(triple)
        except Exception:
            continue
        if len(parts) != 3:
            continue

        source, relation, target = parts
        for entity in (source, target):
            if entity not in seen:
                seen.add(entity)
                nodes.append({
                    "id": str(entity),
                    "name": str(entity)[:200],
                    "category": "entity",
                    "symbolSize": 20,
                })
        links.append({
            "source": str(source),
            "target": str(target),
            "name": str(relation),
        })

    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}],
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    steps_data = [{
        "step": i + 1,
        "type": s.get("type", "unknown"),
        "question": (s.get("question", "") or "")[:50],
        "triples_count": s.get("triples_count", 0),
        "chunks_count": s.get("chunks_count", 0),
        "processing_time": s.get("processing_time", 0),
    } for i, s in enumerate(reasoning_steps)]
    return {
        "steps": steps_data,
        "timeline": [s["processing_time"] for s in steps_data],
    }