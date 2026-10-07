# utils/graph_processor.py
import networkx as nx
import json

from utils.logger import logger


def _coerce_chunk_ids(raw) -> list:
    """Normalise a chunk_id field into a list of non-empty strings.

    Accepts list/tuple/set, a bare string, or None.  Always returns a
    fresh list so callers can mutate safely.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, (list, tuple, set)):
        return [str(c) for c in raw if c]
    return []


def load_graph_from_json(input_path: str) -> nx.MultiDiGraph:
    """Load a knowledge graph from JSON.

    Preserves BOTH node-level and edge-level `chunk_id` provenance written
    by `kt_gen.format_output()`.  Edge provenance is the load-bearing
    signal for retrieval — dropping it degrades recall silently.
    """
    graph = nx.MultiDiGraph()

    with open(input_path, "r", encoding="utf-8") as f:
        relationships = json.load(f)

    node_mapping = {}
    node_counter = 0

    def _ensure_node(node_data: dict) -> str:
        nonlocal node_counter
        name = node_data["properties"].get("name", "")
        if isinstance(name, list):
            name = ", ".join(str(item) for item in name)
        elif not isinstance(name, str):
            name = str(name)

        key = (node_data["label"], name)
        if key in node_mapping:
            return node_mapping[key]

        node_id = f"{node_data['label']}_{node_counter}"
        node_mapping[key] = node_id
        node_counter += 1

        label = node_data["label"]
        level = {
            "attribute": 1,
            "entity": 2,
            "keyword": 3,
            "community": 4,
        }.get(label, 2)

        graph.add_node(
            node_id,
            label=label,
            properties=node_data["properties"],
            level=level,
        )
        return node_id

    for rel in relationships:
        start_id = _ensure_node(rel["start_node"])
        end_id = _ensure_node(rel["end_node"])

        edge_chunk_ids = _coerce_chunk_ids(rel.get("chunk_id"))

        graph.add_edge(
            start_id,
            end_id,
            relation=rel["relation"],
            chunk_id=edge_chunk_ids,
        )

    return graph


def save_graph_to_json(graph: nx.MultiDiGraph, output_path: str):
    """Mirror of `load_graph_from_json` — preserves edge chunk_id.

    Keeps round-trip fidelity: load → save → load produces an identical
    graph including all provenance.
    """
    output = []

    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]

        output.append(
            {
                "start_node": {
                    "label": u_data["label"],
                    "properties": u_data["properties"],
                },
                "relation": data["relation"],
                "chunk_id": _coerce_chunk_ids(data.get("chunk_id")),
                "end_node": {
                    "label": v_data["label"],
                    "properties": v_data["properties"],
                },
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# Legacy function for backward compatibility
def load_graph(input_path: str) -> nx.MultiDiGraph:
    """
    Load graph from either JSON or GraphML format (legacy support)
    """
    if input_path.endswith(".json"):
        return load_graph_from_json(input_path)
    elif input_path.endswith(".graphml"):
        return load_graph_from_graphml(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")


def load_graph_from_graphml(input_path: str) -> nx.MultiDiGraph:
    """
    Load graph from GraphML format (legacy function)
    """
    graph_data = nx.read_graphml(input_path)

    for node_id, data in graph_data.nodes(data=True):
        # Handle properties (d1)
        if "d1" in data:
            try:
                data["properties"] = json.loads(data["d1"])
                del data["d1"]
            except json.JSONDecodeError:
                logger.warning(
                    f"Warning: Could not parse properties for node {node_id}"
                )
                data["properties"] = {"name": str(data["d1"])}
                del data["d1"]

        # Handle level (d2)
        if "d2" in data:
            try:
                data["level"] = int(data["d2"])
                del data["d2"]
            except (ValueError, TypeError):
                data["level"] = 2  # Default level if conversion fails
                del data["d2"]

        # Handle label (d0)
        if "d0" in data:
            data["label"] = str(data["d0"])
            del data["d0"]

    for u, v, data in graph_data.edges(data=True):
        # Handle relation (d3)
        if "d3" in data:
            data["relation"] = str(data["d3"]).strip('"')
            del data["d3"]
        # GraphML serializes complex attributes; chunk_id may arrive as JSON string.
        if "chunk_id" in data and isinstance(data["chunk_id"], str):
            try:
                data["chunk_id"] = json.loads(data["chunk_id"])
            except json.JSONDecodeError:
                data["chunk_id"] = _coerce_chunk_ids(data["chunk_id"])

    return graph_data


def save_graph(graph: nx.MultiDiGraph, output_path: str):
    """
    Save graph to either JSON or GraphML format based on file extension
    """
    if output_path.endswith(".json"):
        save_graph_to_json(graph, output_path)
    elif output_path.endswith(".graphml"):
        save_graph_to_graphml(graph, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")


def save_graph_to_graphml(graph: nx.MultiDiGraph, output_path: str):
    """
    Save graph to GraphML format (legacy function)
    """
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()

    for n, data in graph_copy.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, dict):
                graph_copy.nodes[n][k] = json.dumps(v, ensure_ascii=False)

    for u, v, data in graph_copy.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, dict):
                graph_copy.edges[u, v][k] = json.dumps(v, ensure_ascii=False)

    nx.write_graphml(graph_copy, output_path)
