import networkx as nx
import json

from utils.logger import logger


def load_graph_from_json(input_path: str) -> nx.MultiDiGraph:
    """
    Load a knowledge graph from JSON format
    
    Expected JSON format:
    [
        {
            "start_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            },
            "relation": "relation_type",
            "end_node": {
                "label": "entity", 
                "properties": {"name": "Entity Name", "description": "..."}
            }
        }
    ]
    """
    graph = nx.MultiDiGraph()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        relationships = json.load(f)
    
    # Track nodes to avoid duplicates and assign consistent IDs
    node_mapping = {}  # (label, name) -> node_id
    node_counter = 0
    
    for rel in relationships:
        start_node_data = rel["start_node"]
        end_node_data = rel["end_node"]
        relation = rel["relation"]
        
        # Create unique key for start node - ensure name is a string
        start_name = start_node_data["properties"].get("name", "")
        if isinstance(start_name, list):
            start_name = ", ".join(str(item) for item in start_name)
        elif not isinstance(start_name, str):
            start_name = str(start_name)
        
        start_key = (start_node_data["label"], start_name)
        if start_key not in node_mapping:
            node_id = f"{start_node_data['label']}_{node_counter}"
            node_mapping[start_key] = node_id
            node_counter += 1
            
            # Add node with all properties
            node_attrs = {
                "label": start_node_data["label"],
                "properties": start_node_data["properties"]
            }
            
            # Add level based on label
            if start_node_data["label"] == "attribute":
                node_attrs["level"] = 1
            elif start_node_data["label"] == "entity":
                node_attrs["level"] = 2
            elif start_node_data["label"] == "keyword":
                node_attrs["level"] = 3
            elif start_node_data["label"] == "community":
                node_attrs["level"] = 4
            else:
                node_attrs["level"] = 2  # Default level
            
            graph.add_node(node_id, **node_attrs)
        
        # Create unique key for end node - ensure name is a string
        end_name = end_node_data["properties"].get("name", "")
        if isinstance(end_name, list):
            end_name = ", ".join(str(item) for item in end_name)
        elif not isinstance(end_name, str):
            end_name = str(end_name)
        
        end_key = (end_node_data["label"], end_name)
        if end_key not in node_mapping:
            node_id = f"{end_node_data['label']}_{node_counter}"
            node_mapping[end_key] = node_id
            node_counter += 1
            
            # Add node with all properties
            node_attrs = {
                "label": end_node_data["label"],
                "properties": end_node_data["properties"]
            }
            
            # Add level based on label
            if end_node_data["label"] == "attribute":
                node_attrs["level"] = 1
            elif end_node_data["label"] == "entity":
                node_attrs["level"] = 2
            elif end_node_data["label"] == "keyword":
                node_attrs["level"] = 3
            elif end_node_data["label"] == "community":
                node_attrs["level"] = 4
            else:
                node_attrs["level"] = 2  # Default level
            
            graph.add_node(node_id, **node_attrs)
        
        # Add edge
        start_id = node_mapping[start_key]
        end_id = node_mapping[end_key]
        graph.add_edge(start_id, end_id, relation=relation)
    
    return graph


def save_graph_to_json(graph: nx.MultiDiGraph, output_path: str):
    """
    Save a knowledge graph to JSON format
    
    Output format:
    [
        {
            "start_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            },
            "relation": "relation_type", 
            "end_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            }
        }
    ]
    """
    output = []
    
    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        
        relationship = {
            "start_node": {
                "label": u_data["label"],
                "properties": u_data["properties"],
            },
            "relation": data["relation"],
            "end_node": {
                "label": v_data["label"],
                "properties": v_data["properties"],
            },
        }
        output.append(relationship)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# Legacy function for backward compatibility
def load_graph(input_path: str) -> nx.MultiDiGraph:
    """
    Load graph from either JSON or GraphML format (legacy support)
    """
    if input_path.endswith('.json'):
        return load_graph_from_json(input_path)
    elif input_path.endswith('.graphml'):
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
                logger.warning(f"Warning: Could not parse properties for node {node_id}")
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
    
    return graph_data


def save_graph(graph: nx.MultiDiGraph, output_path: str):
    """
    Save graph to either JSON or GraphML format based on file extension
    """
    if output_path.endswith('.json'):
        save_graph_to_json(graph, output_path)
    elif output_path.endswith('.graphml'):
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