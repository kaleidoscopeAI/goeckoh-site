"""
Represents the memory of the system as a graph, storing data, insights, and their relationships.
"""
def __init__(self):
    self.graph = nx.DiGraph()

def add_data(self, data_id: str, data: Dict[str, Any]):
    """
    Adds a data node to the memory graph.
    """
    self.graph.add_node(data_id, type="data", **data)

def add_insight(self, insight_id: str, insight: Dict[str, Any]):
    """
    Adds an insight node to the memory graph.
    """
    self.graph.add_node(insight_id, type="insight", **insight)

def add_relationship(self, source_id: str, target_id: str, relationship: Dict[str, Any]):
    """
    Adds a directed edge between two nodes in the memory graph, representing a relationship.
    """
    self.graph.add_edge(source_id, target_id, **relationship)

def get_data(self, data_id: str) -> Dict[str, Any]:
    """
    Retrieves data associated with a given data ID.
    """
    if data_id in self.graph.nodes:
        return self.graph.nodes[data_id]
    return None

def get_insight(self, insight_id: str) -> Dict[str, Any]:
    """
    Retrieves an insight associated with a given insight ID.
    """
    if insight_id in self.graph.nodes:
        return self.graph.nodes[insight_id]
    return None

def get_related_nodes(self, node_id: str, relationship_type: str) -> List[str]:
    """
    Retrieves nodes related to a given node based on relationship type.
    """
    related_nodes = []
    for neighbor in self.graph.neighbors(node_id):
        if self.graph.edges[node_id, neighbor].get('type') == relationship_type:
            related_nodes.append(neighbor)
    return related_nodes

def remove_node(self, node_id: str):
    """
    Removes a node from the memory graph.
    """
    if node_id in self.graph.nodes:
        self.graph.remove_node(node_id)

def clear_graph(self):
    """
    Clears all nodes and edges from the memory graph.
    """
    self.graph.clear()




