"""
Represents the memory of the system as a graph, storing data and their relationships.
"""
def __init__(self):
    self.graph = nx.Graph()

def add_node(self, node_id: str, node_data: Dict[str, Any]):
    """
    Adds a node to the memory graph.

    :param node_id: Unique identifier for the node.
    :param node_data: Data associated with the node.
    """
    self.graph.add_node(node_id, **node_data)

def add_edge(self, node1_id: str, node2_id: str, relationship: Dict[str, Any]):
    """
    Adds an edge between two nodes in the memory graph.

    :param node1_id: ID of the first node.
    :param node2_id: ID of the second node.
    :param relationship: Data associated with the edge (relationship).
    """
    self.graph.add_edge(node1_id, node2_id, **relationship)

def get_node_data(self, node_id: str) -> Dict[str, Any]:
    """
    Retrieves data associated with a node.

    :param node_id: ID of the node.
    :return: Data associated with the node.
    """
    return self.graph.nodes[node_id]

def get_related_nodes(self, node_id: str, relationship_type: str) -> list:
    """
    Retrieves nodes related to a given node based on relationship type.

    :param node_id: ID of the node.
    :param relationship_type: Type of relationship to filter by.
    :return: List of nodes connected by the specified relationship type.
    """
    related_nodes = []
    for neighbor in self.graph.neighbors(node_id):
        if self.graph[node_id][neighbor].get('relationship_type') == relationship_type:
            related_nodes.append(neighbor)
    return related_nodes

def visualize_graph(self):
    """
    Visualizes the memory graph.
    """
    nx.draw(self.graph, with_labels=True)


