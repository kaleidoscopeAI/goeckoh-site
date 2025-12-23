def __init__(self):
    self.nodes = {}
    self.edges = defaultdict(list)

def add_node(self, node_id: str, data: Dict = None):
    """Adds a node to the network."""
    if node_id not in self.nodes:
        self.nodes[node_id] = data if data else {}

def add_edge(self, node1: str, node2: str, weight: float):
    """Adds a weighted edge between two nodes."""
    self.edges[node1].append((node2, weight))

def observe(self, concept: Dict):
    """Updates the network based on new observations."""
    if 'id' in concept:
        node_id = concept['id']
        if node_id in self.nodes:
            self.nodes[node_id]['strength'] = self.nodes[node_id].get('strength', 0) + 0.1
            for neighbor, weight in self.edges.get(node_id, []):
              self.nodes[neighbor]['strength'] = self.nodes[neighbor].get('strength', 0) + weight * 0.05

def get_probabilities(self) -> Dict[str, float]:
    """Returns the probabilities of nodes in the network."""
    probabilities = {}
    for node_id, data in self.nodes.items():
        probabilities [node_id] = data.get('strength', 0.0)
    return probabilities

