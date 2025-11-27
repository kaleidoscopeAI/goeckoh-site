import networkx as nx
from ..core.semantic_mapper import HolographicSemanticMapper

class MemoryGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.mapper = HolographicSemanticMapper()

    def add_node(self, node):
        self.graph.add_node(node.node_id, embedding=node.embedding, energy=node.energy)

    def add_edge(self, node_a, node_b, weight=None):
        if weight is None:
            weight = np.linalg.norm(node_a.embedding - node_b.embedding)
        self.graph.add_edge(node_a.node_id, node_b.node_id, weight=weight)

    def neighbors_embeddings(self, node_id):
        neighbors = self.graph.neighbors(node_id)
        return [self.graph.nodes[n]['embedding'] for n in neighbors]
