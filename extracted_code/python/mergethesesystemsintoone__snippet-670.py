def __init__(self):
    self.nodes = []
    self.network = nx.Graph()

def add_node(self, node: KnowledgeNode):
    """Adds a knowledge node to the graph."""
    self.nodes.append(node)
    self.network.add_node(node.node_id, energy=node.energy, specialization=node.specialization)

def connect_nodes(self, node_id_1: str, node_id_2: str):
    """Connects two knowledge nodes in the graph."""
    node1 = next((n for n in self.nodes if n.node_id == node_id_1), None)
    node2 = next((n for n in self.nodes if n.node_id == node_id_2), None)
    if node1 and node2:
        node1.connect(node2)
        self.network.add_edge(node_id_1, node_id_2)

def distribute_insights(self, insights: List[Dict]):
    """Distributes pharmaceutical insights across the graph."""
    for insight in insights:
        for node in self.nodes:
            node.ingest_insight(insight)

def propagate_insights(self):
    """Propagates insights across the network."""
    for node in self.nodes:
        node.share_insights()

def summarize_knowledge(self) -> Dict:
    """Summarizes the collective pharmaceutical knowledge in the graph."""
    summary = {}
    for node in self.nodes:
        summary[node.node_id] = {
            "Specialization": node.specialization,
            "Insights": len(node.memory),
            "Sample Insights": node.memory[:3],
        }
    return summary

def visualize(self):
    """Visualizes the pharmaceutical knowledge graph."""
    pos = nx.spring_layout(self.network)
    labels = {
        node: f"{node}\nEnergy: {data['energy']}\nSpecialization: {data['specialization']}"
        for node, data in self.network.nodes(data=True)
    }
    nx.draw(self.network, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue")


