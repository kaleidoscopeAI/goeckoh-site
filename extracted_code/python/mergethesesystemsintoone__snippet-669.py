def __init__(self, node_id, specialization="general", energy=100):
    self.node_id = node_id
    self.specialization = specialization
    self.energy = energy
    self.memory = []  # Stores pharmaceutical insights
    self.connections = []

def ingest_insight(self, insight: Dict):
    """Processes a pharmaceutical insight."""
    relevance_score = self.calculate_relevance(insight)
    if relevance_score >= 0.5:
        self.learn(insight, relevance_score)

def calculate_relevance(self, insight: Dict) -> float:
    """Calculates the relevance of an insight."""
    if self.specialization in insight.get("tags", []):
        return 0.8
    return 0.4

def learn(self, insight: Dict, relevance_score: float):
    """Learns from a pharmaceutical insight."""
    if len(self.memory) >= 100:
        self.memory.pop(0)  # Remove oldest insight
    self.memory.append(insight)
    self.energy -= 5  # Learning costs energy

def connect(self, other_node):
    """Connects to another knowledge node."""
    self.connections.append(other_node)

def share_insights(self):
    """Shares pharmaceutical insights with connected nodes."""
    for node in self.connections:
        node.receive_insights(self.memory[-1:])

def receive_insights(self, insights: List[Dict]):
    """Receives insights from connected nodes."""
    for insight in insights:
        if insight not in self.memory:
            self.memory.append(insight)

def status(self) -> Dict:
    """Returns the current status of the node."""
    return {
        "Node ID": self.node_id,
        "Energy": self.energy,
        "Specialization": self.specialization,
        "Memory Count": len(self.memory),
    }

