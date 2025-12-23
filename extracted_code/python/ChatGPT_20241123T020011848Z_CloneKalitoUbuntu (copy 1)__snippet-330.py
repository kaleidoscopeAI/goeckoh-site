def __init__(self, node_id, parent_id=None, dna=None, knowledge_base=None):
    self.node_id = node_id
    self.parent_id = parent_id  # Track the parent node
    self.dna = dna if dna is not None else np.random.rand(5)  # Node DNA
    self.knowledge_base = knowledge_base if knowledge_base else {}  # Knowledge stored as {key: value}
    self.resources = {"memory": 0.5, "energy": 1.0}  # Initial resources
    self.thresholds = {"memory": 0.8, "energy": 0.2}  # Replication thresholds
    self.synapse_overlay = defaultdict(float)  # Connections to other nodes with relevance weights

def learn(self, key, value):
    """Simulate learning and storing information."""
    self.knowledge_base[key] = value
    self.resources["memory"] += 0.1  # Simulate memory usage
    self.resources["energy"] -= 0.05  # Simulate energy expenditure
    print(f"Node {self.node_id}: Learned - {key}: {value}")

def replicate(self, node_id_counter):
    """Replicate the node with inherited DNA and partial knowledge."""
    child_dna = self.dna.copy()  # Inherit DNA
    child_knowledge = dict(random.sample(self.knowledge_base.items(), k=min(3, len(self.knowledge_base))))  # Share partial knowledge
    print(f"Node {self.node_id}: Replicating to create Node {node_id_counter}")
    return LearningNode(node_id=node_id_counter, parent_id=self.node_id, dna=child_dna, knowledge_base=child_knowledge)

def check_thresholds(self):
    """Check if thresholds for replication are reached."""
    return (
        self.resources["memory"] >= self.thresholds["memory"]
        or self.resources["energy"] <= self.thresholds["energy"]
    )

def add_synapse(self, node_id, relevance):
    """Add or update a synapse connection with a weight."""
    self.synapse_overlay[node_id] = relevance
    print(f"Node {self.node_id}: Added synapse to Node {node_id} with relevance {relevance:.2f}")

def query_network(self, query, all_nodes):
    """Query the network for specific knowledge."""
    if query in self.knowledge_base:
        print(f"Node {self.node_id}: Found knowledge - {query}: {self.knowledge_base[query]}")
        return self.knowledge_base[query]

    # Search connected nodes via synapses
    for node_id in sorted(self.synapse_overlay, key=self.synapse_overlay.get, reverse=True):
        if node_id in all_nodes:
            result = all_nodes[node_id].query_network(query, all_nodes)
            if result:
                return result

    print(f"Node {self.node_id}: Query '{query}' not found in the network.")
    return None


