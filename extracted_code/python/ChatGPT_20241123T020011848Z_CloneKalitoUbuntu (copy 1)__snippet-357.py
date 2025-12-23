def __init__(self, node_id, dna, memory_limit=10):
    self.node_id = node_id
    self.dna = dna  # Genetic blueprint for node creation
    self.energy = 1.0  # Node energy
    self.memory = []  # Local storage of learned data
    self.memory_limit = memory_limit  # Max items in memory
    self.connected_nodes = []  # References to other nodes
    self.logs = []  # Activity logs

def learn(self, data):
    """Learn from new data and store in memory."""
    if len(self.memory) >= self.memory_limit:
        self.memory.pop(0)  # Remove oldest memory to make space
    self.memory.append(data)
    self.logs.append(f"Learned: {data}")

def share_resources(self):
    """Share resources or information with connected nodes."""
    shared_data = self.memory[-1] if self.memory else None
    for node in self.connected_nodes:
        if shared_data:
            node.learn(shared_data)

def replicate(self, new_node_id):
    """Replicate the node with slight variations."""
    new_dna = self.dna + np.random.normal(0, 0.01, size=self.dna.shape)
    child_node = Node(new_node_id, dna=new_dna, memory_limit=self.memory_limit)
    child_node.learn("Inherited: " + str(self.memory))
    return child_node

def __repr__(self):
    return f"Node {self.node_id} (Energy: {self.energy}, Memory: {len(self.memory)})"

