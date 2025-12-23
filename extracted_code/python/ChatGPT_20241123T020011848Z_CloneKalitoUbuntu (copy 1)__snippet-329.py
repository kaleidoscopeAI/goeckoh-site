def __init__(self):
    self.nodes = []
    self.node_id_counter = 1  # Unique ID for nodes

def add_node(self, parent_id=None):
    """Add a new node to the network."""
    new_node = LearningNode(node_id=self.node_id_counter, parent_id=parent_id)
    self.nodes.append(new_node)
    self.node_id_counter += 1

def simulate(self, steps=20):
    """Simulate the network over multiple steps."""
    for step in range(steps):
        print(f"\n--- Step {step} ---")
        new_nodes = []
        for node in self.nodes:
            node.learn(f"Data-{step}")  # Simulate learning

            if node.check_thresholds():
                # Replicate if thresholds are reached
                new_nodes.append(node.replicate(self.node_id_counter))
                self.node_id_counter += 1

        # Add new nodes to the network
        self.nodes.extend(new_nodes)

