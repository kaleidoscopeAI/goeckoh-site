def __init__(self):
    self.nodes = {}
    self.node_id_counter = 1  # Unique ID for nodes

def add_node(self, parent_id=None):
    """Add a new node to the network."""
    new_node = LearningNode(node_id=self.node_id_counter, parent_id=parent_id)
    self.nodes[self.node_id_counter] = new_node
    self.node_id_counter += 1

def simulate(self, steps=20):
    """Simulate the network over multiple steps."""
    for step in range(steps):
        print(f"\n--- Step {step} ---")
        new_nodes = []
        for node in self.nodes.values():
            # Simulate learning
            node.learn(f"Data-{step}", f"Value-{step}")

            # Check for replication
            if node.check_thresholds():
                new_node = node.replicate(self.node_id_counter)
                self.nodes[self.node_id_counter] = new_node
                self.node_id_counter += 1
                # Establish a synapse between parent and child
                node.add_synapse(new_node.node_id, relevance=random.uniform(0.5, 1.0))

        # Update synapse overlays (dynamic connections based on knowledge relevance)
        for node in self.nodes.values():
            for target_node_id, target_node in self.nodes.items():
                if node.node_id != target_node_id:
                    shared_keys = set(node.knowledge_base.keys()).intersection(target_node.knowledge_base.keys())
                    if shared_keys:
                        relevance = len(shared_keys) / len(node.knowledge_base)
                        node.add_synapse(target_node_id, relevance)

def query(self, query):
    """Query the network starting from the first node."""
    if 1 in self.nodes:  # Start with the seed node
        return self.nodes[1].query_network(query, self.nodes)
    print("Network is empty.")
    return None


