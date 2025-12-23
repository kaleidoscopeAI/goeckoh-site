import numpy as np

class Node:
    def __init__(self, node_id, energy):
        self.node_id = node_id
        self.energy = energy
        self.neighbors = []

    def share_resources(self):
        """Share energy with neighbors."""
        if self.energy > 0.5:
            shared_energy = self.energy * 0.1  # Share 10% of energy
            for neighbor in self.neighbors:
                neighbor.energy += shared_energy
            self.energy -= shared_energy * len(self.neighbors)
    
    def connect(self, other_node):
        self.neighbors.append(other_node)
        other_node.neighbors.append(self)

def create_network(num_nodes=5):
    """Create a network of connected nodes."""
    nodes = [Node(i, np.random.rand()) for i in range(num_nodes)]
    for i in range(num_nodes - 1):
        nodes[i].connect(nodes[i + 1])
    return nodes

def simulate_network(nodes, steps=10):
    """Simulate resource sharing in the network."""
    for step in range(steps):
        print(f"Step {step}")
        for node in nodes:
            node.share_resources()
            print(f"Node {node.node_id}: Energy {node.energy:.2f}")
        print("-" * 30)

if __name__ == "__main__":
    nodes = create_network()
    simulate_network(nodes)

