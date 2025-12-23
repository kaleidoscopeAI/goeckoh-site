import numpy as np
import random

class LearningNode:
    def __init__(self, node_id, parent_id=None, dna=None, knowledge_base=None):
        self.node_id = node_id
        self.parent_id = parent_id  # Track the parent node
        self.dna = dna if dna else np.random.rand(5)  # Node DNA (randomized or inherited)
        self.knowledge_base = knowledge_base if knowledge_base else []  # Knowledge repository
        self.resources = {"memory": 0.5, "energy": 1.0}  # Initial resources
        self.thresholds = {"memory": 0.8, "energy": 0.2}  # Replication thresholds

    def learn(self, new_data):
        """Simulate learning and knowledge storage."""
        self.knowledge_base.append(new_data)
        self.resources["memory"] += 0.1  # Simulate memory usage
        self.resources["energy"] -= 0.05  # Simulate energy expenditure
        print(f"Node {self.node_id}: Learned - {new_data}")

    def replicate(self, node_id_counter):
        """Replicate the node with inherited DNA and partial knowledge."""
        child_dna = self.dna.copy()  # Inherit DNA
        child_knowledge = random.sample(self.knowledge_base, k=min(3, len(self.knowledge_base)))  # Share partial knowledge
        print(f"Node {self.node_id}: Replicating to create Node {node_id_counter}")
        return LearningNode(node_id=node_id_counter, parent_id=self.node_id, dna=child_dna, knowledge_base=child_knowledge)

    def check_thresholds(self):
        """Check if thresholds for replication are reached."""
        return (
            self.resources["memory"] >= self.thresholds["memory"]
            or self.resources["energy"] <= self.thresholds["energy"]
        )

# Node Network to manage multiple nodes
class NodeNetwork:
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

# Run the simulation
if __name__ == "__main__":
    network = NodeNetwork()
    network.add_node()  # Start with a seed node
    network.simulate(steps=20)

