from core_node import Node

class NodeReplication:
    def __init__(self):
        self.nodes = [Node()]

    def replicate_node(self, parent_node):
        """Replicate a node based on maturity threshold."""
        if parent_node.maturity > 1.0 and parent_node.energy > 0.5:
            child_node = Node()
            child_node.generation = parent_node.generation + 1
            child_node.knowledge = parent_node.knowledge.copy()  # Inherit knowledge
            child_node.energy = parent_node.energy * 0.5  # Share energy
            parent_node.energy *= 0.5  # Retain half energy
            self.nodes.append(child_node)
            return child_node
        return None

    def manage_replication(self):
        """Iterate through nodes and manage replication."""
        for node in self.nodes:
            self.replicate_node(node)

    def get_status(self):
        return {
            "Total Nodes": len(self.nodes),
            "Generations": len(set(node.generation for node in self.nodes))
        }

