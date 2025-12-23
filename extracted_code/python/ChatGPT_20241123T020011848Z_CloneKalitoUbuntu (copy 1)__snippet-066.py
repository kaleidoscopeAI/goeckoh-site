class Node:
    def __init__(self, node_id, threshold=10):
        self.node_id = node_id
        self.learned_knowledge = []
        self.threshold = threshold  # Replication threshold
        self.resources_shared = 0

    def learn(self, knowledge):
        """Add knowledge and evaluate for replication."""
        self.learned_knowledge.append(knowledge)
        if len(self.learned_knowledge) >= self.threshold:
            return self.replicate()

    def replicate(self):
        """Replicate the node with a shared resource base."""
        new_node = Node(node_id=self.node_id + "_child")
        new_node.learned_knowledge = self.learned_knowledge.copy()
        return new_node

