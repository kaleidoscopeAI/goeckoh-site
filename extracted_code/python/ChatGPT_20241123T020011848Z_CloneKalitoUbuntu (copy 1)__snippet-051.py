class AutonomousReflection:
    def __init__(self):
        self.reflections = []

    def generate_insight(self, node):
        """Generate insights from node's knowledge."""
        if node.knowledge:
            insight = f"Node {node.id} has {len(node.knowledge)} knowledge entries."
            self.reflections.append(insight)
            return insight
        return "No knowledge available for reflection."

    def reflect_all(self, nodes):
        """Reflect on all nodes."""
        insights = [self.generate_insight(node) for node in nodes]
        return insights

