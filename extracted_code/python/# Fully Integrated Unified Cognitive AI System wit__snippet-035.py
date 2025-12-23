def update_knowledge_graph(self, node, new_text=None):
    if node.knowledge > self.crystallization_threshold:
        symbolic_data = redact_pii(new_text) if new_text else None
        self.knowledge_graph.update_node_attributes(node.id,
            {'E': node.energy, 'A': node.awareness, 'K': node.knowledge,
             'position': node.position.components, 'symbolic_data': symbolic_data})
