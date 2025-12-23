def run_cycle(self, step, input_texts):
    self._compute_master_state_psi(self.nodes)
    self.apply_master_state_operators()

    self.update_node_hypotheses(step)

    for node, text in zip(self.nodes, input_texts):
        self.reflect_with_emotion(text, node)
        self._update_knowledge_graph(node, text)
