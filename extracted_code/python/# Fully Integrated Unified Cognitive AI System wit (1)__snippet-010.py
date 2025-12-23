class CognitiveMachine:
    def __init__(self):
        # Existing initialization ...
        self.episodic_memory = EpisodicMemory()

    def remember_context(self, node_vectors):
        # For example, average vectors or key supernode prototypes
        context_vec = np.mean(node_vectors, axis=0)
        self.episodic_memory.add(context_vec, metadata={"cycle_time": time.time()})

    def recall_similar(self, query_vec):
        return self.episodic_memory.retrieve_similar(query_vec)

    def run_cycle(self):
        # After updating and reflecting
        node_vectors = np.array([n.vector for n in self.cube.nodes])
        self.remember_context(node_vectors)
        recalled = self.recall_similar(node_vectors[-1])  # Example recall recent vector
        # Use recalled info to influence reasoning or device control
