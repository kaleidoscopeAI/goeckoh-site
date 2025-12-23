class EpisodicMemory:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add_experience(self, context_vector, metadata=None):
        entry = {"context": context_vector, "meta": metadata, "timestamp": time.time()}
        self.memory.append(entry)

    def retrieve_similar(self, query_vector, top_k=5):
        # Naive cosine similarity for example
        sims = []
        for entry in self.memory:
            sim = np.dot(query_vector, entry["context"]) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(entry["context"]) + 1e-8)
            sims.append((sim, entry))
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]

# Add episodic memory instance to your cognitive machine and update cycle accordingly.
