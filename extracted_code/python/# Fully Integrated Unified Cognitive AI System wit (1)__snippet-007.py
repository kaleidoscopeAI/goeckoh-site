from collections import deque
import time
import numpy as np

class EpisodicMemory:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, context_vector, metadata=None):
        entry = {"context": context_vector, "metadata": metadata, "timestamp": time.time()}
        self.memory.append(entry)

    def retrieve_similar(self, query_vector, top_k=5):
        sims = []
        for entry in self.memory:
            c = entry["context"]
            sim = np.dot(query_vector, c) / (np.linalg.norm(query_vector) * np.linalg.norm(c) + 1e-8)
            sims.append((sim, entry))
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]

# Instantiate and use episodic memory in the cognitive loop
episodic_memory = EpisodicMemory()
