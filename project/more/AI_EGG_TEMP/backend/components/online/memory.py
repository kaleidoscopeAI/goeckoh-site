# backend/components/online/memory.py
import numpy as np
import faiss
from typing import List, Dict, Any
from interfaces import BaseMemory

class OnlineMemory(BaseMemory):
    def __init__(self, dim=384, capacity=10000):
        self.capacity = capacity
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []

    def add(self, emb: np.ndarray, meta: Dict[str, Any]):
        if self.index.ntotal >= self.capacity:
            # Remove the oldest entry
            self.index.remove_ids(np.array([0], dtype='int64'))
            self.meta.pop(0)
        
        self.index.add(emb.reshape(1, -1).astype('float32'))
        self.meta.append(meta)

    def query(self, emb: np.ndarray, k=5) -> list:
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(emb.reshape(1, -1).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # L2 distance to similarity (0-1 range)
                similarity = 1 / (1 + distances[0][i])
                results.append((self.meta[idx], similarity))
        return results
