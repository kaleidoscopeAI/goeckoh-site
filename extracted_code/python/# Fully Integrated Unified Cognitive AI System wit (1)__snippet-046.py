import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class CrystallineKnowledgeBase:
    def __init__(self):
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.dim = 768
        self.index = faiss.IndexFlatL2(self.dim)
        self.data = []  # Store original texts or metadata

    def add(self, text):
        embedding = self.embedder.encode(text)
        self.index.add(np.array([embedding], dtype='float32'))
        self.data.append(text)

    def recall(self, query_text, k=5):
        query_embedding = self.embedder.encode(query_text).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = [self.data[i] for i in indices[0] if i != -1]
        return results

    def consolidate(self, texts):
        for t in texts:
            self.add(t)
