# backend/components/online/embedder.py
import numpy as np
from typing import List
from interfaces import BaseEmbedder

# This component requires the sentence-transformers library.
# pip install -U sentence-transformers

class OnlineEmbedder(BaseEmbedder):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("Please install sentence-transformers: pip install -U sentence-transformers")
            self.model = None

    def fit(self, texts: List[str]):
        # Not needed for pre-trained sentence transformers
        pass

    def embed(self, text: str) -> np.ndarray:
        if self.model is None:
            # Fallback to simple hashing if library is not installed
            import hashlib
            h = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(np.float32)
            return (h - h.mean()) / (h.std() + 1e-9)
        
        embedding = self.model.encode([text])
        return embedding[0].astype(np.float32)
