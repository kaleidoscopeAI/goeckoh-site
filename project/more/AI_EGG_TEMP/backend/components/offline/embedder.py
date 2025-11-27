# backend/components/offline/embedder.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import hashlib
from typing import List
from interfaces import BaseEmbedder

class OfflineEmbedder(BaseEmbedder):
    def __init__(self, n_components=64):
        self.vectorizer = TfidfVectorizer(max_features=1024)
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        self.n_components = n_components

    def fit(self, texts: List[str]):
        X = self.vectorizer.fit_transform(texts).toarray()
        if X.shape[1] < self.pca.n_components:
            pad = np.zeros((X.shape[0], self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        self.pca.fit(X)
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            h = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(np.float32)
            if len(h) < self.n_components:
                h = np.pad(h, (0, self.n_components - len(h)))
            h = h[:self.n_components]
            return (h - h.mean()) / (h.std() + 1e-9)
        
        X = self.vectorizer.transform([text]).toarray()
        if X.shape[1] < self.pca.n_components:
            pad = np.zeros((1, self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        emb = self.pca.transform(X)[0]
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm
