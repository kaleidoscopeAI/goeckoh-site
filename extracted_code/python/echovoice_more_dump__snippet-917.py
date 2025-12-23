class LocalEmbedder:
    def __init__(self, n_components=64):
        self.vectorizer = TfidfVectorizer(max_features=1024)
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, texts: List[str]):
        X = self.vectorizer.fit_transform(texts).toarray()
        if X.shape[1] < self.pca.n_components:
            # pad columns
            pad = np.zeros((X.shape[0], self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        self.pca.fit(X)
        self.fitted = True
        logging.info('LocalEmbedder fitted on corpus')

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            # fall back to hashing-based embedding
            h = np.frombuffer(hashlib_sha256(text.encode()).digest()[:64], dtype=np.uint8).astype(np.float32)
            return (h - h.mean()) / (h.std() + 1e-9)
        X = self.vectorizer.transform([text]).toarray()
        if X.shape[1] < self.pca.n_components:
            pad = np.zeros((1, self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        emb = self.pca.transform(X)[0]
        # normalize
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm

