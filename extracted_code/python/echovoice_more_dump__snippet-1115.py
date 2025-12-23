def __init__(self, dim=64, capacity=1024):
    self.capacity = capacity
    self.dim = dim
    self.embeddings = np.zeros((0, dim), dtype=np.float32)
    self.meta = []

def add(self, emb: np.ndarray, meta: Dict[str, Any]):
    if emb.shape[0] != self.dim:
        # project or pad
        if emb.shape[0] > self.dim:
            emb = emb[:self.dim]
        else:
            pad = np.zeros(self.dim - emb.shape[0], dtype=np.float32)
            emb = np.concatenate([emb, pad])
    emb = emb.astype(np.float32)
    if self.embeddings.shape[0] >= self.capacity:
        # FIFO
        self.embeddings = np.roll(self.embeddings, -1, axis=0)
        self.embeddings[-1] = emb
        self.meta = self.meta[1:] + [meta]
    else:
        self.embeddings = np.vstack([self.embeddings, emb]) if self.embeddings.size else emb.reshape(1, -1)
        self.meta.append(meta)

def query(self, emb: np.ndarray, k=5):
    if self.embeddings.size == 0:
        return []
    # cosine similarity
    dots = np.dot(self.embeddings, emb)
    norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(emb) + 1e-9)
    sims = dots / (norms + 1e-9)
    idx = np.argsort(-sims)[:k]
    return [(self.meta[i], float(sims[i])) for i in idx]

