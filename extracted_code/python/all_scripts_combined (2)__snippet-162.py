def __init__(self, embedding_dim: int = 128, capacity: int = 10000):
    self.embedding_dim = embedding_dim
    self.capacity = capacity
    self.episodic = []  # list of (ts, emb, text)
    self.semantic = {}  # key -> emb
    self._rng = np.random.RandomState(12345)
    # We'll derive a deterministic random projection matrix seeded by a constant
    self.random_proj = self._rng.randn(self.embedding_dim, 256) * 0.01

def embed(self, text: str) -> np.ndarray:
    b = stable_hash_bytes(text, length=256)
    arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
    emb = self.random_proj @ arr
    n = np.linalg.norm(emb)
    return emb / (n + 1e-12)

def store_episode(self, text: str):
    emb = self.embed(text)
    ts = now_seconds()
    if len(self.episodic) >= self.capacity:
        self.episodic.pop(0)
    self.episodic.append((ts, emb, text))

def retrieve(self, query: str, top_k: int = 5):
    q_emb = self.embed(query)
    sims = []
    for ts, emb, text in self.episodic:
        sims.append((float(np.dot(q_emb, emb)), ts, text))
    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:top_k]

def store_semantic(self, key: str, text: str):
    self.semantic[key] = self.embed(text)

def lookup_semantic(self, key: str):
    return self.semantic.get(key, None)

