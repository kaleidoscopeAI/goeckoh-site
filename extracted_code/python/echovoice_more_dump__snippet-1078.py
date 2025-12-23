def __init__(self, db_path="ica_memory.db", dim=384, model_name="all-MiniLM-L6-v2"):
    self.dim = dim
    self.conn = sqlite3.connect(db_path, check_same_thread=False)
    self._ensure_tables()
    self.model = SentenceTransformer(model_name)
    self.index = faiss.IndexFlatIP(dim)  # Cosine sim
    self.ids = []
    self.memory_metadata = []
    self.memory_embeddings = []

def _ensure_tables(self):
    cur = self.conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        vec BLOB,
        valence REAL,
        arousal REAL,
        meta TEXT
    );
    CREATE TABLE IF NOT EXISTS meta (id TEXT PRIMARY KEY, valence REAL, arousal REAL, meta TEXT);
    """)
    self.conn.commit()

def _vec_to_blob(self, vec):
    return vec.astype(np.float32).tobytes()

def _blob_to_vec(self, blob):
    return np.frombuffer(blob, dtype=np.float32)

def _to_vec(self, text):
    emb = self.model.encode([text], show_progress_bar=False)
    v = np.array(emb, dtype=np.float32)
    faiss.normalize_L2(v)
    return v[0]

def store(self, id: str, text_or_vec: Any, valence: float, arousal: float, meta: dict):
    if isinstance(text_or_vec, str):
        vec = self._to_vec(text_or_vec)
    else:
        vec = np.array(text_or_vec, dtype=np.float32)
        faiss.normalize_L2(vec.reshape(1, -1))
    self.index.add(vec.reshape(1, -1))
    self.ids.append(id)
    self.memory_metadata.append({'valence': valence, 'arousal': arousal, 'meta': meta})
    self.memory_embeddings.append(vec)
    vec_blob = self._vec_to_blob(vec)
    cur = self.conn.cursor()
    cur.execute("REPLACE INTO memories (id,vec,valence,arousal,meta) VALUES (?,?,?, ?, ?)",
                (id, vec_blob, float(valence), float(arousal), json.dumps(meta)))
    cur.execute("REPLACE INTO meta (id, valence, arousal, meta) VALUES (?,?,?,?)",
                (id, float(valence), float(arousal), json.dumps(meta)))
    self.conn.commit()

def query(self, query_text_or_vec: Any, valence: float, arousal: float, alpha=0.7, beta=0.3, top_k=5):
    if isinstance(query_text_or_vec, str):
        qv = self._to_vec(query_text_or_vec).reshape(1, -1)
    else:
        qv = np.array(query_text_or_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
    D, I = self.index.search(qv, top_k)
    results = []
    cur = self.conn.cursor()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(self.ids): continue
        id = self.ids[idx]
        cur.execute("SELECT valence, arousal, meta FROM meta WHERE id=?", (id,))
        row = cur.fetchone()
        if row:
            sval, sar, meta = row
            emo_dist = abs(valence - float(sval)) + abs(arousal - float(sar))
            sem_dist = 1.0 - float(score)
            final_score = alpha * sem_dist + beta * emo_dist
            results.append((final_score, id, json.loads(meta)))
    results.sort(key=lambda x: x[0])
    return results[:top_k]

