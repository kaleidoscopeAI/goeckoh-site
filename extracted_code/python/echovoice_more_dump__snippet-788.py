class UnifiedMemory:
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

# Annealer and PT (from annealer.py, rust_pt.py)
class CognitiveAnnealer:
    def __init__(self, evaluate_energy: Callable[[np.ndarray], float], initial_temp=1.0, final_temp=1e-3, schedule='exp', steps=2000):
        self.eval = evaluate_energy
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule = schedule
        self.steps = steps

    def temp_at(self, step):
        if self.schedule == 'exp':
            return self.initial_temp * (self.final_temp / self.initial_temp) ** (step / max(1, self.steps-1))
        elif self.schedule == 'linear':
            return self.initial_temp + (self.final_temp - self.initial_temp) * (step / max(1, self.steps-1))
        return self.initial_temp

    def anneal(self, initial_state: np.ndarray, neighbor_fn: Callable[[np.ndarray, float], np.ndarray]):
        state = initial_state.copy()
        best = state.copy()
        best_e = self.eval(state)
        current_e = best_e
        for step in range(self.steps):
            T = self.temp_at(step)
            candidate = neighbor_fn(state, T)
            ce = self.eval(candidate)
            delta = ce - current_e
            accept = delta < 0 or np.random.rand() < math.exp(-delta / (T + 1e-12))
            if accept:
                state = candidate
                current_e = ce
            if ce < best_e:
                best = candidate.copy()
                best_e = ce
        return best, best_e

class ParallelTempering:
    def __init__(self, evaluate_energy: Callable[[np.ndarray], float], temps: List[float]):
        self.eval = evaluate_energy
        self.temps = temps

    def run_ensemble(self, initial_states: List[np.ndarray], neighbor_fn: Callable[[np.ndarray, float], np.ndarray], steps_per_replica=200):
        n = len(initial_states)
        reps = [state.copy() for state in initial_states]
        energies = [self.eval(r) for r in reps]
        with ThreadPoolExecutor(max_workers=min(8, n)) as ex:
            futures = [ex.submit(self._local_search, reps[i], self.temps[i], neighbor_fn, steps_per_replica) for i in range(n)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        states = [r[0] for r in results]
        energies = [r[1] for r in results]
        for i in range(len(self.temps)-1):
            e1 = energies[i]
            e2 = energies[i+1]
            T1 = self.temps[i]
            T2 = self.temps[i+1]
            delta = (e2 - e1) * (1.0/T1 - 1.0/T2)
            if delta < 0 or np.random.rand() < math.exp(-delta):
                states[i], states[i+1] = states[i+1], states[i]
                energies[i], energies[i+1] = energies[i+1], energies[i]
        best_idx = int(np.argmin(energies))
        return states[best_idx], energies[best_idx]

    def _local_search(self, state, temp, neighbor_fn, steps):
        cur = state.copy()
        cur_e = self.eval(cur)
        for _ in range(steps):
            cand = neighbor_fn(cur, temp)
            ce = self.eval(cand)
            delta = ce - cur_e
            if delta < 0 or np.random.rand() < math.exp(-delta / (temp+1e-12)):
                cur = cand
                cur_e = ce
        return cur, cur_e

# Load Rust (from rust_pt.py)
