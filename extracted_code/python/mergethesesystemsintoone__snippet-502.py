def __init__(self, path: str, D: int = 512):
    self.path = path; self.D = D; self._init()

def _init(self):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, tension REAL, energy REAL, size INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, text TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, json TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, url TEXT, title TEXT, text TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS facets (id INTEGER PRIMARY KEY, summary TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, vec BLOB)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS energetics (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits REAL, sfield REAL, L REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS captions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, caption TEXT, top_ids TEXT, weights TEXT)""")
    con.commit(); con.close()

def add_state(self, tick: int, tension: float, energy: float, size: int):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO states(ts, tick, tension, energy, size) VALUES(?,?,?,?,?)", (time.time(), tick, tension, energy, size))
    con.commit(); con.close()

def add_reflection(self, tick: int, text: str):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO reflections(ts, tick, text) VALUES(?,?,?)", (time.time(), tick, text))
    con.commit(); con.close()

def add_suggestion(self, tick: int, js: Dict[str, Any]):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO suggestions(ts, tick, json) VALUES(?,?,?)", (time.time(), tick, json.dumps(js)))
    con.commit(); con.close()

def add_doc_with_embed(self, url: str, title: str, text: str):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO docs(ts, url, title, text) VALUES(?,?,?,?)", (time.time(), url, title, text))
    doc_id = cur.lastrowid
    summary = (text.strip().split("\n")[0] if text else title)[:280]
    e = embed_text(text or title, D=self.D).astype(np.float32)
    cur.execute("INSERT OR REPLACE INTO facets(id, summary) VALUES(?,?)", (doc_id, summary))
    cur.execute("INSERT OR REPLACE INTO embeddings(id, vec) VALUES(?,?)", (doc_id, e.tobytes()))
    con.commit(); con.close()
    return doc_id

def add_energetics(self, tick: int, sigma: float, H_bits: float, S_field: float, L: float):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO energetics(ts, tick, sigma, hbits, sfield, L) VALUES(?,?,?,?,?,?)", (time.time(), tick, sigma, H_bits, S_field, L))
    con.commit(); con.close()

def add_caption(self, tick: int, caption: str, top_ids: List[int], weights: List[float]):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("INSERT INTO captions(ts, tick, caption, top_ids, weights) VALUES(?,?,?, ?, ?)", (time.time(), tick, caption, json.dumps(top_ids), json.dumps(weights)))
    con.commit(); con.close()

def get_embeddings(self, max_items: int | None = None) -> Tuple[np.ndarray, List[int]]:
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute("SELECT id, vec FROM embeddings ORDER BY id ASC")
    rows = cur.fetchall(); con.close()
    if not rows:
        return np.zeros((0, 0), dtype=np.float64), []
    ids = [int(r[0]) for r in rows]
    arrs = [np.frombuffer(r[1], dtype=np.float32).astype(np.float64) for r in rows]
    E = np.stack(arrs, axis=0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    if max_items and len(ids) > max_items:
        idx = np.random.RandomState(123).choice(len(ids), size=max_items, replace=False)
        E = E[idx]; ids = [ids[i] for i in idx]
    return E, ids

def recent(self, table: str, limit: int = 50):
    con = sqlite3.connect(self.path); cur = con.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); con.close()
    return rows

