    Q = V[h] @ Wqs[h]
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    S = (Q @ K.T) / (d * tau)
    S -= S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= (P.sum(axis=1, keepdims=True) + 1e-12)
    for t in range(T):
       w = P[t]
       top = np.argsort(-w)[:8]
       shapes.append({"head": int(h), "t": int(t), "ids": top.astype(int).tolist(), "weights": w[top].astype(float).tolist()})
  return {"shapes": shapes}

# Captioner
def fetch_summaries(db_path: str) -> Dict[int, str]:
  con = sqlite3.connect(db_path)
  cur = con.cursor()
  cur.execute("SELECT id, summary FROM facets")
  out = {int(i): (s or "") for (i, s) in cur.fetchall()}
  con.close()
  return out

def kw(text: str, k: int = 10) -> str:
  return " ".join(text.replace("\n", " ").split()[:k])

def captions_from_shapes(db_path: str, shapes: Dict, top_k: int = 3, window: int = 5, stride: int = 5, hbits: float = None, sfield: float = None) ->
Dict[str, Any]:
  id2sum = fetch_summaries(db_path)
  by_t: Dict[int, List[Tuple[List[int], List[float]]]] = {}
  T=0
  for rec in shapes["shapes"]:
     t = int(rec["t"])
     T = max(T, t + 1)
     by_t.setdefault(t, []).append((rec["ids"], rec["weights"]))
  caps = []
  t0 = 0
  while t0 < T:
     t1 = min(T - 1, t0 + window - 1)
     score: Dict[int, float] = {}
     denom = 0.0
     for t in range(t0, t1 + 1):
        for ids, wts in by_t.get(t, []):
           for i, w in zip(ids, wts):
              score[i] = score.get(i, 0.0) + float(w)
              denom += float(w)
     if denom > 0:
        for i in list(score.keys()):
           score[i] /= denom
     top = sorted(score.items(), key=lambda x: -x[1])[:top_k]
     top_ids = [i for i, _ in top]
     phrases = [kw(id2sum.get(i, ""), 10) for i in top_ids]
     cap = {"t_start_frame": t0, "t_end_frame": t1, "top_ids": top_ids, "weights": [float(w) for _, w in top], "caption": "; ".join([p for p in phrases if p])}
     if hbits is not None:
        cap["H_bits"] = float(hbits)
     if sfield is not None:
        cap["S_field"] = float(sfield)
     caps.append(cap)
     t0 += stride
  return {"captions": caps, "meta": {"window": window, "stride": stride, "top_k": top_k}}

# Memory Store
class MemoryStore:
   def __init__(self, path: str, D: int = 512):
     self.path = path
     self.D = D
     self._init()

  def _init(self):
    con = sqlite3.connect(self.path)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, tension REAL, energy
REAL, size INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, text TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, json TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, url TEXT, title TEXT, text TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS facets (id INTEGER PRIMARY KEY, summary TEXT)""")

