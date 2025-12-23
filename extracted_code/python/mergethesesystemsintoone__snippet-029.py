def synth_signal(seconds: float, sr: int, a_fn, m_fn, rho_fn, fc_fn, alpha: float = 0.8, beta: float = 0.4) -> List[float]:
    n = int(seconds * sr)
    out = []
    for i in range(n):
        t = i / sr
        a = a_fn(t); m = m_fn(t); rho = rho_fn(t); fc = max(5.0, fc_fn(t))
        y = a * (1.0 + beta * math.sin(2.0 * math.pi * m * t)) * math.sin(2.0 * math.pi * fc * t + alpha * math.sin(2.0 * math.pi * rho * t))
        out.append(y)
    return out

def default_maps(H_bits: float, S_field: float, latency: float, fitness: float, fmin: float = 110.0, fdelta: float = 440.0):
    H = max(0.0, min(1.0, H_bits))
    S = max(0.0, min(1.0, S_field))
    L = max(0.0, min(1.0, latency))
    F = max(0.0, min(1.0, fitness))
    def a_fn(t): return 0.25 + 0.5 * (1.0 - H) * (1.0 - S)
    def m_fn(t): return 2.0 + 10.0 * S
    def rho_fn(t): return 0.2 + 3.0 * (1.0 - L)
    def fc_fn(t): return fmin + fdelta * F
    return {"a": a_fn, "m": m_fn, "rho": rho_fn, "fc": fc_fn}

def stft_mag(x: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
    # Simple STFT mag using NumPy
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    w = np.hanning(win)
    T = 1 + (len(x) - win) // hop
    F = win // 2 + 1
    X = np.zeros((F, T), dtype=np.float64)
    for t in range(T):
        s = t * hop
        seg = x[s:s+win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.fft.rfft(seg * w)
        X[:, t] = np.abs(spec)
    return X

def make_bands(F: int, H: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, F, H + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(H)]

def head_features(X: np.ndarray, bands: List[Tuple[int, int]]) -> np.ndarray:
    F, T = X.shape
    H = len(bands)
    E = np.zeros((H, T), dtype=np.float64)
    for h, (a, b) in enumerate(bands):
        if b <= a: b = min(a + 1, F)
        E[h] = X[a:b].mean(axis=0)
    d1 = np.pad(np.diff(E, axis=1), ((0,0),(1,0)))
    d2 = np.pad(np.diff(d1, axis=1), ((0,0),(1,0)))
    return np.stack([E, d1, d2], axis=-1)

def project_and_attention(V: np.ndarray, E_mem: np.ndarray, d: int, sigma_temp: float) -> Dict[str, Any]:
    H, T, F3 = V.shape
    D = E_mem.shape[1]
    rng = np.random.RandomState(1234)
    Wk = rng.normal(0, 1.0 / math.sqrt(D), size=(D, d))
    Wqs = rng.normal(0, 1.0, size=(H, F3, d))
    K = E_mem @ Wk
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    shapes = []
    tau = max(1e-3, sigma_temp)
    for h in range(H):
        Q = V[h] @ Wqs[h]
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        S = (Q @ K.T) / (d * tau)
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= (P.sum(axis=1, keepdims=True) + 1e-12)
        for t in range(V.shape[1]):
            w = P[t]
            top = np.argsort(-w)[:8]
            shapes.append({"head": int(h), "t": int(t), "ids": top.astype(int).tolist(), "weights": w[top].astype(float).tolist()})
    return {"shapes": shapes}

def fetch_summaries(db_path: str) -> Dict[int, str]:
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute("SELECT id, summary FROM facets")
    out = {int(i): (s or "") for (i, s) in cur.fetchall()}
    con.close(); return out

def kw(text: str, k: int = 10) -> str:
    return " ".join(text.replace("\n", " ").split()[:k])

def captions_from_shapes(db_path: str, shapes: Dict[str, Any], top_k: int = 3, window: int = 5, stride: int = 5, hbits: float = None, sfield: float = None) -> Dict[str, Any]:
    id2sum = fetch_summaries(db_path)
    by_t: Dict[int, List[Tuple[List[int], List[float]]]] = {}
    T = 0
    for rec in shapes["shapes"]:
        t = int(rec["t"]); T = max(T, t+1)
        by_t.setdefault(t, []).append((rec["ids"], rec["weights"]))
    caps = []; t0 = 0
    while t0 < T:
        t1 = min(T-1, t0 + window - 1)
        score: Dict[int, float] = {}; denom = 0.0
        for t in range(t0, t1+1):
            for ids, wts in by_t.get(t, []):
                for i, w in zip(ids, wts):
                    score[i] = score.get(i, 0.0) + float(w); denom += float(w)
        if denom > 0:
            for i in list(score.keys()):
                score[i] /= denom
        top = sorted(score.items(), key=lambda x: -x[1])[:top_k]
        top_ids = [i for i,_ in top]
        phrases = [kw(id2sum.get(i, ""), 10) for i in top_ids]
        cap = {"t_start_frame": t0, "t_end_frame": t1, "top_ids": top_ids, "weights": [float(w) for _, w in top], "caption": "; ".join([p for p in phrases if p])}
        if hbits is not None: cap["H_bits"] = float(hbits)
        if sfield is not None: cap["S_field"] = float(sfield)
        caps.append(cap); t0 += stride
    return {"captions": caps, "meta": {"window": window, "stride": stride, "top_k": top_k}}

