def stft_mag(x: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
    from scipy.signal import get_window
    w = get_window("hann", win)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    T = 1 + (len(x) - win) // hop
    F = win // 2 + 1
    X = np.zeros((F, T), dtype=np.float64)
    for t in range(T):
        s = t * hop
        seg = x[s:s + win]
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
        if b <= a:
            b = min(a + 1, F)
        E[h] = X[a:b].mean(axis=0)
    d1 = np.pad(np.diff(E, axis=1), ((0, 0), (1, 0)))
    d2 = np.pad(np.diff(d1, axis=1), ((0, 0), (1, 0)))
    return np.stack([E, d1, d2], axis=-1)  # (H, T, 3)

def project_and_attention(V: np.ndarray, E_mem: np.ndarray, d: int, sigma_temp: float) -> Dict:
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
        for t in range(T):
            w = P[t]
            top = np.argsort(-w)[:8]
            shapes.append({"head": int(h), "t": int(t), "ids": top.astype(int).tolist(), "weights": w[top].astype(float).tolist()})
    return {"shapes": shapes}

