def write_wav_mono16(path: Path, sr: int, samples: Iterable[float]) -> None:
    x = np.asarray(list(samples), dtype=np.float32)
    if x.size == 0: x = np.zeros(1, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(y.tobytes())

def stft_mag(x: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
    w = get_window("hann", win)
    if len(x) < win: x = np.pad(x, (0, win - len(x)))
    T = 1 + (len(x) - win)//hop
    F = win//2 + 1
    X = np.zeros((F, T), dtype=np.float64)
    for t in range(T):
        s = t*hop
        seg = x[s:s+win]
        if len(seg) < win: seg = np.pad(seg, (0, win-len(seg)))
        spec = np.fft.rfft(seg * w)
        X[:, t] = np.abs(spec)
    return X

def make_bands(F: int, H: int) -> List[Tuple[int,int]]:
    edges = np.linspace(0, F, H+1, dtype=int)
    return [(int(edges[i]), int(edges[i+1])) for i in range(H)]

def head_features(X: np.ndarray, bands: List[Tuple[int,int]]) -> np.ndarray:
    F, T = X.shape; H = len(bands)
    E = np.zeros((H, T), dtype=np.float64)
    for h,(a,b) in enumerate(bands):
        if b<=a: b=min(a+1,F)
        E[h] = X[a:b].mean(axis=0)
    d1 = np.pad(np.diff(E,axis=1), ((0,0),(1,0)))
    d2 = np.pad(np.diff(d1,axis=1), ((0,0),(1,0)))
    return np.stack([E,d1,d2], axis=-1)  # (H,T,3)

