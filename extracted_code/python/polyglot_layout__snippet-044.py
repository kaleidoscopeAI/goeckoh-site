  H_bits = float(np.mean(1.0 - S) if N else 0.0)
  S_field = float(np.mean(tau))
  L = float(np.sum(tau * tau))
  return {"H_bits": H_bits, "S_field": S_field, "L": L}

# Sonification
def synth_signal(seconds: float, sr: int, a_fn, m_fn, rho_fn, fc_fn, alpha: float = 0.8, beta: float = 0.4) -> List[float]:
  n = int(seconds * sr)
  out = []
  for i in range(n):
     t = i / sr
     a = a_fn(t)
     m = m_fn(t)
     rho = rho_fn(t)
     fc = max(5.0, fc_fn(t))
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

# Audio to Shapes (STFT + Attention)
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
  return np.stack([E, d1, d2], axis=-1) # (H, T, 3)

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

