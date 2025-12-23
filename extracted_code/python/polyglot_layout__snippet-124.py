  def __init__(self, n=18000, seed=123):
      self.n = int(n)
      rng = np.random.RandomState(seed)
      r = 0.8 * np.cbrt(rng.rand(self.n))
      theta = 2*np.pi*rng.rand(self.n)
      phi = np.arccos(2*rng.rand(self.n)-1)
      x = r*np.sin(phi)*np.cos(theta)
      y = r*np.sin(phi)*np.sin(theta)
      z = r*np.cos(phi)
      self.pos = np.stack([x,y,z], axis=1).astype(np.float32)
      self.vel = np.zeros_like(self.pos, dtype=np.float32)
      self._bytes = None
      self._lock = threading.Lock()
      self.shape_id = 0

  @staticmethod
  def _hash_str(s: str) -> int:
      h = 1469598103934665603
      for ch in s.encode("utf-8","ignore"):
          h ^= ch; h *= 1099511628211; h &= (1<<64)-1
      return h if h>=0 else -h

  def _target_field(self, shape_id:int) -> np.ndarray:
      N = self.n; P = self.pos
      t = shape_id % 4
      if t == 0: # sphere
          radius = 0.9
          return radius * P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-6)
      if t == 1: # torus
          R, r = 0.8, 0.25
          x, y, z = P[:,0], P[:,1], P[:,2]
          q = np.sqrt(x*x + y*y)+1e-6
          tx = R * (x / q); ty = R * (y / q); rz = np.sign(z) * r
          return np.stack([tx, ty, rz], axis=1).astype(np.float32)
      if t == 2: # helix bundle
          u = np.linspace(0, 6*np.pi, N).astype(np.float32)
          return np.stack([0.6*np.cos(u), 0.6*np.sin(u), 0.3*np.sin(5*u)], axis=1)
      # flat disk (logo canvas)
      ang = np.linspace(0, 2*np.pi, N, endpoint=False).astype(np.float32)
      rad = 0.9*np.sqrt(np.linspace(0,1,N)).astype(np.float32)
      return np.stack([rad*np.cos(ang), rad*np.sin(ang), np.zeros(N, np.float32)], axis=1)

  def update(self, H_bits: float, S_field: float, caption_text: Optional[str]):
      H = float(np.clip(H_bits, 0.0, 1.0))
      S = float(np.clip(S_field, 0.0, 1.0))
      k_spring = 0.6 + 0.8*(1.0 - H)

