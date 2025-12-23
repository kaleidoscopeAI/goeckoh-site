  def __init__(self, n=18000, seed=123):
      self.n = int(n)
      rng = np.random.RandomState(seed)
      # start near a sphere shell for interesting motion
      r = 0.8 * np.cbrt(rng.rand(self.n))
      theta = 2*np.pi*rng.rand(self.n)
      phi = np.arccos(2*rng.rand(self.n)-1)
      x = r*np.sin(phi)*np.cos(theta)
      y = r*np.sin(phi)*np.sin(theta)
      z = r*np.cos(phi)
      self.pos = np.stack([x,y,z], axis=1).astype(np.float32)      # (N,3)
      self.vel = np.zeros_like(self.pos, dtype=np.float32)
      self._bytes = None
      self._lock = threading.Lock()
      self.shape_id = 0

  def _target_field(self, shape_id:int) -> np.ndarray:
      """Return a soft target position per node for a given shape id."""
      N = self.n
      t = shape_id % 4
      P = self.pos
      if t == 0: # sphere
          radius = 0.9
          return radius * P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-6)
      if t == 1: # torus (R, r)
          R, r = 0.8, 0.25
          x, y, z = P[:,0], P[:,1], P[:,2]
          q = np.sqrt(x*x + y*y)+1e-6
          tx = (R * x / q) if N==1 else (R * (x / q))
          ty = (R * y / q) if N==1 else (R * (y / q))
          tz = 0*z
          # push toward minor tube via z and radial diff
          rx = tx * (1 + (r*(x-tx)))
          ry = ty * (1 + (r*(y-ty)))
          rz = (z/np.maximum(np.abs(z),1e-6)) * r
          return np.stack([rx, ry, rz], axis=1).astype(np.float32)
      if t == 2: # helix bundle
          k = 5.0
          u = np.linspace(0, 4*np.pi, N).astype(np.float32)
          return np.stack([0.6*np.cos(u), 0.6*np.sin(u), 0.6*np.sin(k*u)/k], axis=1)
      # t == 3: flat disk (logo canvas)
      ang = np.linspace(0, 2*np.pi, N, endpoint=False).astype(np.float32)
      rad = 0.9*np.sqrt(np.linspace(0,1,N)).astype(np.float32)
      return np.stack([rad*np.cos(ang), rad*np.sin(ang), np.zeros(N, np.float32)], axis=1)

  @staticmethod
  def _hash_str(s: str) -> int:
      h = 1469598103934665603
      for ch in s.encode("utf-8", "ignore"):
          h ^= ch
          h *= 1099511628211
          h &= (1<<64)-1
      return h if h>=0 else -h

  def update(self, H_bits: float, S_field: float, caption_text: str | None):
      # dynamics params from energetics
      H = float(np.clip(H_bits, 0.0, 1.0))
      S = float(np.clip(S_field, 0.0, 1.0))
      k_spring = 0.6 + 0.8*(1.0 - H)         # tighter as bits stabilize
      swirl = 0.2 + 1.0*S                    # more swirl = more field tension
      noise = 0.02 + 0.15*(1.0 - S)          # noisy when field coherent is low
      damp = 0.86 + 0.08*H

       sid = self._hash_str(caption_text or "") % 99991

