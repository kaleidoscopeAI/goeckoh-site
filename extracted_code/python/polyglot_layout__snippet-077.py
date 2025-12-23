  """
  18,000 nodes (60x60x5). If spec/avatar_spec.zip exists, tries:
    - nodes.npy   (shape [N,3] in [-1,1])
    - nodes.csv   (x,y,z rows)
  Each tick, field forces from energetics move the mesh; renders PNG & periodic GIFs.
  """
  def __init__(self, out_dir: Path, width=768, height=768):
      self.out = out_dir; self.out.mkdir(parents=True, exist_ok=True)
      self.W, self.H = width, height
      self.N = 18000
      self.pos = self._load_or_make_positions()          # (N,3)
      self.vel = np.zeros_like(self.pos)
      self.col = np.ones((self.N, 3), dtype=np.float32)   # rgb in [0,1]
      self._rng = np.random.RandomState(123)
      self._frames_for_gif: List[Path] = []

  def _load_or_make_positions(self) -> np.ndarray:
      zip_path = SPEC_DIR / "avatar_spec.zip"
      if zip_path.exists():
          try:
               with zipfile.ZipFile(zip_path, 'r') as zf:
                   if "nodes.npy" in zf.namelist():
                       import io
                       arr = np.load(io.BytesIO(zf.read("nodes.npy")))
                       assert arr.ndim==2 and arr.shape[1]==3
                       return self._normalize(arr)
                   if "nodes.csv" in zf.namelist():
                       txt = zf.read("nodes.csv").decode("utf-8", "ignore").splitlines()
                       rows = list(csv.reader(txt))
                       pts = []
                       for r in rows:
                           if len(r)>=3:
                               try:
                                    pts.append([float(r[0]), float(r[1]), float(r[2])])
                               except: pass
                       arr = np.array(pts, dtype=np.float64)
                       assert arr.ndim==2 and arr.shape[1]==3
                       return self._normalize(arr)
          except Exception as e:
               print(">> Spec load failed, using default grid:", e)
      # Default: exact 60×60×5 lattice in [-1,1]^3
      xs, ys, zs = [np.linspace(-1,1,n) for n in (60,60,5)]
      X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')
      arr = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
      assert arr.shape[0] == 18000
      return arr.astype(np.float64)

  def _normalize(self, arr: np.ndarray) -> np.ndarray:
      # scale to [-1,1] cube
      mn = arr.min(axis=0); mx = arr.max(axis=0); span = (mx - mn); span[span==0]=1.0
      norm = (arr - mn)/span; norm = norm*2.0 - 1.0
      if norm.shape[0] > self.N:
          idx = self._rng.choice(norm.shape[0], size=self.N, replace=False)
          norm = norm[idx]
      elif norm.shape[0] < self.N:
          # tile to reach N
          reps = int(math.ceil(self.N / norm.shape[0]))
          norm = np.tile(norm, (reps,1))[:self.N]
      return norm.astype(np.float64)

  def step(self, en: Dict[str,float], dt: float = 0.05, damp: float = 0.96):
      # Field forces from energetics
      H_bits = float(en.get("H_bits", 0.0))
      S_field = float(en.get("S_field", 0.0))
      L = float(en.get("L", 0.0))

      # swirl strength from S_field; expansion/compression from H_bits
      swirl = 0.5 + 2.5 * S_field
      expand = 1.0 + 0.6 * (0.5 - H_bits)   # >1.0 expands if low uncertainty

      # rotate around z (swirl), small jitter, then relax toward scaled lattice (expand)
      theta = swirl * 0.02
      c, s = math.cos(theta), math.sin(theta)
      R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
      target = self.pos * expand
      force = (target - self.pos)

      # apply rotation & force
      self.pos = (self.pos @ R.T)
      self.vel = damp*self.vel + 0.15*force + 0.01*self._rng.normal(0,1.0,size=self.pos.shape)
      self.pos = np.clip(self.pos + dt*self.vel, -1.4, 1.4)

      # color map: H_bits -> cool/warm blend, S_field -> saturation, L -> brightness
      base = np.stack([
          0.5 + 0.5*(1.0-H_bits),             # R
          0.5 + 0.4*(0.5-S_field),            # G
          0.7 + 0.3*H_bits                    # B
      ], axis=0).astype(np.float32)
      self.col[:] = np.clip(base, 0.0, 1.0)

  def _project(self) -> Tuple[np.ndarray, np.ndarray]:
      # simple perspective projection
      cam_z = 3.2
      z = self.pos[:,2] + cam_z

