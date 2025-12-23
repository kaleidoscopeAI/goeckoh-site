  def __init__(self, seed: int = 7):
      self.rng = np.random.RandomState(seed)
      self.nx, self.ny, self.nz = 60, 60, 5 # 60*60*5 = 18,000
      X, Y, Z = np.meshgrid(np.linspace(-1,1,self.nx), np.linspace(-1,1,self.ny), np.linspace(-1,1,self.nz), indexing="ij")
      self.pos = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
      self.vel = np.zeros_like(self.pos); self.N = self.pos.shape[0]
      self.target = None; self.canvas_size = (720, 720)
      self.fx = 0.9; self.damp = 0.94; self.jitter = 0.001
  def set_target_from_mask(self, mask_img: np.ndarray):
      from PIL import Image
      H, W = self.canvas_size
      m = Image.fromarray((mask_img*255).astype(np.uint8)).resize((W,H))
      m = np.array(m).astype(np.float32)/255.0; self.target = (m > 0.5).astype(np.float32)
  def _default_mask(self):
      H,W = self.canvas_size
      yy, xx = np.mgrid[0:H,0:W]; cx, cy = W/2, H/2; r = 0.35*min(H,W)
      return (((xx-cx)**2 + (yy-cy)**2) <= r*r).astype(np.float32)
  def step(self, H_bits: float = 0.2, S_field: float = 0.2, caption_hash: float = 0.0, dt: float = 0.06):
      stiff = 0.5 + 0.8*(1.0 - H_bits); spread = 0.3 + 0.7*(S_field); rng_j = 0.0005 + 0.003*abs(caption_hash)
      self.fx = 0.85 + 0.3*stiff; self.jitter = (self.jitter*0.9 + rng_j*0.1); self.damp = 0.92 + 0.06*(1.0-stiff)
      target = self.target if self.target is not None else self._default_mask()
      H,W = target.shape
      xy = self.pos[:, :2]; u = ((xy[:,0] + 1.0) * 0.5) * (W-1); v = ((xy[:,1] + 1.0) * 0.5) * (H-1)
      ui = np.clip(u.astype(np.int32), 0, W-1); vi = np.clip(v.astype(np.int32), 0, H-1)
      on = target[vi, ui]; desired_z = np.where(on > 0.5, 0.0, spread)
      z_force = (desired_z - self.pos[:,2]) * self.fx
      swirl = np.stack([-xy[:,1], xy[:,0]], axis=-1) * 0.02 * (0.5 - H_bits)
      acc = np.zeros_like(self.pos); acc[:,2] = z_force; acc[:,:2] += swirl

