  """
  18k nodes arranged 60x60x5 in a cube [-1,1]^3.
  Each tick: spring toward a target silhouette projected onto XY plane.
  Caption & energetics modulate stiffness and jitter for expressive motion.
  """
  def __init__(self, seed: int = 7):
      self.rng = np.random.RandomState(seed)
      self.nx, self.ny, self.nz = 60, 60, 5 # 60*60*5 = 18,000
      X, Y, Z = np.meshgrid(
          np.linspace(-1,1,self.nx),
          np.linspace(-1,1,self.ny),
          np.linspace(-1,1,self.nz),
          indexing="ij"
      )
      self.pos = np.stack([X, Y, Z], axis=-1).reshape(-1,3) # (N,3)
      self.vel = np.zeros_like(self.pos)
      self.N = self.pos.shape[0]
      # default target: concentric circle face if no mask found
      self.target = None # (H,W) binary mask in image coords
      self.canvas_size = (720, 720) # H,W
      self.fx = 0.9 # spring factor base
      self.damp = 0.94
      self.jitter = 0.001

     def set_target_from_mask(self, mask_img: np.ndarray):

