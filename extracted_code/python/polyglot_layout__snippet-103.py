      acc += self.rng.normal(0.0, self.jitter, size=acc.shape)
      self.vel = (self.vel + dt*acc) * self.damp; self.pos = self.pos + dt*self.vel
      self.pos = np.clip(self.pos, -1.3, 1.3)
  def render_png(self, path: Path, point_size: int = 1):
      try:
          from PIL import Image
      except Exception:
          return False, "Pillow not available"
      H, W = self.canvas_size; xy = self.pos[:, :2]
      u = ((xy[:,0] + 1.0) * 0.5) * (W-1); v = ((xy[:,1] + 1.0) * 0.5) * (H-1)
      ui = np.clip(u.astype(np.int32), 0, W-1); vi = np.clip(v.astype(np.int32), 0, H-1)
      img = np.zeros((H, W), dtype=np.uint8); img[vi, ui] = 255
      if point_size > 1:
          from scipy.ndimage import grey_dilation
          img = grey_dilation(img, size=(point_size, point_size))
      Image.fromarray(img).save(path); return True, str(path)

