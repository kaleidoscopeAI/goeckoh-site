      # Expect mask in HxW (0..1); keep as 720x720 internal
      H, W = self.canvas_size
      from PIL import Image
      m = Image.fromarray((mask_img*255).astype(np.uint8)).resize((W,H))
      m = np.array(m).astype(np.float32)/255.0
      self.target = (m > 0.5).astype(np.float32)

  def _default_mask(self):
      H,W = self.canvas_size
      yy, xx = np.mgrid[0:H,0:W]
      cx, cy = W/2, H/2
      r = 0.35*min(H,W)
      mask = (((xx-cx)**2 + (yy-cy)**2) <= r*r).astype(np.float32)
      return mask

  def step(self, H_bits: float = 0.2, S_field: float = 0.2, caption_hash: float = 0.0, dt: float = 0.06):
      # Modulate dynamics by energetics & caption content
      stiff = 0.5 + 0.8*(1.0 - H_bits)           # more order => stiffer
      spread = 0.3 + 0.7*(S_field)               # more field tension => spread
      rng_j = 0.0005 + 0.003*abs(caption_hash)   # caption alters jitter

      self.fx = 0.85 + 0.3*stiff
      self.jitter = (self.jitter*0.9 + rng_j*0.1)
      self.damp = 0.92 + 0.06*(1.0-stiff)

      # Project target shape into XY on [-1,1]^2 plane
      target = self.target if self.target is not None else self._default_mask()
      H,W = target.shape
      # normalized coords
      xy = self.pos[:, :2]
      # map [-1,1] -> [0,W/H]
      u = ((xy[:,0] + 1.0) * 0.5) * (W-1)
      v = ((xy[:,1] + 1.0) * 0.5) * (H-1)
      ui = np.clip(u.astype(np.int32), 0, W-1)
      vi = np.clip(v.astype(np.int32), 0, H-1)
      on = target[vi, ui] # (N,) 0 or 1

      # Desired Z as function of silhouette boundary: inside pull to plane z=0; outside push to z=spread
      desired_z = np.where(on > 0.5, 0.0, spread)

      # Forces: planar attraction + slight XY swirl for liveliness
      z_force = (desired_z - self.pos[:,2]) * self.fx
      swirl = np.stack([-xy[:,1], xy[:,0]], axis=-1) * 0.02 * (0.5 - H_bits)
      acc = np.zeros_like(self.pos)
      acc[:,2] = z_force
      acc[:,:2] += swirl

      # Caption-driven noise
      acc += self.rng.normal(0.0, self.jitter, size=acc.shape)

      # Integrate
      self.vel = (self.vel + dt*acc) * self.damp
      self.pos = self.pos + dt*self.vel
      # keep in bounds
      self.pos = np.clip(self.pos, -1.3, 1.3)

  def render_png(self, path: Path, point_size: int = 1):
      if not PIL_OK:
          return False, "Pillow not available"
      H, W = self.canvas_size
      # Simple orthographic projection XY -> image
      xy = self.pos[:, :2]
      u = ((xy[:,0] + 1.0) * 0.5) * (W-1)
      v = ((xy[:,1] + 1.0) * 0.5) * (H-1)
      ui = np.clip(u.astype(np.int32), 0, W-1)
      vi = np.clip(v.astype(np.int32), 0, H-1)
      img = np.zeros((H, W), dtype=np.uint8)
      img[vi, ui] = 255
      if point_size > 1:
          # very small dilation to improve visibility
          from scipy.ndimage import grey_dilation
          img = grey_dilation(img, size=(point_size, point_size))
      Image.fromarray(img).save(path)
      return True, str(path)

