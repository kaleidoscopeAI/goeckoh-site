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
      swirl = 0.2 + 1.0*S
      noise = 0.02 + 0.15*(1.0 - S)
      damp = 0.86 + 0.08*H

      sid = self._hash_str(caption_text or "") % 99991
      target = self._target_field(sid)

      P = self.pos; V = self.vel
      F_spring = k_spring*(target - P)
      sw = np.stack([-P[:,1], P[:,0], 0*P[:,2]], axis=1).astype(np.float32) * swirl
      F = F_spring + sw + noise*np.random.normal(0,1.0,P.shape).astype(np.float32)

      V = damp*V + 0.03*F
      P = P + V
      np.clip(P, -1.0, 1.0, out=P)

      self.pos, self.vel = P, V
      with self._lock:
          self._bytes = struct.pack("<I", self.n) + P.astype(np.float32).tobytes()

  def frame_bytes(self) -> Optional[bytes]:
      with self._lock:
          return self._bytes

