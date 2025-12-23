      z[z<1e-3]=1e-3
      fx = fy = 420.0
      u = (self.pos[:,0]/z)*fx + self.W/2
      v = (self.pos[:,1]/z)*fy + self.H/2
      return np.stack([u,v],axis=1), z

  def render(self, tick: int) -> Path:
      uv, z = self._project()
      # depth sort
      order = np.argsort(z)[::-1]
      uv = uv[order]; col = self.col[order]; z = z[order]
      img = Image.new("RGB", (self.W,self.H), (6,6,10))
      draw = ImageDraw.Draw(img, "RGBA")

      # point size from depth; sample subset for speed
      N = uv.shape[0]
      step = 1
      if N > 24000: step = 2
      idxs = range(0, N, step)
      for i in idxs:
          x,y = uv[i]
          if x< -4 or x >= self.W+4 or y< -4 or y >= self.H+4: continue
          d = z[i]
          r = int(max(1, 3.5 - 0.6*d)) # nearer → bigger
          c = tuple((col[i]*255).astype(np.uint8).tolist())
          draw.ellipse((x-r, y-r, x+r, y+r), fill=c+(180,))

      path = self.out / f"frame_{tick:06d}.png"
      img.save(path, "PNG", optimize=True)
      self._frames_for_gif.append(path)
      if len(self._frames_for_gif) >= 30:
          gif_path = self.out / f"avatar_{tick:06d}.gif"
          frames = [Image.open(p) for p in self._frames_for_gif[-30:]]
          frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=60, loop=0)
      return path

