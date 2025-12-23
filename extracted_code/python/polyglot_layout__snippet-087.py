  def _fetch_latest_caption_hash(self) -> float:
      try:
          con = sqlite3.connect(self.sqlite_path)
          cur = con.cursor()
          cur.execute("SELECT caption FROM captions ORDER BY id DESC LIMIT 1")
          row = cur.fetchone()
          con.close()
          if row and row[0]:
               import hashlib
               h = hashlib.sha256(row[0].encode('utf-8','ignore')).hexdigest()
               return (int(h[:8], 16) / 0xffffffff) * 2.0 - 1.0
      except Exception:
          pass
      return 0.0

  async def run(self, stop_event: Optional[asyncio.Event] = None, fps: int = 10):
      # Runs indefinitely, reading DB signals and producing frames
      t = 0
      dt = 1.0/max(1,fps)
      while True:
          if stop_event and stop_event.is_set():
              break
          H_bits, S_field = self._fetch_latest_energetics()
          c_hash = self._fetch_latest_caption_hash()
          self.field.step(H_bits=H_bits, S_field=S_field, caption_hash=c_hash, dt=dt*0.6)
          # render every few steps
          if t % 2 == 0:
              out = self.frames_dir / f"avatar_{t:06d}.png"
              self.field.render_png(out, point_size=2)
          t += 1
          await asyncio.sleep(dt)

