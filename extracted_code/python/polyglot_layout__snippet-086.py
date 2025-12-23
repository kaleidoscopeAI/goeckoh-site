  def __init__(self, sqlite_path: str, frames_dir: Path):
      self.sqlite_path = sqlite_path
      self.frames_dir = frames_dir
      self.field = AvatarNodeField(seed=42)
      # try to find an existing mask in extracted zips (best-effort)
      search_roots = [
          Path("/mnt/data/extracted/organic_ai_full_infra_bundle"),
          Path("/mnt/data/extracted/cognitive-nebula_1"),
          ROOT
      ]
      try:
          target = find_mask_image(search_roots)
          if target and target.exists():
               from PIL import Image
               m = Image.open(target).convert("L")
               arr = (np.array(m).astype(np.float32)/255.0)
               self.field.set_target_from_mask(arr)
      except Exception:
          pass

  def _fetch_latest_energetics(self) -> Tuple[float,float]:
      # Returns (H_bits, S_field)
      try:
          con = sqlite3.connect(self.sqlite_path)
          cur = con.cursor()
          cur.execute("SELECT hbits, sfield FROM energetics ORDER BY id DESC LIMIT 1")
          row = cur.fetchone()
          con.close()
          if row and row[0] is not None and row[1] is not None:
               return float(row[0]), float(row[1])
      except Exception:
          pass
      return 0.2, 0.2


