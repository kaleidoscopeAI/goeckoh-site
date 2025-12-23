  def __init__(self, sqlite_path: str, frames_dir: Path):
      self.sqlite_path = sqlite_path; self.frames_dir = frames_dir
      self.frames_dir.mkdir(parents=True, exist_ok=True)
      self.field = AvatarNodeField(seed=42)
      # try to pick a mask from working dirs (optional)
      search_roots = [ROOT, Path("/mnt/data/extracted/organic_ai_full_infra_bundle"), Path("/mnt/data/extracted/cognitive-nebula_1")]
      try:
          from PIL import Image
          for root in search_roots:
               for p in root.rglob("*"):
                   if p.suffix.lower() in [".png",".jpg",".jpeg"] and any(k in p.name.lower() for k in ["mask","avatar","silhouette"]):
                       m = Image.open(p).convert("L"); self.field.set_target_from_mask((np.array(m).astype(np.float32)/255.0)); raise
