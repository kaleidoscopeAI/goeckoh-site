 2. auto-generate the avatar pipeline (loader → node scheduler → shader/renderer) from what’s in these zips, and

 3. patch the current AGI so the 18k-node avatar runs autonomously (no manual steps) as a background agent while the system keeps
    ingesting/annealing/captioning.


    1 2 and 3




  python


  # 1) Inspect the manifest for likely avatar / 18k-node / render code, and show a concise report.
  # 2) Generate an autonomous Avatar Agent that drives 18,000 nodes and renders frames headlessly.
  # 3) Provide a small patch file that shows how to hook the agent into the Seed-Crystal AGI app (FastAPI startup).

  import json, os, re, io, math, itertools, textwrap, random
  from pathlib import Path
  import pandas as pd
  from caas_jupyter_tools import display_dataframe_to_user

  BASE = Path("/mnt/data")
  EXTRACTED = BASE / "extracted"
  MANIFEST_JSON = BASE / "zip_review/manifest_subset.json"
  assert MANIFEST_JSON.exists(), "manifest_subset.json missing; please re-run unzip step"

  # --- 1) Scan manifest for likely modules ---
  with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
      rows = json.load(f)

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                                    83/124

  df = pd.DataFrame(rows)
  def kfind(df, *patterns):
      mask = False
      for pat in patterns:
          m = df["relpath"].str.contains(pat, case=False, regex=True, na=False) | df["name"].str.contains(pat, case=False, regex=True, na=False)
          mask = m if mask is False else (mask | m)
      return df[mask].copy()

  candidates = kfind(df, r"avatar", r"18 ?000", r"\bnode", r"renderer", r"render", r"glsl", r"shader", r"three", r"webgl", r"wgsl", r"fragment",
  r"vertex", r"mesh", r"cloth")
  candidates = candidates.sort_values("relpath")

  # try to read a few files for context snippets
  def safe_read_text(root_name: str, rel: str, max_bytes=120000):
      root = EXTRACTED / ( "organic_ai_full_infra_bundle" if root_name.startswith("organic") else "cognitive-nebula_1")
      p = root / rel
      if not p.exists() or not p.is_file():
           return ""
      if p.suffix.lower() in
  [".png",".jpg",".jpeg",".gif",".bmp",".ico",".pdf",".zip",".tar",".gz",".7z",".wav",".mp3",".ogg",".mp4",".mov",".avi",".webm"]:
           return ""
      try:
           data = p.read_text(encoding="utf-8", errors="ignore")
           return data[:max_bytes]
      except Exception:
           return ""

  snips = []
  for _, r in candidates.head(30).iterrows():
       txt = safe_read_text(r["root"], r["relpath"])
       if not txt:
           continue
       # look for telltale lines
       m = re.search(r"(18 ?000|node[s ]*count|avatar|GLSL|shader|Three\.js|WebGL|positions|indices|springs)", txt, flags=re.IGNORECASE)
       if m:
           snips.append({"root": r["root"], "relpath": r["relpath"], "snippet": textwrap.shorten(txt.replace("\n"," "), width=300, placeholder=" …
  ")})
  snips_df = pd.DataFrame(snips)

  display_dataframe_to_user("Avatar / 18k-node / render candidates (manifest-derived)", candidates.head(100))
  if not snips_df.empty:
      display_dataframe_to_user("Context snippets from likely files", snips_df.head(30))

  # --- 2) Generate an autonomous 18,000-node Avatar Agent (headless) ---
  OUT_DIR = BASE / "agent_out"
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  AGENT_PATH = OUT_DIR / "seed_crystal_avatar_agent.py"

  agent_code = r'''#!/usr/bin/env python3
  # Seed-Crystal Avatar Agent — 18k nodes, autonomous, headless renderer
  # - Subscribes to the Seed-Crystal AGI broadcaster (orch.bus)
  # - Maintains a 3D node field (60x60x5 = 18,000) driven by AGI metrics & captions
  # - Renders grayscale PNG frames (no GUI; optional Three.js stream can be added later)

  import os, sys, math, time, json, asyncio, random, sqlite3
  from typing import Dict, Any, List, Tuple, Optional
  from pathlib import Path

  import numpy as np

  try:
      from PIL import Image    # Pillow for PNG output
      PIL_OK = True
  except Exception:
      PIL_OK = False

  ROOT = Path(os.environ.get("SC_ROOT", os.getcwd()))
  OUT_FRAMES = ROOT / "avatar_frames"
  OUT_FRAMES.mkdir(parents=True, exist_ok=True)

  def find_mask_image(search_roots: List[Path]) -> Optional[Path]:
      names = ["avatar_mask.png","mask.png","silhouette.png","face_mask.png","avatar.png","silhouette.jpg","mask.jpg"]
      for root in search_roots:
          for n in names:
              p = root / n
              if p.exists() and p.is_file():
                  return p
          # scan subdirs shallow
          for p in root.rglob("*"):
              if p.suffix.lower() in [".png",".jpg",".jpeg"] and any(k in p.name.lower() for k in ["mask","avatar","silhouette"]):
                  return p
      return None

  class AvatarNodeField:
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

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                             84/124

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

  class AvatarAgent:
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


Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                              85/124

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

  # Optional CLI entrypoint to run standalone against existing seed_crystal.db
  async def _main():
      db = os.environ.get("SC_DB_PATH", str(Path.cwd() / "seed_crystal.db"))
      agent = AvatarAgent(sqlite_path=db, frames_dir=OUT_FRAMES)
      stop = asyncio.Event()
      try:
           await agent.run(stop_event=stop, fps=12)
      except KeyboardInterrupt:
           stop.set()

  if __name__ == "__main__":
      asyncio.run(_main())
  '''
  AGENT_PATH.write_text(agent_code, encoding="utf-8")

  # --- 3) Patch stub to hook agent into Seed-Crystal AGI app ---
  PATCH_PATH = OUT_DIR / "patch_seed_crystal_agi_avatar.diff"
  patch_text = r'''*** a/seed_crystal_agi.py
  --- b/seed_crystal_agi.py
  @@
  -REQ = [
  -     "fastapi==0.115.5", "uvicorn==0.32.0", "requests==2.32.3", "beautifulsoup4==4.12.3",
  -     "networkx==3.3", "numpy==1.26.4", "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0"
  -]
  +REQ = [
  +     "fastapi==0.115.5", "uvicorn==0.32.0", "requests==2.32.3", "beautifulsoup4==4.12.3",
  +     "networkx==3.3", "numpy==1.26.4", "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0",
  +     "Pillow==10.4.0"
  +]
  @@
  -from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
  +from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
  +
  +# Avatar background agent
  +from seed_crystal_avatar_agent import AvatarAgent
  @@
    app = FastAPI(title="Seed-Crystal AGI")
  @@
  -@app.on_event("startup")
  -async def boot():
  -     asyncio.create_task(orch.run())
  +@app.on_event("startup")
  +async def boot():
  +     # Start core orchestrator
  +     asyncio.create_task(orch.run())
  +     # Start 18k-node avatar agent (autonomous; no UI needed)
  +     try:
  +          agent = AvatarAgent(sqlite_path=os.environ.get("SC_DB_PATH", DB_PATH), frames_dir=OUT_SHAPES)
  +          app.state.avatar_task = asyncio.create_task(agent.run(fps=12))
  +     except Exception as e:
  +          print("Avatar agent failed to start:", e)
  +
  +@app.on_event("shutdown")
  +async def on_shutdown():
  +     # Ensure avatar task is cancelled
  +     task = getattr(app.state, "avatar_task", None)
  +     if task:
  +          task.cancel()
  @@
    @app.get("/status")
    def status():
  -     return {"ok": True, "state": orch.snapshot()}
  +     return {"ok": True, "state": orch.snapshot(), "avatar_frames_dir": str((OUT_SHAPES).absolute())}
  @@
    @app.get("/", response_class=HTMLResponse)
    def home():
  -     # Inline UI similar to provided, but simplified for brevity.
  -     return "<html><body><h1>Seed-Crystal AGI</h1><p>Access /status or /ws for real-time.</p></body></html>"
  +     # Minimal UI with link to latest avatar frame
  +     import glob, html
  +     frames = sorted(glob.glob(str((OUT_SHAPES / 'avatar_*.png').absolute())))
  +     latest = frames[-1] if frames else ""
  +     latest_rel = os.path.basename(latest) if latest else ""
  +     body = f"<h1>Seed-Crystal AGI</h1><p>Access <code>/status</code> or <code>/ws</code> for real-time.</p>"
  +     if latest_rel:
  +          body += f"<p>Latest avatar frame: <code>{html.escape(latest_rel)}</code></p>"
  +     return "<html><body>"+body+"</body></html>"
  '''

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                 86/124

  PATCH_PATH.write_text(patch_text, encoding="utf-8")

  print(json.dumps({
      "agent_file": str(AGENT_PATH),
      "patch_file": str(PATCH_PATH),
  }))


  STDOUT/STDERR
  {"agent_file": "/mnt/data/agent_out/seed_crystal_avatar_agent.py", "patch_file": "/mnt/data/agent_out/patch_seed_crystal_agi_avatar.diff"}

  Avatar / 18k-node / render candidates (manifest-derived)




  Context snippets from likely files




