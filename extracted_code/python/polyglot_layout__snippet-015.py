  # Avatar: 18,000 nodes (60×60×5) by default; can load your spec from spec/avatar_spec.zip (nodes.npy or nodes.csv).

  import os, sys, subprocess, venv, json, time, asyncio, math, sqlite3, traceback, wave, random, zipfile, csv
  from dataclasses import dataclass
  from typing import Dict, Any, List, Tuple, Iterable, Optional
  from pathlib import Path

  # ---------------- Bootstrapping: create .venv and re-exec (offline deps only) ----------------
  ROOT = Path.cwd() / "onbrain_auto"
  VENV = ROOT / ".venv"
  REQ = [
      "fastapi==0.115.5", "uvicorn==0.32.0", "numpy==1.26.4", "networkx==3.3",
      "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0",
      "scipy==1.11.4", "sympy==1.13.3", "langdetect==1.0.9",
      "Pillow==10.4.0", "imageio==2.35.1"
  ]
  def ensure_venv_and_reexec():
      ROOT.mkdir(parents=True, exist_ok=True)
      if os.environ.get("OBA_BOOTED") == "1": return
      if not VENV.exists():
          print(">> Creating venv at", VENV)
          venv.create(VENV, with_pip=True)
      pip = VENV / ("Scripts/pip.exe" if os.name=="nt" else "bin/pip")
      py = VENV / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                                      72/124

      print(">> Upgrading pip and installing deps")
      subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
      subprocess.check_call([str(pip), "install"] + REQ)
      env = os.environ.copy(); env["OBA_BOOTED"] = "1"
      print(">> Relaunching inside venv")
      os.execvpe(str(py), [str(py), __file__], env)

  if __file__ == "<stdin>":
      script_path = ROOT / "onbrain_autonomous.py"
      script_path.write_text(sys.stdin.read(), encoding="utf-8")
      __file__ = str(script_path)

  ensure_venv_and_reexec()

  # ---------------- Imports (post-venv) ----------------
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.responses import HTMLResponse, JSONResponse
  import uvicorn
  import numpy as np
  from scipy.signal import get_window
  from sympy import sympify, simplify, Eq, solve
  from langdetect import detect, detect_langs
  from PIL import Image, ImageDraw
  import imageio

  # ---------------- Config & paths ----------------
  PORT = int(os.getenv("OBA_PORT", "8772"))
  HOST = os.getenv("OBA_HOST", "0.0.0.0")
  TICK_SEC = float(os.getenv("OBA_TICK_SEC", "0.5"))
  REFLECT_EVERY = int(os.getenv("OBA_REFLECT_EVERY", "4"))
  AUTON_INGEST_EVERY = int(os.getenv("OBA_AUTON_INGEST_EVERY", "10"))
  SIGMA0 = float(os.getenv("OBA_SIGMA0", "0.9"))
  GAMMA = float(os.getenv("OBA_GAMMA", "0.93"))
  SIGMA_MIN = float(os.getenv("OBA_SIGMA_MIN", "0.10"))
  DB_PATH = os.getenv("OBA_DB_PATH", str(ROOT / "onbrain.db"))

  INBOX = ROOT / "inbox"; INBOX.mkdir(parents=True, exist_ok=True)         # drop .txt/.md/.html here; autopilot ingests
  SPEC_DIR = ROOT / "spec"; SPEC_DIR.mkdir(parents=True, exist_ok=True)     # put avatar_spec.zip here to override layout
  OUT_AUDIO = ROOT / "audio"; OUT_AUDIO.mkdir(parents=True, exist_ok=True)
  OUT_AVATAR = ROOT / "avatar_frames"; OUT_AVATAR.mkdir(parents=True, exist_ok=True)

  # ---------------- Utilities ----------------
  def write_wav_mono16(path: Path, sr: int, samples: Iterable[float]) -> None:
      import wave
      x = np.asarray(list(samples), dtype=np.float32)
      if x.size == 0: x = np.zeros(1, dtype=np.float32)
      x = np.clip(x, -1.0, 1.0)
      y = (x * 32767.0).astype(np.int16)
      with wave.open(str(path), 'wb') as wf:
          wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y.tobytes())

  def stft_mag(x: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
      w = get_window("hann", win)
      if len(x) < win: x = np.pad(x, (0, win - len(x)))
      T = 1 + (len(x) - win)//hop
      F = win//2 + 1
      X = np.zeros((F, T), dtype=np.float64)
      for t in range(T):
          s = t*hop
          seg = x[s:s+win]
          if len(seg) < win: seg = np.pad(seg, (0, win-len(seg)))
          spec = np.fft.rfft(seg * w)
          X[:, t] = np.abs(spec)
      return X

  # --------------- Multilingual hashing embeddings (unicode 1–4grams) ---------------
  _P = (1<<61)-1; _A = 1371731309; _B = 911382323
  def sha_to_u64(s: str, salt: str="")->int:
      import hashlib
      h=hashlib.sha256((salt+s).encode("utf-8","ignore")).digest()
      return int.from_bytes(h[:8],"little")
  def u_hash(x:int, a:int=_A, b:int=_B, p:int=_P, D:int=512)->int:
      return ((a*x+b)%p)%D
  def sign_hash(x:int)->int:
      return 1 if (x ^ (x>>1) ^ (x>>2)) & 1 else -1
  def ngrams(s:str, n_min=1, n_max=4)->List[str]:
      s = "".join(ch for ch in s.lower())
      grams = []
      for n in range(n_min, n_max+1):
          for i in range(len(s)-n+1):
              grams.append(s[i:i+n])
      return grams
  def embed_text(text:str, D:int=512)->np.ndarray:
      v=np.zeros(D,dtype=np.float64)
      for g in ngrams(text,1,4):
          x=sha_to_u64(g)
          d=u_hash(x,D=D)
          s=sign_hash(sha_to_u64(g,"sign"))
          v[d]+=s
      nrm=np.linalg.norm(v)+1e-9
      return v/nrm

  # --------------- Crystallization & energetics ----------------
  def cos_sim(a:np.ndarray, b:np.ndarray, eps:float=1e-9)->float:
      return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+eps))
  def knn_idx(E:np.ndarray, i:int, k:int=8)->List[int]:
      x=E[i]; sims=(E@x)/(np.linalg.norm(E,axis=1)*(np.linalg.norm(x)+1e-9)+1e-12)
      order=np.argsort(-sims); return [j for j in order if j!=i][:k]
  def mc_var(E:np.ndarray, i:int, k:int, sigma:float, M:int=6, rng=None)->float:
      if rng is None: rng=np.random.RandomState(7)
      idx=knn_idx(E,i,k=max(1,min(k,E.shape[0]-1))); vals=[]; D=E.shape[1]
      for _ in range(M):
          ei=E[i]+sigma*rng.normal(0.0,1.0,size=D); ei/= (np.linalg.norm(ei)+1e-9)
          sims=[]
          for j in idx:
              ej=E[j]+sigma*rng.normal(0.0,1.0,size=D); ej/= (np.linalg.norm(ej)+1e-9)
              sims.append(cos_sim(ei,ej))
          vals.append(max(sims) if sims else 0.0)
      return float(np.var(vals))

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                          73/124

  def stability(var_sigma:float)->float: return 1.0/(1.0+var_sigma)
  def anneal_sigma(sigma0:float, gamma:float, step:int, sigma_min:float)->float:
      return max(sigma0*(gamma**step), sigma_min)
  def ring_edges(N:int,k:int=6)->np.ndarray:
      edges=set()
      for i in range(N):
          edges.add(tuple(sorted((i,(i+1)%N))))
          for d in range(1,k//2+1): edges.add(tuple(sorted((i,(i+d)%N))))
      if not edges: return np.zeros((0,2),dtype=np.int32)
      return np.array(sorted(list(edges)),dtype=np.int32)
  def energetics(E:np.ndarray, S:np.ndarray, edges:np.ndarray, sigma:float)->dict:
      if len(edges)==0:
          return {"H_bits": float(np.mean(1.0-S) if E.shape[0] else 0.0), "S_field":0.0, "L":0.0}
      w=np.zeros(len(edges));
      rng=np.random.RandomState(11)
      for k,(i,j) in enumerate(edges):
          ei,ej=E[i],E[j]
          sims=[]
          for _ in range(4):
              ein=ei+sigma*rng.normal(0.0,1.0,size=ei.shape); ein/= (np.linalg.norm(ein)+1e-9)
              ejn=ej+sigma*rng.normal(0.0,1.0,size=ej.shape); ejn/= (np.linalg.norm(ejn)+1e-9)
              sims.append(float(np.dot(ein,ejn)))
          w[k]=float(np.mean(sims))
      tau=1.0-w
      H_bits=float(np.mean(1.0-S) if E.shape[0] else 0.0)
      S_field=float(np.mean(tau)); L=float(np.sum(tau*tau))
      return {"H_bits":H_bits,"S_field":S_field,"L":L}

  # --------------- Memory (SQLite) ----------------
  class Memory:
      def __init__(self, path:str, D:int=512):
          self.path=path; self.D=D; self._init()
      def _init(self):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("""CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, lang TEXT, text TEXT)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS embeds(id INTEGER PRIMARY KEY, vec BLOB)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, type TEXT, json TEXT)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS energetics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits
  REAL, sfield REAL, L REAL)""")
          con.commit(); con.close()
      def teach(self, text:str, lang:str):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("INSERT INTO facts(ts,lang,text) VALUES(?,?,?)",(time.time(), lang, text))
          fid=cur.lastrowid
          e=embed_text(text, D=self.D).astype(np.float32)
          cur.execute("INSERT OR REPLACE INTO embeds(id,vec) VALUES(?,?)",(fid, e.tobytes()))
          con.commit(); con.close(); return fid
      def embeddings(self, max_items:Optional[int]=None)->Tuple[np.ndarray,List[int]]:
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("SELECT id,vec FROM embeds ORDER BY id ASC"); rows=cur.fetchall(); con.close()
          if not rows: return np.zeros((0,0),dtype=np.float64), []
          ids=[int(r[0]) for r in rows]; arr=[np.frombuffer(r[1],dtype=np.float32).astype(np.float64) for r in rows]
          E=np.stack(arr,axis=0); E/= (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
          if max_items and len(ids)>max_items:
              idx=np.random.RandomState(123).choice(len(ids), size=max_items, replace=False)
              E=E[idx]; ids=[ids[i] for i in idx]
          return E, ids
      def log(self, tick:int, type_:str, data:dict):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("INSERT INTO traces(ts,tick,type,json) VALUES(?,?,?,?)",(time.time(), tick, type_, json.dumps(data)))
          con.commit(); con.close()

  # --------------- Sonification maps ----------------
  def synth_signal(seconds: float, sr: int, a_fn, m_fn, rho_fn, fc_fn, alpha: float = 0.8, beta: float = 0.4)->List[float]:
      n=int(seconds*sr); out=[]
      for i in range(n):
          t=i/sr; a=a_fn(t); m=m_fn(t); rho=rho_fn(t); fc=max(5.0, fc_fn(t))
          y = a*(1.0+beta*math.sin(2*math.pi*m*t))*math.sin(2*math.pi*fc*t + alpha*math.sin(2*math.pi*rho*t))
          out.append(y)
      return out
  def default_maps(H_bits:float, S_field:float, latency:float, fitness:float, fmin:float=110.0, fdelta:float=440.0):
      H=max(0.0,min(1.0,H_bits)); S=max(0.0,min(1.0,S_field)); L=max(0.0,min(1.0,latency)); F=max(0.0,min(1.0,fitness))
      def a_fn(t): return 0.25 + 0.5*(1.0-H)*(1.0-S)
      def m_fn(t): return 2.0 + 10.0*S
      def rho_fn(t): return 0.2 + 3.0*(1.0-L)
      def fc_fn(t): return fmin + fdelta*F
      return {"a":a_fn,"m":m_fn,"rho":rho_fn,"fc":fc_fn}

  # --------------- Polyglot helpers ----------------
  LANG_LABEL = {
      "en":"Answer", "es":"Respuesta", "fr":"Réponse", "de":"Antwort", "it":"Risposta",
      "pt":"Resposta", "nl":"Antwoord", "ru":"Ответ", "zh-cn":"      答复
                                                                  ", "zh-tw":"答覆  ",
      "ja":"回答 ", "ko":" 답변", "ar":"‫"اإلجابة‬
  }
  def label_for(lang:str)->str: return LANG_LABEL.get(lang.lower(), "Answer")

  # --------------- Domain solvers ----------------
  class MathSolver:
      @staticmethod
      def solve_expr(q:str)->Tuple[bool,str]:
          try:
              if "=" in q:
                  left,right=q.split("=",1)
                  expr_l=sympify(left); expr_r=sympify(right)
                  sol=solve(Eq(expr_l,expr_r))
                  return True, f"solutions: {sol}"
              expr=sympify(q)
              return True, f"{simplify(expr)}"
          except Exception as e:
              return False, f"math_error: {e}"

  class LogicPlanner:
      @staticmethod
      def plan(prompt:str)->List[str]:
          steps=[]
          s=prompt.strip().lower()
          if any(k in s for k in ["prove","show","why","because"]):
              steps += ["Define terms","List axioms/facts","Split into subgoals","Check counterexamples","Synthesize argument"]
          if any(k in s for k in ["design","build","create","implement"]):

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                            74/124

              steps += ["Clarify requirements","Sketch architecture","List modules","Draft algorithm","Test cases","Refine edges","Document"]
          if not steps: steps=["Clarify intent","Retrieve facts","Draft","Critique","Finalize"]
          return steps

  class Retriever:
      def __init__(self, mem:'Memory'): self.mem=mem
      def topk(self, query:str, k:int=6)->Tuple[List[int], List[float]]:
          E, ids = self.mem.embeddings(max_items=512)
          if E.size==0: return [], []
          qv = embed_text(query)
          sims = (E @ qv) / (np.linalg.norm(E,axis=1) * (np.linalg.norm(qv)+1e-9) + 1e-12)
          order = np.argsort(-sims)[:k]
          return [ids[i] for i in order], [float(sims[i]) for i in order]

  # --------------- AvatarSynth: 18k nodes (grid or ZIP spec), autonomous rendering ----------------
  class AvatarSynth:
      """
      18,000 nodes (60x60x5). If spec/avatar_spec.zip exists, tries:
        - nodes.npy   (shape [N,3] in [-1,1])
        - nodes.csv   (x,y,z rows)
      Each tick, field forces from energetics move the mesh; renders PNG & periodic GIFs.
      """
      def __init__(self, out_dir: Path, width=768, height=768):
          self.out = out_dir; self.out.mkdir(parents=True, exist_ok=True)
          self.W, self.H = width, height
          self.N = 18000
          self.pos = self._load_or_make_positions()          # (N,3)
          self.vel = np.zeros_like(self.pos)
          self.col = np.ones((self.N, 3), dtype=np.float32)   # rgb in [0,1]
          self._rng = np.random.RandomState(123)
          self._frames_for_gif: List[Path] = []

      def _load_or_make_positions(self) -> np.ndarray:
          zip_path = SPEC_DIR / "avatar_spec.zip"
          if zip_path.exists():
              try:
                   with zipfile.ZipFile(zip_path, 'r') as zf:
                       if "nodes.npy" in zf.namelist():
                           import io
                           arr = np.load(io.BytesIO(zf.read("nodes.npy")))
                           assert arr.ndim==2 and arr.shape[1]==3
                           return self._normalize(arr)
                       if "nodes.csv" in zf.namelist():
                           txt = zf.read("nodes.csv").decode("utf-8", "ignore").splitlines()
                           rows = list(csv.reader(txt))
                           pts = []
                           for r in rows:
                               if len(r)>=3:
                                   try:
                                        pts.append([float(r[0]), float(r[1]), float(r[2])])
                                   except: pass
                           arr = np.array(pts, dtype=np.float64)
                           assert arr.ndim==2 and arr.shape[1]==3
                           return self._normalize(arr)
              except Exception as e:
                   print(">> Spec load failed, using default grid:", e)
          # Default: exact 60×60×5 lattice in [-1,1]^3
          xs, ys, zs = [np.linspace(-1,1,n) for n in (60,60,5)]
          X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')
          arr = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
          assert arr.shape[0] == 18000
          return arr.astype(np.float64)

      def _normalize(self, arr: np.ndarray) -> np.ndarray:
          # scale to [-1,1] cube
          mn = arr.min(axis=0); mx = arr.max(axis=0); span = (mx - mn); span[span==0]=1.0
          norm = (arr - mn)/span; norm = norm*2.0 - 1.0
          if norm.shape[0] > self.N:
              idx = self._rng.choice(norm.shape[0], size=self.N, replace=False)
              norm = norm[idx]
          elif norm.shape[0] < self.N:
              # tile to reach N
              reps = int(math.ceil(self.N / norm.shape[0]))
              norm = np.tile(norm, (reps,1))[:self.N]
          return norm.astype(np.float64)

      def step(self, en: Dict[str,float], dt: float = 0.05, damp: float = 0.96):
          # Field forces from energetics
          H_bits = float(en.get("H_bits", 0.0))
          S_field = float(en.get("S_field", 0.0))
          L = float(en.get("L", 0.0))

          # swirl strength from S_field; expansion/compression from H_bits
          swirl = 0.5 + 2.5 * S_field
          expand = 1.0 + 0.6 * (0.5 - H_bits)   # >1.0 expands if low uncertainty

          # rotate around z (swirl), small jitter, then relax toward scaled lattice (expand)
          theta = swirl * 0.02
          c, s = math.cos(theta), math.sin(theta)
          R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
          target = self.pos * expand
          force = (target - self.pos)

          # apply rotation & force
          self.pos = (self.pos @ R.T)
          self.vel = damp*self.vel + 0.15*force + 0.01*self._rng.normal(0,1.0,size=self.pos.shape)
          self.pos = np.clip(self.pos + dt*self.vel, -1.4, 1.4)

          # color map: H_bits -> cool/warm blend, S_field -> saturation, L -> brightness
          base = np.stack([
              0.5 + 0.5*(1.0-H_bits),             # R
              0.5 + 0.4*(0.5-S_field),            # G
              0.7 + 0.3*H_bits                    # B
          ], axis=0).astype(np.float32)
          self.col[:] = np.clip(base, 0.0, 1.0)

      def _project(self) -> Tuple[np.ndarray, np.ndarray]:
          # simple perspective projection
          cam_z = 3.2
          z = self.pos[:,2] + cam_z

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                              75/124

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

  # --------------- Broadcaster & Brain state ---------------
  class Broadcaster:
      def __init__(self): self._subs: List[asyncio.Queue]=[]
      def subscribe(self):
          q=asyncio.Queue(maxsize=200); self._subs.append(q); return q
      async def pub(self, msg:Dict[str,Any]):
          for q in list(self._subs):
              try: await q.put(msg)
              except asyncio.QueueFull: pass

  @dataclass
  class BrainState:
      tick:int=0; sigma:float=SIGMA0; anneal_step:int=0

  # --------------- Brain (autonomous) ----------------
  class Brain:
      def __init__(self):
          self.mem=Memory(DB_PATH); self.bus=Broadcaster()
          self.state=BrainState()
          self._rng=np.random.RandomState(101)
          self.avatar = AvatarSynth(OUT_AVATAR)
          self._seen_files: set[str] = set()

      # --- headless ingest (no user needed) ---
      def _poll_inbox(self) -> int:
          added = 0
          for p in sorted(INBOX.glob("*")):
              if not p.is_file(): continue
              if p.suffix.lower() not in {".txt",".md",".html",".htm"}: continue
              fp = str(p.resolve())
              if fp in self._seen_files: continue
              try:
                   txt = p.read_text(encoding="utf-8", errors="ignore")
                   lang = detect(txt[:500]) if txt.strip() else "en"
                   self.mem.teach(txt, lang)
                   self.mem.log(self.state.tick, "aut_ingest", {"file":p.name,"bytes":len(txt)})
                   self._seen_files.add(fp); added += 1
              except Exception as e:
                   self.mem.log(self.state.tick, "aut_ingest_err", {"file":p.name,"err":str(e)})
          return added

      # --- one anneal+energy step; returns energetics dict ---
      def _anneal(self) -> Dict[str,float]:
          E, ids = self.mem.embeddings(max_items=192)
          if E.size==0:
              return {"H_bits":0.0,"S_field":0.0,"L":0.0}
          N=E.shape[0]
          edges = ring_edges(N, k=max(4,min(12,N-1)))
          S=np.zeros(N)
          for i in range(N):
              var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self._rng)
              S[i]=stability(var)
          en=energetics(E,S,edges,self.state.sigma)
          self.mem.log(self.state.tick, "energetics", en)
          self.state.anneal_step += 1
          self.state.sigma = anneal_sigma(SIGMA0, GAMMA, self.state.anneal_step, SIGMA_MIN)
          return en

      async def loop(self):
          while True:
              try:
                   self.state.tick += 1
                   # autonomous ingest
                   if self.state.tick % AUTON_INGEST_EVERY == 0:
                       n = self._poll_inbox()
                       if n: await self.bus.pub({"type":"ingest","data":{"tick":self.state.tick,"files":n}})
                   # periodic anneal & avatar render
                   if self.state.tick % REFLECT_EVERY == 0:
                       en = self._anneal()
                       await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en, "sigma": self.state.sigma}})
                       # sonify (short tone) → optional; informs color/tempo implicitly
                       maps=default_maps(en["H_bits"], en["S_field"], latency=0.2, fitness=max(0.0, 1.0-en["H_bits"]))
                       sig=synth_signal(0.8, 22050, maps["a"], maps["m"], maps["rho"], maps["fc"])
                       write_wav_mono16(OUT_AUDIO/f"onbrain_{self.state.tick}.wav", 22050, sig)

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                  76/124

                      # drive avatar
                      self.avatar.step(en, dt=0.05)
                      img_path = self.avatar.render(self.state.tick)
                      await self.bus.pub({"type":"avatar","data":{"tick":self.state.tick, "frame":str(img_path)}})
              except Exception as e:
                  await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
              await asyncio.sleep(TICK_SEC)

      # interactive thinking stays available but not required
      async def think(self, text:str)->Dict[str,Any]:
          try:
              langs = [str(l) for l in detect_langs(text)]
          except Exception:
              langs = [detect(text)] if text.strip() else ["en"]
          lang = (langs[0].split(":")[0] if langs else "en").lower()
          retr = Retriever(self.mem)
          top_ids, top_sims = retr.topk(text, k=8)
          async def math_task():
              ok,res = MathSolver.solve_expr(text)
              return {"ok":ok, "res":res, "weight": 0.9 if ok else 0.0, "tag":"math"}
          async def logic_task():
              plan = LogicPlanner.plan(text)
              return {"ok":True, "res":"; ".join(plan), "weight":0.6, "tag":"plan"}
          async def compose_task():
              pieces=["Synthesis:"]
              if any(k in text.lower() for k in ["why","because","explain","how"]):
                  pieces.append("Explaining step-by-step, then summarizing.")
              else:
                  pieces.append("Combining relevant facts with a logical sequence.")
              return {"ok":True,"res":" ".join(pieces),"weight":0.5,"tag":"compose"}
          r_math, r_plan, r_comp = await asyncio.gather(math_task(), logic_task(), compose_task())
          best = max([r for r in [r_math,r_plan,r_comp] if r["ok"]], key=lambda r:r["weight"], default={"res":"(no
  solver)","tag":"none","weight":0.0})
          label = label_for(lang)
          ans = f"{label}: {best['res']}"
          self.mem.log(self.state.tick, "think", {"lang":lang,"query":text,"selected":best["tag"]})
          await self.bus.pub({"type":"think","data":{"tick":self.state.tick,"lang":lang,"selected":best["tag"]}})
          return {"ok":True,"lang":lang,"selected":best["tag"],"answer":ans,"context_ids":top_ids,"context_sims":top_sims}

  # --------------- API (optional UI) ----------------
  app = FastAPI(title="OnBrain Autonomous")
  app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
  brain = Brain()

  @app.on_event("startup")
  async def _boot():
      asyncio.create_task(brain.loop())

  @app.get("/", response_class=HTMLResponse)
  def home():
       return """<!doctype html><html><head><meta charset="utf-8"><title>OnBrain Auto</title>
  <style>body{font-family:system-ui,Inter,sans-serif;padding:20px;max-width:960px;margin:auto}
  input,textarea{width:100%;padding:10px;margin:8px 0;border:1px solid #ddd;border-radius:10px}
  button{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
  .card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin:10px 0}

  <h1>🧠
  img{max-width:100%;border-radius:12px;border:1px solid #eee}</style></head><body>
          OnBrain — Autonomous + 18k-Node Avatar</h1>
  <p>Running autonomously. UI is optional. Drop .txt/.md/.html into <code>onbrain_auto/inbox/</code> and watch it evolve.</p>
  <div class="card"><h3>Think</h3><form id="fThink"><textarea id="q" rows="3" placeholder="Ask anything (math, logic, design, multilingual)">
  </textarea><button>Think</button></form><pre id="ans"></pre></div>
  <div class="card"><h3>Status</h3><pre id="st"></pre><img id="im" /></div>
  <script>
  async function poll(){
     const s=await (await fetch('/status')).json();
     document.getElementById('st').textContent=JSON.stringify(s,null,2);
     if(s.latest_frame){document.getElementById('im').src='/frame?path='+encodeURIComponent(s.latest_frame)+'&t='+Date.now()}
     setTimeout(poll, 1200);
  }
  document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
    const text=document.getElementById('q').value; if(!text.trim())return;
    const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
    const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
  poll()
  </script></body></html>"""

  @app.post("/teach")
  def teach(payload: Dict[str, str] = Body(...)):
      text = payload.get("text","").strip()
      if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
      try: lang = detect(text)
      except Exception: lang = "en"
      fid = brain.mem.teach(text, lang)
      brain.mem.log(brain.state.tick, "teach", {"id":fid,"lang":lang})
      return {"ok":True, "id": fid, "lang": lang}

  @app.post("/think")
  async def think(payload: Dict[str,str] = Body(...)):
      text = payload.get("text","").strip()
      if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
      return await brain.think(text)

  @app.get("/status")
  def status():
      latest = sorted(OUT_AVATAR.glob("frame_*.png"))[-1] if any(OUT_AVATAR.glob("frame_*.png")) else None
      return {"ok":True, "tick":brain.state.tick, "sigma":brain.state.sigma, "latest_frame": (str(latest) if latest else None)}

  @app.get("/frame")
  def frame(path: str = Query(...)):
      p = Path(path)
      if not p.exists(): return JSONResponse({"ok":False,"error":"not found"}, status_code=404)
      from fastapi.responses import FileResponse
      return FileResponse(str(p))

  @app.websocket("/ws")
  async def ws(ws: WebSocket):
      await ws.accept()
      q = brain.bus.subscribe()
      try:
           await ws.send_text(json.dumps({"type":"hello","data":{"tick":brain.state.tick,"sigma":brain.state.sigma}}))
           while True:

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                              77/124

                 msg = await q.get()
                 await ws.send_text(json.dumps(msg))
         except WebSocketDisconnect:
             pass

  if __name__=="__main__":
      print(f"Open: http://{HOST}:{PORT}/ (optional)")
      uvicorn.run(app, host=HOST, port=PORT)



