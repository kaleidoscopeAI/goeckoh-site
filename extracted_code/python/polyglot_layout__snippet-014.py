  import os, sys, subprocess, venv, json, time, asyncio, math, sqlite3, traceback, wave, random
  from dataclasses import dataclass
  from typing import Dict, Any, List, Tuple, Iterable, Optional
  from pathlib import Path

  # ---------------- Bootstrapping: create .venv and re-exec ----------------
  ROOT = Path.cwd() / "onbrain"
  VENV = ROOT / ".venv"
  REQ = [
      "fastapi==0.115.5", "uvicorn==0.32.0", "numpy==1.26.4", "networkx==3.3",
      "beautifulsoup4==4.12.3", # for local HTML cleaning (no web fetch used)
      "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0",
      "scipy==1.11.4", "sympy==1.13.3", "langdetect==1.0.9",
  ]
  def ensure_venv_and_reexec():
      ROOT.mkdir(parents=True, exist_ok=True)
      if os.environ.get("ONB_BOOTED") == "1":
          return
      if not VENV.exists():
          print(">> Creating venv at", VENV)
          venv.create(VENV, with_pip=True)
      pip = VENV / ("Scripts/pip.exe" if os.name=="nt" else "bin/pip")
      py = VENV / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
      print(">> Upgrading pip and installing deps")
      subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
      subprocess.check_call([str(pip), "install"] + REQ)
      env = os.environ.copy(); env["ONB_BOOTED"] = "1"
      print(">> Relaunching inside venv")
      os.execvpe(str(py), [str(py), __file__], env)

  if __file__ == "<stdin>":
      script_path = ROOT / "onbrain.py"
      script_path.write_text(sys.stdin.read(), encoding="utf-8")
      __file__ = str(script_path)

  ensure_venv_and_reexec()

  # ---------------- Imports (post-venv) ----------------
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.responses import HTMLResponse, JSONResponse
  import uvicorn
  import numpy as np
  import networkx as nx
  from bs4 import BeautifulSoup
  from scipy.signal import stft, get_window
  from sympy import sympify, simplify, Eq, solve
  from langdetect import detect, detect_langs

  # ---------------- Config & paths ----------------
  PORT = int(os.getenv("ONB_PORT", "8770"))
  HOST = os.getenv("ONB_HOST", "0.0.0.0")
  TICK_SEC = float(os.getenv("ONB_TICK_SEC", "0.6"))
  REFLECT_EVERY = int(os.getenv("ONB_REFLECT_EVERY", "4"))
  SIGMA0 = float(os.getenv("ONB_SIGMA0", "0.9"))
  GAMMA = float(os.getenv("ONB_GAMMA", "0.93"))
  SIGMA_MIN = float(os.getenv("ONB_SIGMA_MIN", "0.10"))
  DB_PATH = os.getenv("ONB_DB_PATH", str(ROOT / "onbrain.db"))

  OUT_AUDIO = ROOT / "audio"; OUT_AUDIO.mkdir(parents=True, exist_ok=True)
  OUT_SHAPES = ROOT / "shapes"; OUT_SHAPES.mkdir(parents=True, exist_ok=True)

  # ---------------- Utilities ----------------
  def write_wav_mono16(path: Path, sr: int, samples: Iterable[float]) -> None:
      x = np.asarray(list(samples), dtype=np.float32)
      if x.size == 0: x = np.zeros(1, dtype=np.float32)
      x = np.clip(x, -1.0, 1.0)
      y = (x * 32767.0).astype(np.int16)
      with wave.open(str(path), 'wb') as wf:
          wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                                        67/124

          wf.writeframes(y.tobytes())

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

  def make_bands(F: int, H: int) -> List[Tuple[int,int]]:
      edges = np.linspace(0, F, H+1, dtype=int)
      return [(int(edges[i]), int(edges[i+1])) for i in range(H)]

  def head_features(X: np.ndarray, bands: List[Tuple[int,int]]) -> np.ndarray:
      F, T = X.shape; H = len(bands)
      E = np.zeros((H, T), dtype=np.float64)
      for h,(a,b) in enumerate(bands):
          if b<=a: b=min(a+1,F)
          E[h] = X[a:b].mean(axis=0)
      d1 = np.pad(np.diff(E,axis=1), ((0,0),(1,0)))
      d2 = np.pad(np.diff(d1,axis=1), ((0,0),(1,0)))
      return np.stack([E,d1,d2], axis=-1) # (H,T,3)

  # --------------- Multilingual hashing embeddings (subword-ish) ---------------
  # Language-agnostic: unicode n-grams (1–4), signed hashing into D dims.
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

  # --------------- Crystallization (annealing) & energetics ----------------
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
  def stability(var_sigma:float)->float: return 1.0/(1.0+var_sigma)
  def anneal_sigma(sigma0:float, gamma:float, step:int, sigma_min:float)->float:
      return max(sigma0*(gamma**step), sigma_min)
  def expected_cos_noise(ei,ej,sigma,M=4)->float:
      rng=np.random.RandomState(11); sims=[]
      for _ in range(M):
          ein=ei+sigma*rng.normal(0.0,1.0,size=ei.shape); ein/= (np.linalg.norm(ein)+1e-9)
          ejn=ej+sigma*rng.normal(0.0,1.0,size=ej.shape); ejn/= (np.linalg.norm(ejn)+1e-9)
          sims.append(float(np.dot(ein,ejn)))
      return float(np.mean(sims))
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
      for k,(i,j) in enumerate(edges): w[k]=expected_cos_noise(E[i],E[j],sigma)
      tau=1.0-w
      H_bits=float(np.mean(1.0-S) if E.shape[0] else 0.0)
      S_field=float(np.mean(tau)); L=float(np.sum(tau*tau))
      return {"H_bits":H_bits,"S_field":S_field,"L":L}

  # --------------- Memory store (SQLite) ----------------
  class Memory:
      def __init__(self, path:str, D:int=512):
          self.path=path; self.D=D; self._init()
      def _init(self):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("""CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, lang TEXT, text TEXT)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS embeds(id INTEGER PRIMARY KEY, vec BLOB)""")

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                   68/124

          cur.execute("""CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, type TEXT, json TEXT)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS energetics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits
  REAL, sfield REAL, L REAL)""")
          cur.execute("""CREATE TABLE IF NOT EXISTS captions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, caption TEXT, meta
  TEXT)""")
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
      def fact_text(self, by_ids:List[int])->Dict[int,str]:
          if not by_ids: return {}
          q=",".join(str(i) for i in by_ids)
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute(f"SELECT id,text FROM facts WHERE id IN ({q})")
          out={int(i):t for i,t in cur.fetchall()}
          con.close(); return out
      def log(self, tick:int, type_:str, data:dict):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("INSERT INTO traces(ts,tick,type,json) VALUES(?,?,?,?)",(time.time(), tick, type_, json.dumps(data)))
          con.commit(); con.close()
      def log_energy(self, tick:int, sigma:float, en:dict):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("INSERT INTO energetics(ts,tick,sigma,hbits,sfield,L) VALUES(?,?,?,?,?,?)",
                      (time.time(),tick,sigma,en["H_bits"],en["S_field"],en["L"]))
          con.commit(); con.close()
      def log_caption(self, tick:int, caption:str, meta:dict):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute("INSERT INTO captions(ts,tick,caption,meta) VALUES(?,?,?,?)",(time.time(),tick,caption,json.dumps(meta)))
          con.commit(); con.close()
      def recent(self, table:str, limit:int=50):
          con=sqlite3.connect(self.path); cur=con.cursor()
          cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
          rows=cur.fetchall(); con.close(); return rows

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

  # --------------- Domain solvers (offline) ----------------
  class MathSolver:
      @staticmethod
      def solve_expr(q:str)->Tuple[bool,str]:
          try:
              # quick detect equation vs expression
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
           # simple heuristic planner (no LLM): split goals, produce steps
           steps=[]
           s=prompt.strip()
           if any(k in s.lower() for k in ["prove","show","why","because"]):
               steps += ["Define terms precisely","List known axioms/facts","Transform goal into subgoals","Check counterexamples","Synthesize
  final argument"]
           if any(k in s.lower() for k in ["design","build","create","implement"]):
               steps += ["Clarify requirements","Sketch architecture","List modules & interfaces","Draft algorithm","Test on cases","Refine edge-
  cases","Document"]
           if not steps: steps = ["Clarify intent","Retrieve relevant facts","Draft candidate answer","Critique and improve","Produce final
  answer"]
           return steps

  class Retriever:
      def __init__(self, mem:Memory): self.mem=mem
      def topk(self, query:str, k:int=6)->Tuple[List[int], List[float]]:
          E, ids = self.mem.embeddings(max_items=512)
          if E.size==0: return [], []

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                             69/124

          qv = embed_text(query)
          sims = (E @ qv) / (np.linalg.norm(E,axis=1) * (np.linalg.norm(qv)+1e-9) + 1e-12)
          order = np.argsort(-sims)[:k]
          return [ids[i] for i in order], [float(sims[i]) for i in order]

  # --------------- Orchestrator (dual-hemisphere) ----------------
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

  class OnBrain:
      def __init__(self):
          self.mem=Memory(DB_PATH); self.bus=Broadcaster()
          self.state=BrainState()
          self.rng=np.random.RandomState(101)

      def _anneal_round(self):
          E, ids = self.mem.embeddings(max_items=192)
          if E.size==0: return None
          N=E.shape[0]
          edges = ring_edges(N, k=max(4,min(12,N-1)))
          S=np.zeros(N)
          for i in range(N):
              var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self.rng)
              S[i]=stability(var)
          en=energetics(E,S,edges,self.state.sigma)
          self.mem.log_energy(self.state.tick, self.state.sigma, en)
          maps=default_maps(en["H_bits"], en["S_field"], latency=0.2, fitness=max(0.0, 1.0-en["H_bits"]))
          sig=synth_signal(1.6, 22050, maps["a"], maps["m"], maps["rho"], maps["fc"])
          wav_path=OUT_AUDIO/f"onbrain_{self.state.tick}.wav"; write_wav_mono16(wav_path,22050,sig)
          X=stft_mag(np.array(sig,dtype=np.float64), sr=22050, win=1024, hop=256)
          V=head_features(X, make_bands(X.shape[0], H=4))
          # project "attention" to memory (no LLM)
          H,T,_=V.shape; D=E.shape[1]; d=24; rng=np.random.RandomState(1234)
          Wk=rng.normal(0, 1.0/math.sqrt(D), size=(D,d)); K=E@Wk; K/= (np.linalg.norm(K,axis=1,keepdims=True)+1e-9)
          captions=[]
          for h in range(H):
              Wq=rng.normal(0,1.0,size=(V.shape[2], d))
              Q=V[h]@Wq; Q/= (np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
              Satt=(Q@K.T)/(d*max(self.state.sigma, SIGMA_MIN))
              Satt -= Satt.max(axis=1, keepdims=True)
              P=np.exp(Satt); P/= (P.sum(axis=1,keepdims=True)+1e-12)
              svec=P.mean(axis=0); top=list(np.argsort(-svec)[:5])
              facts=self.mem.fact_text([ids[i] for i in top])
              cap="; ".join(facts.get(ids[i],"")[:80] for i in top if ids[i] in facts)
              if cap: captions.append(cap)
          if captions:
              self.mem.log_caption(self.state.tick, captions[-1], {"H_bits":en["H_bits"], "S_field":en["S_field"]})
          self.state.anneal_step += 1
          self.state.sigma = anneal_sigma(SIGMA0, GAMMA, self.state.anneal_step, SIGMA_MIN)
          return en, (captions[-1] if captions else "")

      async def loop(self):
          while True:
              try:
                   self.state.tick += 1
                   if self.state.tick % REFLECT_EVERY == 0:
                       out=self._anneal_round()
                       if out:
                           en, cap = out
                           await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en, "sigma": self.state.sigma}})
                           if cap: await self.bus.pub({"type":"caption","data":{"tick":self.state.tick, "text":cap}})
              except Exception as e:
                   await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
              await asyncio.sleep(TICK_SEC)

      # ---- Main thinking entry ----
      async def think(self, text:str)->Dict[str,Any]:
          # 1) Detect languages (may be multiple)
          try:
              langs = [str(l) for l in detect_langs(text)]
          except Exception:
              langs = [detect(text)] if text.strip() else ["en"]
          lang = (langs[0].split(":")[0] if langs else "en").lower()

          # 2) Parallel domain solvers (no APIs)
          retr = Retriever(self.mem)
          top_ids, top_sims = retr.topk(text, k=8)
          facts = self.mem.fact_text(top_ids)
          ctx = [facts.get(i,"") for i in top_ids]

          async def math_task():
              ok,res = MathSolver.solve_expr(text)
              return {"ok":ok, "res":res, "weight": 0.9 if ok else 0.0, "tag":"math"}

          async def logic_task():
              plan = LogicPlanner.plan(text)
              # tiny critique: prefer steps that use retrieved context if any
              if ctx: plan = plan[:1]+["Review retrieved facts for relevance"]+plan[1:]
              return {"ok":True, "res":"; ".join(plan), "weight":0.6, "tag":"plan"}

          async def compose_task():
              # Compose an answer without LLMs: rule-based template over context + simple reasoning
              pieces=[]
              if ctx:
                  pieces.append("Context:")
                  for i,(cid,sim) in enumerate(zip(top_ids, top_sims)):
                      t=facts.get(cid,"")
                      if t: pieces.append(f"- [{i+1}] {t[:160]} (sim={sim:.3f})")
              pieces.append("Synthesis:")
              # very small heuristics

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                      70/124

              if any(k in text.lower() for k in ["why","because","explain","how"]):
                  pieces.append("I’ll explain step-by-step, then summarize.")
              elif any(k in text.lower() for k in ["solve","=", "integrate","differentiate","derivative","limit"]):
                  pieces.append("See the math result and explanation above.")
              else:
                  pieces.append("Combining the most relevant known facts with a logical sequence to address your request.")
              return {"ok":True,"res":"\n".join(pieces),"weight":0.5,"tag":"compose"}

          r_math, r_plan, r_comp = await asyncio.gather(math_task(), logic_task(), compose_task())

          # 3) Score & merge
          candidates=[r for r in [r_math,r_plan,r_comp] if r["ok"]]
          best = max(candidates, key=lambda r:r["weight"]) if candidates else {"res":"(no solver matched)", "tag":"none", "weight":0.0}
          label = label_for(lang)
          answer = f"{label}: {best['res']}"

          # 4) Log trace
          self.mem.log(self.state.tick, "think", {"lang":lang,"query":text,"selected":best["tag"]})
          await self.bus.pub({"type":"think","data":{"tick":self.state.tick,"lang":lang,"selected":best["tag"]}})

          return {
              "ok": True,
              "lang": lang,
              "selected": best["tag"],
              "answer": answer,
              "context_ids": top_ids,
              "context_sims": top_sims,
          }

  # --------------- API ----------------
  app = FastAPI(title="OnBrain (Offline Polyglot Reasoner)")
  app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
  brain = OnBrain()

  @app.on_event("startup")
  async def _boot():
      asyncio.create_task(brain.loop())

  @app.get("/", response_class=HTMLResponse)
  def home():
       return """<!doctype html><html><head><meta charset="utf-8"><title>OnBrain</title>
  <style>body{font-family:system-ui,Segoe UI,Inter,sans-serif;padding:24px;max-width:920px;margin:auto;}
  input,textarea{width:100%;padding:10px;margin:8px 0;border:1px solid #ddd;border-radius:10px}
  button{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
  .card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin:10px 0}

  <h1>🧠
  small{color:#777}</style></head><body>
          OnBrain — Offline Polyglot Reasoner</h1>
  <p>No APIs. Runs locally. Teach it facts, then ask questions. Open a second tab to <code>/ws</code> to watch the live thought stream.</p>
  <div class="card"><h3>Teach</h3><form id="fTeach"><textarea id="teach" rows="3" placeholder="Teach a fact in any language"></textarea>
  <button>Teach</button></form><div id="teachOut"></div></div>
  <div class="card"><h3>Think</h3><form id="fThink"><textarea id="q" rows="3" placeholder="Ask me anything (math, logic, design, etc)"></textarea>
  <button>Think</button></form><pre id="ans"></pre></div>
  <div class="card"><h3>Recent</h3><a href="/recent?table=facts" target="_blank">facts</a> · <a href="/recent?table=energetics"
  target="_blank">energetics</a> · <a href="/recent?table=captions" target="_blank">captions</a></div>
  <script>
  document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
   const text=document.getElementById('teach').value; if(!text.trim())return;
   const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
   const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught fact id: '+j.id; document.getElementById('teach').value=''}
  document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
   const text=document.getElementById('q').value; if(!text.trim())return;
   const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
   const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
  </script></body></html>"""

  @app.post("/teach")
  def teach(payload: Dict[str, str] = Body(...)):
      text = payload.get("text","").strip()
      if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
      try:
           lang = detect(text)
      except Exception:
           lang = "en"
      fid = brain.mem.teach(text, lang)
      brain.mem.log(brain.state.tick, "teach", {"id":fid,"lang":lang})
      return {"ok":True, "id": fid, "lang": lang}

  @app.post("/think")
  async def think(payload: Dict[str,str] = Body(...)):
      text = payload.get("text","").strip()
      if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
      out = await brain.think(text)
      return out

  @app.get("/status")
  def status():
      return {"ok":True, "tick":brain.state.tick, "sigma":brain.state.sigma}

  @app.get("/recent")
  def recent(table: str = Query("facts"), limit: int = Query(50)):
      return {"ok": True, "rows": brain.mem.recent(table, limit)}

  @app.websocket("/ws")
  async def ws(ws: WebSocket):
      await ws.accept()
      q = brain.bus.subscribe()
      try:
           await ws.send_text(json.dumps({"type":"hello","data":{"tick":brain.state.tick,"sigma":brain.state.sigma}}))
           while True:
               msg = await q.get()
               await ws.send_text(json.dumps(msg))
      except WebSocketDisconnect:
           pass

  if __name__=="__main__":
      print(f"Open: http://{HOST}:{PORT}/")
      uvicorn.run(app, host=HOST, port=PORT)




Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                            71/124

