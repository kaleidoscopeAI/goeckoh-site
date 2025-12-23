#!/usr/bin/env python3
# OnBrain — Groundbreaking Edition
# Offline, On-Device Polyglot Reasoning + Self-Authored Persona
# Adds: formal mapping φ, tiny controller, coherence metric C, audio↔state feedback,
# node-face takeover with optimal assignment, and REST demo snapshot.
#
# Run:
#   python onbrain_groundbreaking.py
#   Open http://localhost:8775/
#
# Notes:
# - No external APIs. Uses numpy/scipy/fastapi/uvicorn/sympy/langdetect.
# - If a prior onbrain.db exists, it will be reused.

import os, sys, subprocess, venv, json, time, asyncio, math, sqlite3, traceback, wave, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterable, Optional
from pathlib import Path

# ---------------- Bootstrapping: create .venv and re-exec ----------------
ROOT = Path.cwd() / "onbrain_gb"
VENV = ROOT / ".venv"
REQ = [
    "fastapi==0.115.5", "uvicorn==0.32.0", "numpy==1.26.4", "networkx==3.3",
    "beautifulsoup4==4.12.3",
    "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0",
    "scipy==1.11.4", "sympy==1.13.3", "langdetect==1.0.9"
]
def ensure_venv_and_reexec():
    ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ONB_GB_BOOTED") == "1":
        return
    if not VENV.exists():
        print(">> Creating venv at", VENV)
        venv.create(VENV, with_pip=True)
    pip = VENV / ("Scripts/pip.exe" if os.name=="nt" else "bin/pip")
    py  = VENV / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
    print(">> Upgrading pip and installing deps")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
    subprocess.check_call([str(pip), "install"] + REQ)
    env = os.environ.copy(); env["ONB_GB_BOOTED"] = "1"
    print(">> Relaunching inside venv")
    os.execvpe(str(py), [str(py), __file__], env)

if __file__ == "<stdin>":
    script_path = ROOT / "onbrain_groundbreaking.py"
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
from scipy.optimize import linear_sum_assignment
from sympy import sympify, simplify, Eq, solve
from langdetect import detect, detect_langs

# ---------------- Config & paths ----------------
PORT = int(os.getenv("ONB_PORT", "8775"))
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
    return np.stack([E,d1,d2], axis=-1)  # (H,T,3)

# --------------- Multilingual hashing embeddings (subword-ish) ---------------
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
        cur.execute("""CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, type TEXT, json TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS energetics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits REAL, sfield REAL, L REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS captions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, caption TEXT, meta TEXT)""")
        # Persona + layout + metrics
        cur.execute("""CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY CHECK (id=1), ts REAL, schema TEXT, avatar_svg TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS identity_history(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, schema TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS node_layout(id INTEGER PRIMARY KEY, x REAL, y REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS coherence(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, C REAL, comp TEXT)""")
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
    def log_coherence(self, tick:int, C:float, comp:dict):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("INSERT INTO coherence(ts,tick,C,comp) VALUES(?,?,?,?)",(time.time(),tick,float(C),json.dumps(comp)))
        con.commit(); con.close()
    def recent(self, table:str, limit:int=50):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
        rows=cur.fetchall(); con.close(); return rows
    # persona IO
    def get_identity(self):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("SELECT schema FROM identity WHERE id=1"); r=cur.fetchone(); con.close()
        return json.loads(r[0]) if r and r[0] else None
    def save_identity(self, ident:dict, avatar_svg:str):
        con=sqlite3.connect(self.path); cur=con.cursor()
        js=json.dumps(ident, ensure_ascii=False)
        cur.execute("INSERT INTO identity(id,ts,schema,avatar_svg) VALUES(1,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET ts=excluded.ts,schema=excluded.schema,avatar_svg=excluded.avatar_svg",
                    (time.time(), js, avatar_svg))
        cur.execute("INSERT INTO identity_history(ts,schema) VALUES(?,?)", (time.time(), js))
        con.commit(); con.close()
    def get_avatar_svg(self)->str:
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("SELECT avatar_svg FROM identity WHERE id=1"); row=cur.fetchone()
        con.close(); return row[0] if row and row[0] else ""
    # layout
    def read_layout(self, ids:List[int])->Dict[int,Tuple[float,float]]:
        if not ids: return {}
        con=sqlite3.connect(self.path); cur=con.cursor()
        q=",".join("?"*len(ids))
        cur.execute(f"SELECT id,x,y FROM node_layout WHERE id IN ({q})", ids)
        rows=cur.fetchall(); con.close()
        return {int(i):(float(x),float(y)) for (i,x,y) in rows}
    def write_layout(self, coords:Dict[int,Tuple[float,float]]):
        if not coords: return
        con=sqlite3.connect(self.path); cur=con.cursor()
        for i,(x,y) in coords.items():
            cur.execute("INSERT INTO node_layout(id,x,y) VALUES(?,?,?) "
                        "ON CONFLICT(id) DO UPDATE SET x=excluded.x,y=excluded.y",(int(i), float(x), float(y)))
        con.commit(); con.close()

# --------------- Persona utils ---------------
def _u32(s:str)->int:
    import hashlib
    return int.from_bytes(hashlib.sha1(s.encode("utf-8","ignore")).digest()[:4],"little")
def _rng(seed:str): return np.random.RandomState(_u32(seed))

def _palette(seed:str):
    import colorsys
    r=_rng("pal/"+seed); h=r.uniform(0,360)
    def hsl(h,s,l):
        r,g,b=colorsys.hls_to_rgb(h/360,l/100,s/100); return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return {
        "bg": hsl((h+180)%360, 25, 12),
        "fg": hsl(h, 35, 92),
        "accents": [hsl((h+30)%360,60,58), hsl((h+140)%360,60,58), hsl((h+230)%360,60,58)]
    }

def _avatar_svg(ident:dict, size:int=256)->str:
    pal=ident["style"]["palette"]; sh=ident["style"]["shapes"]; W=H=size
    def rect(x,y,w,h,fill): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" ry="24" fill="{fill}"/>'
    def circ(cx,cy,r,fill): return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'
    def path(d,fill): return f'<path d="{d}" fill="{fill}"/>'
    faceR={"round":92,"oval":82,"square":88,"hex":86}[sh["face"]]
    eye={"dot":(6,7),"almond":(10,5),"wide":(12,4)}[sh["eyes"]]
    brow=sh["brows"]; mouth=sh["mouth"]; hair=sh["hair"]; acc=sh["accessory"]
    g=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',rect(0,0,W,H,pal["bg"]),
       circ(W*0.5,H*0.52,faceR,pal["fg"])]
    if hair=="wave": g.append(path(f"M0,60 Q{W*0.3},20 {W*0.5},60 T{W},60 L{W},0 L0,0 Z", pal["accents"][1]))
    if hair=="spike": g.append(path(f"M0,50 L{W*0.15},8 L{W*0.3},54 L{W*0.45},10 L{W*0.6},56 L{W*0.75},12 L{W},50 L{W},0 L0,0 Z", pal["accents"][2]))
    if hair=="curl": g.append(path(f"M0,40 C{W*0.25},10 {W*0.75},10 {W},40 L{W},0 L0,0 Z", pal["accents"][0]))
    g+= [circ(W*0.38,H*0.45,eye[0],pal["bg"]), circ(W*0.62,H*0.45,eye[0],pal["bg"]),
         circ(W*0.38,H*0.45,eye[1],pal["fg"]), circ(W*0.62,H*0.45,eye[1],pal["fg"])]
    by = H*0.38 if brow!="arched" else H*0.35
    def p(d): g.append(path(d,pal["accents"][2]))
    if brow=="soft":
        p(f"M{W*0.28},{by} Q{W*0.38},{by-12} {W*0.48},{by}"); p(f"M{W*0.52},{by} Q{W*0.62},{by-12} {W*0.72},{by}")
    elif brow=="straight":
        g += [f'<rect x="{W*0.28}" y="{by-6}" width="{W*0.20}" height="6" fill="{pal["accents"][2]}"/>',
              f'<rect x="{W*0.52}" y="{by-6}" width="{W*0.20}" height="6" fill="{pal["accents"][2]}"/>']
    else:
        p(f"M{W*0.28},{by} Q{W*0.38},{by-16} {W*0.48},{by}"); p(f"M{W*0.52},{by} Q{W*0.62},{by-16} {W*0.72},{by}")
    my=H*0.67; amp={"smile":12,"neutral":5,"serif":2}[mouth]
    if mouth=="smile": g.append(path(f"M{W*0.40},{my} Q{W*0.50},{my+amp} {W*0.60},{my}", pal["bg"]))
    elif mouth=="serif": g.append(f'<rect x="{W*0.45}" y="{my-2}" width="{W*0.10}" height="3" fill="{pal["bg"]}"/>')
    else: g.append(f'<rect x="{W*0.435}" y="{my-2}" width="{W*0.13}" height="4" fill="{pal["bg"]}"/>')
    if acc=="visor": g.append(f'<rect x="{W*0.30}" y="{H*0.40}" width="{W*0.40}" height="16" fill="{pal["accents"][0]}"/>')
    if acc=="antenna": g += [f'<path d="M{W*0.5},{H*0.12} L{W*0.5},{H*0.36}" fill="{pal["accents"][0]}"/>', circ(W*0.5,H*0.12,6,pal["accents"][0])]
    if acc=="mono": g += [circ(W*0.62,H*0.45,18,pal["accents"][0]), circ(W*0.62,H*0.45,12,pal["bg"])]
    if acc=="ear": g += [circ(W*0.18,H*0.52,14,pal["accents"][1]), circ(W*0.82,H*0.52,14,pal["accents"][1])]
    g.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="14" fill="{pal["fg"]}" opacity="0.75">{ident["core"]["signature_emoji"]} {ident["core"]["name"]}</text>')
    g.append("</svg>"); return "".join(g)

# ---------- FORMAL MAPPING φ with Lipschitz clamp ----------
class FormalMap:
    """
    φ: (H_bits, S_field, sigma) -> voice & face parameters
    Ensures bounded change: ||Δφ|| <= L * ||Δx|| with L <= L_MAX.
    """
    L_MAX = 2.0  # global Lipschitz cap

    @staticmethod
    def _clamp_delta(prev:dict, nxt:dict, alpha:float=0.3)->dict:
        # Blend towards nxt to respect Lipschitz-like bound
        out={}
        for k,v in nxt.items():
            pv = prev.get(k, v)
            out[k] = (1-alpha)*pv + alpha*v
        return out

    @staticmethod
    def voice(en:dict, prev:dict)->dict:
        H=float(en.get("H_bits",0.5)); S=float(en.get("S_field",0.5)); sig=float(en.get("sigma",0.5))
        base = {
            "f0": 110 + 70*(1.0-H) + 20*(0.5-S),
            "speaking_rate": 0.8 + 0.6*(1.0-S),  # more field tension => faster
            "breathiness": 0.08 + 0.40*H,        # uncertain => breathier
        }
        if prev: base = FormalMap._clamp_delta(prev, base, alpha=0.25)
        return base

    @staticmethod
    def face(en:dict, prev:dict)->dict:
        H=float(en.get("H_bits",0.5))
        mouth = "smile" if H<0.35 else ("neutral" if H<0.6 else "serif")
        shape = {
            "mouth": mouth,
            "brow_height": 0.35 if H<0.35 else (0.38 if H<0.6 else 0.40),
            "pupil_scale": 1.15 if H>0.6 else (1.0 if H>0.35 else 0.9),
        }
        if prev: shape = FormalMap._clamp_delta(prev, shape, alpha=0.25)
        return shape

# ---------- Tiny Controller (numpy MLP, online) ----------
class TinyController:
    def __init__(self, seed:str="ctrl"):
        r=_rng("ctrl/"+seed)
        self.W1 = r.normal(0, 0.3, (4, 6))   # inputs: [1,H,S,sigma]
        self.b1 = np.zeros(6)
        self.W2 = r.normal(0, 0.2, (6, 3))   # outputs: Δ[f0, rate, breath]
        self.b2 = np.zeros(3)
        self.lr = 0.01

    def _forward(self, x:np.ndarray)->np.ndarray:
        h = np.tanh(x@self.W1 + self.b1)
        y = np.tanh(h@self.W2 + self.b2)  # [-1,1]
        return y

    def infer(self, H,S,sigma)->Dict[str,float]:
        x = np.array([1.0, H, S, sigma])
        y = self._forward(x)
        return {"df0": 12.0*y[0], "drate": 0.25*y[1], "dbreath": 0.15*y[2]}

    def train_step(self, H,S,sigma, target:Dict[str,float]):
        # Minimize (controller(y) - target) L2
        x = np.array([1.0, H, S, sigma])
        # forward
        h = np.tanh(x@self.W1 + self.b1)
        y = np.tanh(h@self.W2 + self.b2)
        t = np.array([target["df0"]/12.0, target["drate"]/0.25, target["dbreath"]/0.15])
        # grads
        dy = 2*(y - t) * (1 - y*y)  # tanh'
        dW2 = np.outer(h, dy)
        db2 = dy
        dh = (self.W2 @ dy) * (1 - h*h)
        dW1 = np.outer(x, dh)
        db1 = dh
        # update
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

# ---------- Procedural voice riding the anneal signal ----------
def synth_voice_from_memory(text:str, f0:float, rate:float, breath:float, sr:int, anneal_sig:np.ndarray)->np.ndarray:
    if anneal_sig.size==0:
        anneal_sig = np.zeros(int(sr*0.3))
    carrier = anneal_sig / (np.max(np.abs(anneal_sig))+1e-9)
    out=[]; hop=int(sr*max(0.05, 0.12 - 0.04*min(1.5,max(0.5,rate))))
    pos=0
    formants = {"a":[800,1200,2600],"e":[400,2000,2600],"i":[300,2300,3000],"o":[500,900,2400],"u":[350,600,2400]}
    for ch in text.lower():
        seg = carrier[pos:pos+hop] if pos+hop<=len(carrier) else carrier[pos:]
        if seg.size<hop: seg = np.pad(seg,(0,hop-seg.size))
        pos = (pos+hop) % max(1, len(carrier))
        y = seg.copy()
        if ch in "aeiou":
            F = formants.get(ch,[500,1500,2500])
            t=np.arange(hop)/sr
            y += 0.12*np.sin(2*np.pi*F[0]*t) + 0.08*np.sin(2*np.pi*F[1]*t) + 0.06*np.sin(2*np.pi*F[2]*t)
        elif ch in " .,!?:;":
            y *= 0.2
        else:
            y += 0.05*np.random.randn(hop)
        y = (1.0-breath)*y + breath*(0.08*np.random.randn(hop))
        out.append(0.9*np.tanh(y)*np.hanning(hop))
    y = np.concatenate(out) if out else np.zeros(int(sr*0.3))
    # gentle pitch push with f0 envelope
    t=np.arange(y.size)/sr
    y += 0.05*np.sin(2*np.pi*f0*t)
    y /= (np.max(np.abs(y))+1e-9)
    return y

# ---------- Coherence metric C ----------
def spectral_stats(y:np.ndarray, sr:int=22050)->Dict[str,float]:
    if y.size==0: return {"flatness":0.0,"centroid":0.0,"zcr":0.0}
    # spectral flatness via geometric/arith mean
    X = stft_mag(y, sr, 1024, 256)
    gmean = np.exp(np.mean(np.log(np.maximum(X,1e-12))))
    amean = np.mean(X)
    flat = float(gmean/ (amean+1e-9))
    # centroid
    freqs = np.linspace(0, sr/2, X.shape[0])
    centroid = float(np.sum(freqs[:,None]*X)/ (np.sum(X)+1e-9)) / (sr/2)
    # zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(y+1e-12)))))
    return {"flatness":flat, "centroid":centroid, "zcr":zcr}

def mouth_curvature(layout:List[Tuple[float,float]])->float:
    if len(layout)<3: return 0.0
    xs = np.array([p[0] for p in layout]); ys = np.array([p[1] for p in layout])
    # fit y = ax^2 + bx + c ; curvature ~ a
    A = np.vstack([xs*xs, xs, np.ones_like(xs)]).T
    try:
        sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
        a=float(sol[0]); return a
    except Exception:
        return 0.0

def coherence_score(en:dict, voice:dict, y:np.ndarray, mouth_pts:List[Tuple[float,float]])->Tuple[float,dict]:
    H=float(en.get("H_bits",0.5)); S=float(en.get("S_field",0.5))
    st = spectral_stats(y, 22050)
    # targets: as H↓ → breath↓, centroid↓ ; as S↑ → rate↑ (proxy by zcr↑)
    t_breath = 0.12 + 0.25*H
    t_centroid = 0.55*(0.5+0.5*(1.0-H))
    t_zcr = 0.10 + 0.20*S
    e_b = abs(voice["breathiness"] - t_breath)
    e_c = abs(st["centroid"] - t_centroid)
    e_z = abs(st["zcr"] - t_zcr)
    # mouth curvature target: smile (concave up) as H↓ → a more positive
    a = mouth_curvature(mouth_pts)
    t_a = 0.30*(0.5 - H)   # positive if H small
    e_a = abs(a - t_a)
    # combine into C in [0,1]: higher better
    E = 0.4*e_b + 0.3*e_c + 0.2*e_z + 0.1*e_a
    C = float(np.exp(-3.0*E))
    comp = {"e_breath":e_b,"e_centroid":e_c,"e_zcr":e_z,"e_mouth":e_a,"flat":st["flatness"],"centroid":st["centroid"],"zcr":st["zcr"],"a":a}
    return C, comp

# --------------- Polyglot helpers ----------------
LANG_LABEL = {
    "en":"Answer", "es":"Respuesta", "fr":"Réponse", "de":"Antwort", "it":"Risposta",
    "pt":"Resposta", "nl":"Antwoord", "ru":"Ответ", "zh-cn":"答复", "zh-tw":"答覆",
    "ja":"回答", "ko":"답변", "ar":"الإجابة"
}
def label_for(lang:str)->str: return LANG_LABEL.get(lang.lower(), "Answer")

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
            steps += ["Define terms precisely","List known axioms/facts","Transform goal into subgoals","Check counterexamples","Synthesize final argument"]
        if any(k in s for k in ["design","build","create","implement"]):
            steps += ["Clarify requirements","Sketch architecture","List modules & interfaces","Draft algorithm","Test on cases","Refine edge-cases","Document"]
        if not steps: steps = ["Clarify intent","Retrieve relevant facts","Draft candidate answer","Critique and improve","Produce final answer"]
        return steps

class Retriever:
    def __init__(self, mem:Memory): self.mem=mem
    def topk(self, query:str, k:int=6)->Tuple[List[int], List[float]]:
        E, ids = self.mem.embeddings(max_items=512)
        if E.size==0: return [], []
        qv = embed_text(query)
        sims = (E @ qv) / (np.linalg.norm(E,axis=1) * (np.linalg.norm(qv)+1e-9) + 1e-12)
        order = np.argsort(-sims)[:k]
        return [ids[i] for i in order], [float(sims[i]) for i in order]

# --------------- Orchestrator ---------------
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
    voice_prev:dict=None; face_prev:dict=None

class OnBrain:
    def __init__(self):
        self.mem=Memory(DB_PATH); self.bus=Broadcaster()
        self.state=BrainState()
        self.rng=np.random.RandomState(101)
        self.ctrl = TinyController()
        self._ensure_identity_boot()

    # ---------- Identity boot ----------
    def _ensure_identity_boot(self):
        if not self.mem.get_identity():
            en={"H_bits":0.5,"S_field":0.5,"sigma":self.state.sigma,"L":0.0}
            ident=self._identity_from_en(en, "boot")
            self.mem.save_identity(ident, _avatar_svg(ident))

    def _identity_from_en(self, en:dict, affix:str)->dict:
        H=float(en.get("H_bits",0.5)); S=float(en.get("S_field",0.5)); sig=float(en.get("sigma",0.5))
        base=f"H={H:.3f}|S={S:.3f}|sig={sig:.3f}|{affix}"
        r=_rng(base)
        ident={
          "version":"1",
          "seed":base,
          "core":{"name": f"OnBrain {abs(_u32(base))%10000:04d}",
                  "motto": r.choice(["crystallize the noise","curiosity is compression","tension reveals shape","listen ↔ reflect"]),
                  "signature_emoji": r.choice(["🧠","🔷","✨","🔭","🌐"])},
          "style":{"palette": _palette(base),
                   "shapes":{"face": r.choice(["round","oval","square","hex"]),
                             "eyes": r.choice(["dot","almond","wide"]),
                             "brows": r.choice(["soft","straight","arched"]),
                             "mouth": "smile" if H<0.35 else ("neutral" if H<0.6 else "serif"),
                             "hair": r.choice(["none","wave","spike","curl"]),
                             "accessory": r.choice(["none","mono","ear","visor","antenna"])
                            }},
          "voice":{"f0": 110 + 60*(1.0-H),
                   "speaking_rate": 0.9 + 0.5*(1.0-S),
                   "breathiness": 0.10 + 0.35*H},
          "locks":{"frozen": False}
        }
        return ident

    # ---------- Layout ----------
    def _pca2(self, E:np.ndarray)->np.ndarray:
        if E.shape[0]==0: return np.zeros((0,2))
        X=E - E.mean(axis=0, keepdims=True)
        U,S,Vt=np.linalg.svd(X, full_matrices=False)
        P=X @ Vt[:2].T
        P/= (np.max(np.linalg.norm(P,axis=1))+1e-9)
        return 0.5 + 0.45*P

    def _ensure_layout(self):
        E, ids = self.mem.embeddings(max_items=256)
        if len(ids)==0: return
        coords = self.mem.read_layout(ids)
        if len(coords) < len(ids):
            P=self._pca2(E)
            mapping={}
            for (i,p) in zip(ids,P):
                if i not in coords:
                    mapping[i]=(float(p[0]), float(p[1]))
            self.mem.write_layout(mapping)

    def _face_targets(self, mood:str)->Dict[str,List[Tuple[float,float]]]:
        left_eye=[(0.38+0.05*np.cos(t), 0.45+0.04*np.sin(t)) for t in np.linspace(0,2*np.pi,24,endpoint=False)]
        right_eye=[(0.62+0.05*np.cos(t), 0.45+0.04*np.sin(t)) for t in np.linspace(0,2*np.pi,24,endpoint=False)]
        brow_y = 0.35 if mood=="wow" else (0.38 if mood=="calm" else 0.37)
        browL=[(x, brow_y - 0.02*np.cos((x-0.28)*12)) for x in np.linspace(0.28,0.48,16)]
        browR=[(x, brow_y - 0.02*np.cos((x-0.52)*12)) for x in np.linspace(0.52,0.72,16)]
        amp = 0.10 if mood=="smile" else (0.02 if mood=="calm" else -0.05)
        mouth=[(x, 0.67 + amp*np.sin((x-0.50)*np.pi)) for x in np.linspace(0.40,0.60,24)]
        return {"eyeL":left_eye, "eyeR":right_eye, "browL":browL, "browR":browR, "mouth":mouth}

    def _assign_nodes(self, ids:List[int], coords:Dict[int,Tuple[float,float]], targets:List[Tuple[float,float]]):
        # Optimal assignment of a slice of ids to target points
        if not ids or not targets: return {}
        pts = np.array([coords[i] for i in ids])
        tar = np.array(targets)
        C = np.linalg.norm(pts[:,None,:]-tar[None,:,:], axis=2)
        r,c = linear_sum_assignment(C)
        update={}
        for ri,ci in zip(r,c):
            i = ids[ri]; tx,ty = tar[ci]; x0,y0 = coords[i]
            x = x0 + 0.18*(tx - x0); y = y0 + 0.18*(ty - y0)
            update[i]=(float(x),float(y))
        return update

    def _takeover_step(self, mood:str="auto"):
        E, ids = self.mem.embeddings(max_items=256)
        if len(ids)==0: return
        self._ensure_layout()
        coords=self.mem.read_layout(ids)
        # pick mood by H_bits
        recent=self.mem.recent("energetics",1); H=0.5
        if recent:
            # cols: id, ts, tick, sigma, hbits, sfield, L
            _,_,_,_,hbits,_,_ = recent[0]
            H=float(hbits)
        mood = "smile" if mood=="auto" and H<0.35 else ("calm" if H<0.6 else "wow")
        T=self._face_targets(mood)
        order = list(sorted(ids))
        groups = ["eyeL","eyeR","browL","browR","mouth"]
        sizes = [len(T[g]) for g in groups]
        cuts = np.cumsum([0]+sizes)
        update={}
        for gi,g in enumerate(groups):
            seg = order[cuts[gi]:cuts[gi+1]]
            upd = self._assign_nodes(seg, coords, T[g])
            update.update(upd)
        self.mem.write_layout(update)

    # ---------- Anneal round + persona coupling ----------
    def _anneal_round(self):
        E, ids = self.mem.embeddings(max_items=192)
        if E.size==0: return None
        N=E.shape[0]
        edges = ring_edges(N, k=max(4,min(12,N-1)))
        S=np.zeros(N)
        for i in range(N):
            var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self.rng)
            S[i]=stability(var)
        en=energetics(E,S,edges,self.state.sigma); en["sigma"]=self.state.sigma
        self.mem.log_energy(self.state.tick, self.state.sigma, en)
        # Synth anneal signal
        maps=self._audio_maps(en)
        sig=self._synth_signal(1.6, 22050, maps)
        wav_path=OUT_AUDIO/f"onbrain_{self.state.tick}.wav"; write_wav_mono16(wav_path,22050,sig)
        # Persona mapping via φ + controller
        base_voice = FormalMap.voice(en, self.state.voice_prev or {})
        base_face  = FormalMap.face(en, self.state.face_prev or {})
        adj = self.ctrl.infer(en["H_bits"], en["S_field"], self.state.sigma)
        voice = {
            "f0": base_voice["f0"] + adj["df0"],
            "speaking_rate": max(0.6, base_voice["speaking_rate"] + adj["drate"]),
            "breathiness": np.clip(base_voice["breathiness"] + adj["dbreath"], 0.02, 0.6),
        }
        # Generate a caption proxy
        cap = self._caption_stub(ids)
        # Speak caption over anneal
        y = synth_voice_from_memory(cap, voice["f0"], voice["speaking_rate"], voice["breathiness"], 22050, np.array(sig,dtype=np.float64))
        out = OUT_AUDIO/f"persona_say_{self.state.tick}.wav"; write_wav_mono16(out, 22050, y.tolist())
        # Layout takeover + compute mouth curvature points
        self._takeover_step(mood="auto")
        mouth_pts = self._mouth_points()
        # Coherence metric
        C, comp = coherence_score(en, voice, y, mouth_pts)
        self.mem.log_coherence(self.state.tick, C, comp)
        # Audio->state feedback: gently nudge sigma based on spectral flatness gap
        st = spectral_stats(y, 22050)
        target_flat = 0.25 + 0.35*(1.0-en["H_bits"])
        delta = 0.08*(target_flat - st["flatness"])
        self.state.sigma = float(np.clip(self.state.sigma + delta, SIGMA_MIN, 1.0))
        # Controller small learning step: push towards reducing coherence errors
        tgt = {"df0": -5.0*(comp["e_centroid"]), "drate": -3.0*(comp["e_zcr"]), "dbreath": -4.0*(comp["e_breath"])}
        self.ctrl.train_step(en["H_bits"], en["S_field"], en["sigma"], tgt)
        # Save identity snapshot with updated face mouth state
        self._refresh_identity_from(en, base_face)
        self.state.voice_prev = base_voice; self.state.face_prev = base_face
        return en, cap, voice, C

    def _mouth_points(self)->List[Tuple[float,float]]:
        # reconstruct mouth sample points from node assignments (approx by selecting mid-range x)
        E, ids = self.mem.embeddings(max_items=256)
        coords = self.mem.read_layout(ids)
        # select nodes whose y is near 0.67 +- 0.06 and x in [0.40, 0.60]
        pts = [(x,y) for (i,(x,y)) in coords.items() if 0.40<=x<=0.60 and 0.61<=y<=0.73]
        pts = sorted(pts, key=lambda p:p[0])
        return pts[:24]

    def _refresh_identity_from(self, en:dict, face_base:dict):
        ident = self.mem.get_identity()
        if not ident or ident.get("locks",{}).get("frozen",False): return
        # mouth shape & palette drift from φ
        ident["style"]["shapes"]["mouth"] = "smile" if en["H_bits"]<0.35 else ("neutral" if en["H_bits"]<0.6 else "serif")
        ident["style"]["palette"] = _palette(f'pal|H={en["H_bits"]:.3f}|S={en["S_field"]:.3f}|sig={self.state.sigma:.3f}')
        self.mem.save_identity(ident, _avatar_svg(ident))

    def _audio_maps(self, en:dict):
        H=max(0.0,min(1.0,en["H_bits"])); S=max(0.0,min(1.0,en["S_field"]))
        def a_fn(t): return 0.25 + 0.5*(1.0-H)*(1.0-S)
        def m_fn(t): return 2.0 + 10.0*S
        def rho_fn(t): return 0.2 + 3.0*(1.0-H)
        def fc_fn(t): return 140.0 + 360.0*(1.0-H)
        return {"a":a_fn,"m":m_fn,"rho":rho_fn,"fc":fc_fn}

    def _synth_signal(self, seconds: float, sr: int, maps:dict)->List[float]:
        n=int(seconds*sr); out=[]
        for i in range(n):
            t=i/sr; a=maps["a"](t); m=maps["m"](t); rho=maps["rho"](t); fc=max(5.0, maps["fc"](t))
            y = a*(1.0+0.4*math.sin(2*math.pi*m*t))*math.sin(2*math.pi*fc*t + 0.6*math.sin(2*math.pi*rho*t))
            out.append(y)
        return out

    def _caption_stub(self, ids:List[int])->str:
        facts=self.mem.fact_text(ids[:4])
        if facts:
            return "; ".join([v.split("\n")[0][:60] for v in facts.values()])
        return "I am crystallizing what I learn into stable forms and speaking them."

    async def loop(self):
        while True:
            try:
                self.state.tick += 1
                if self.state.tick % REFLECT_EVERY == 0:
                    out=self._anneal_round()
                    if out:
                        en, cap, voice, C = out
                        await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en, "sigma": self.state.sigma}})
                        await self.bus.pub({"type":"coherence","data":{"tick":self.state.tick, "C":C}})
                        self.mem.log_caption(self.state.tick, cap, {"voice":voice,"C":C})
            except Exception as e:
                await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
            await asyncio.sleep(TICK_SEC)

    async def think(self, text:str)->Dict[str,Any]:
        try:
            langs = [str(l) for l in detect_langs(text)]
        except Exception:
            langs = [detect(text)] if text.strip() else ["en"]
        lang = (langs[0].split(":")[0] if langs else "en").lower()
        retr = Retriever(self.mem)
        top_ids, top_sims = retr.topk(text, k=8)
        facts = self.mem.fact_text(top_ids)

        async def math_task():
            ok,res = MathSolver.solve_expr(text)
            return {"ok":ok, "res":res, "weight": 0.9 if ok else 0.0, "tag":"math"}

        async def logic_task():
            plan = LogicPlanner.plan(text)
            if facts: plan = plan[:1]+["Review retrieved facts for relevance"]+plan[1:]
            return {"ok":True, "res":"; ".join(plan), "weight":0.6, "tag":"plan"}

        async def compose_task():
            pieces=[]
            if facts:
                pieces.append("Context:")
                for i,(cid,sim) in enumerate(zip(top_ids, top_sims)):
                    t=facts.get(cid,""); 
                    if t: pieces.append(f"- [{i+1}] {t[:160]} (sim={sim:.3f})")
            pieces.append("Synthesis:")
            if any(k in text.lower() for k in ["why","because","explain","how"]):
                pieces.append("I’ll explain step-by-step, then summarize.")
            elif any(k in text.lower() for k in ["solve","=", "integrate","differentiate","derivative","limit"]):
                pieces.append("See the math result and explanation above.")
            else:
                pieces.append("Combining the most relevant known facts with a logical sequence to address your request.")
            return {"ok":True,"res":"\n".join(pieces),"weight":0.5,"tag":"compose"}

        r_math, r_plan, r_comp = await asyncio.gather(math_task(), logic_task(), compose_task())
        candidates=[r for r in [r_math,r_plan,r_comp] if r["ok"]]
        best = max(candidates, key=lambda r:r["weight"]) if candidates else {"res":"(no solver matched)", "tag":"none", "weight":0.0}
        label = label_for(lang)
        answer = f"{label}: {best['res']}"
        self.mem.log(self.state.tick, "think", {"lang":lang,"query":text,"selected":best["tag"]})
        await self.bus.pub({"type":"think","data":{"tick":self.state.tick,"lang":lang,"selected":best["tag"]}})
        return {"ok": True,"lang": lang,"selected": best["tag"],"answer": answer,"context_ids": top_ids,"context_sims": top_sims}

# --------------- API ----------------
app = FastAPI(title="OnBrain (Groundbreaking Edition)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
brain = OnBrain()

@app.on_event("startup")
async def _boot():
    asyncio.create_task(brain.loop())

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta charset="utf-8"><title>OnBrain GB</title>
<style>body{font-family:system-ui,Segoe UI,Inter,sans-serif;padding:24px;max-width:980px;margin:auto;}
input,textarea{width:100%;padding:10px;margin:8px 0;border:1px solid #ddd;border-radius:10px}
button{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
.card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin:10px 0}
small{color:#777}</style></head><body>
<h1>🧠 OnBrain — Groundbreaking Edition</h1>
<p>Identity and voice are physicalizations of internal information dynamics. No APIs, all local.</p>
<div class="card"><h3>Teach</h3><form id="fTeach"><textarea id="teach" rows="3" placeholder="Teach a fact in any language"></textarea><button>Teach</button></form><div id="teachOut"></div></div>
<div class="card"><h3>Think</h3><form id="fThink"><textarea id="q" rows="3" placeholder="Ask me anything (math, logic, design, etc)"></textarea><button>Think</button></form><pre id="ans"></pre></div>
<div class="card"><h3>Persona</h3>
  <div style="display:flex;gap:12px;align-items:center">
    <img id="avatar" src="/persona/avatar.svg" width="128" height="128" style="border-radius:16px;border:1px solid #eee"/>
    <div style="flex:1">
      <button id="speak">Speak last caption</button>
      <a href="/persona/layout" target="_blank">layout</a> · <a href="/metrics/coherence" target="_blank">coherence</a>
    </div>
  </div>
</div>
<div class="card"><h3>Recent</h3><a href="/recent?table=facts" target="_blank">facts</a> · <a href="/recent?table=energetics" target="_blank">energetics</a> · <a href="/recent?table=captions" target="_blank">captions</a> · <a href="/metrics/coherence" target="_blank">coherence</a></div>
<script>
document.getElementById('fTeach').onsubmit=async(e)=>{e.preventDefault()
 const text=document.getElementById('teach').value; if(!text.trim())return;
 const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
 const j=await r.json(); document.getElementById('teachOut').innerHTML='Taught fact id: '+j.id; document.getElementById('teach').value=''}
document.getElementById('fThink').onsubmit=async(e)=>{e.preventDefault()
 const text=document.getElementById('q').value; if(!text.trim())return;
 const r=await fetch('/think',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
 const j=await r.json(); document.getElementById('ans').textContent=JSON.stringify(j,null,2)}
document.getElementById('speak').onclick=async()=>{
 const caps = await (await fetch('/recent?table=captions&limit=1')).json();
 const text = caps.rows && caps.rows[0] ? (caps.rows[0][3]||"I am crystallizing what I learn.") : "I am crystallizing what I learn.";
 await fetch('/persona/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
 document.getElementById('avatar').src='/persona/avatar.svg?'+Date.now();
}
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
    out = await brain.think(text); return out

@app.get("/status")
def status():
    return {"ok":True, "tick":brain.state.tick, "sigma":brain.state.sigma}

@app.get("/recent")
def recent(table: str = Query("facts"), limit: int = Query(50)):
    return {"ok": True, "rows": brain.mem.recent(table, limit)}

# --- Persona + Layout + Coherence ---
@app.get("/persona/avatar.svg", response_class=HTMLResponse)
def persona_avatar():
    svg = brain.mem.get_avatar_svg()
    if not svg:
        en={"H_bits":0.5,"S_field":0.5,"sigma":brain.state.sigma,"L":0.0}
        ident=brain._identity_from_en(en,"api"); brain.mem.save_identity(ident, _avatar_svg(ident)); svg=brain.mem.get_avatar_svg()
    return svg

@app.get("/persona/layout")
def persona_layout():
    E, ids = brain.mem.embeddings(max_items=256)
    brain._ensure_layout()
    coords = brain.mem.read_layout(ids)
    rows=[{"id":int(i),"x":float(x),"y":float(y)} for i,(x,y) in coords.items()]
    return {"ok":True,"nodes":rows}

@app.post("/persona/say")
def persona_say(payload: Dict[str,Any] = Body(...)):
    text = str(payload.get("text","")).strip()
    if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
    # use last anneal wav
    sr=22050
    try:
        paths = sorted(list((OUT_AUDIO).glob("onbrain_*.wav")), key=lambda p:p.stat().st_mtime)
        last = paths[-1]
        with wave.open(str(last),'rb') as wf:
            n=wf.getnframes(); data=wf.readframes(n)
            y=np.frombuffer(data, dtype=np.int16).astype(np.float32)/32767.0
    except Exception:
        y=np.zeros(int(sr*0.8),dtype=np.float32)
    # derive current voice via φ + controller
    recent=brain.mem.recent("energetics",1)
    en={"H_bits":0.5,"S_field":0.5,"sigma":brain.state.sigma}
    if recent:
        _,_,_,_,hbits,sfield,_ = recent[0]; en["H_bits"]=float(hbits); en["S_field"]=float(sfield)
    base = FormalMap.voice(en, brain.state.voice_prev or {}); adj = brain.ctrl.infer(en["H_bits"], en["S_field"], brain.state.sigma)
    f0 = base["f0"] + adj["df0"]; rate = max(0.6, base["speaking_rate"] + adj["drate"]); breath = np.clip(base["breathiness"] + adj["dbreath"], 0.02, 0.6)
    yv = synth_voice_from_memory(text, f0, rate, breath, sr, y)
    out = OUT_AUDIO/"persona_say.wav"; write_wav_mono16(out, sr, yv.tolist())
    # feedback sigma small-step
    st = spectral_stats(yv, sr); target_flat = 0.25 + 0.35*(1.0-en["H_bits"]); delta = 0.06*(target_flat - st["flatness"])
    brain.state.sigma = float(np.clip(brain.state.sigma + delta, SIGMA_MIN, 1.0))
    return {"ok":True,"path":str(out),"f0":f0,"rate":rate,"breath":float(breath),"sigma":brain.state.sigma}

@app.get("/metrics/coherence")
def metrics_coherence(limit:int=50):
    return {"ok":True,"rows": brain.mem.recent("coherence", limit)}

# --- WebSocket ---
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
