#!/usr/bin/env python3
# OnBrain — Online Explorer + 3D Avatar Edition (single file)
# Offline core + optional safe web exploration + Three.js avatar with visemes.
# No paid APIs. Follows robots.txt and a per-domain rate limit. Identity/voice/face
# still derive from internal energetics; the 3D avatar mirrors mouth/eyes in real time.
#
# Run:
#   python onbrain_online_avatar.py
#   Open http://localhost:8780/
#
# Env toggles:
#   ONB_ALLOW_ONLINE=0               # force offline mode (online is enabled by default)
#   ONB_SEEDS="https://example.com,https://en.wikipedia.org/wiki/AI"
#   ONB_PORT=8780
#   ONB_HOST=0.0.0.0
#
# Notes:
# - Uses only stdlib HTTP + BeautifulSoup; respects robots.txt; rate-limited; size-limited.
# - If you previously ran onbrain_groundbreaking.py, this will reuse onbrain.db.

import os, sys, subprocess, venv, json, time, asyncio, math, sqlite3, traceback, wave, random, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterable, Optional
from pathlib import Path

# ---------------- Bootstrapping: create .venv and re-exec ----------------
ROOT = Path.cwd() / "onbrain_online"
VENV = ROOT / ".venv"
REQ = [
    "fastapi==0.115.5", "uvicorn==0.32.0", "numpy==1.26.4", "networkx==3.3",
    "beautifulsoup4==4.12.3", "pydantic==2.9.2", "starlette==0.41.3",
    "websockets==12.0", "scipy==1.11.4", "sympy==1.13.3", "langdetect==1.0.9"
]
def ensure_venv_and_reexec():
    ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ONB_ONLINE_BOOTED") == "1":
        return
    if not VENV.exists():
        print(">> Creating venv at", VENV)
        venv.create(VENV, with_pip=True)
    pip = VENV / ("Scripts/pip.exe" if os.name=="nt" else "bin/pip")
    py  = VENV / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
    print(">> Upgrading pip and installing deps")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
    subprocess.check_call([str(pip), "install"] + REQ)
    env = os.environ.copy(); env["ONB_ONLINE_BOOTED"] = "1"
    print(">> Relaunching inside venv")
    os.execvpe(str(py), [str(py), __file__], env)

if __file__ == "<stdin>":
    script_path = ROOT / "onbrain_online_avatar.py"
    script_path.write_text(sys.stdin.read(), encoding="utf-8")
    __file__ = str(script_path)

ensure_venv_and_reexec()

# ---------------- Imports (post-venv) ----------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
import uvicorn
import numpy as np
from bs4 import BeautifulSoup
from scipy.signal import stft, get_window
from sympy import sympify, simplify, Eq, solve
from langdetect import detect, detect_langs
import urllib.parse, urllib.request, urllib.robotparser

# ---------------- Config & paths ----------------
PORT = int(os.getenv("ONB_PORT", "8780"))
HOST = os.getenv("ONB_HOST", "0.0.0.0")
TICK_SEC = float(os.getenv("ONB_TICK_SEC", "0.6"))
REFLECT_EVERY = int(os.getenv("ONB_REFLECT_EVERY", "4"))
SIGMA0 = float(os.getenv("ONB_SIGMA0", "0.9"))
GAMMA = float(os.getenv("ONB_GAMMA", "0.93"))
SIGMA_MIN = float(os.getenv("ONB_SIGMA_MIN", "0.10"))
ALLOW_ONLINE = os.getenv("ONB_ALLOW_ONLINE", "1") == "1"
DB_PATH = os.getenv("ONB_DB_PATH", str(ROOT / "onbrain.db"))
DEFAULT_SEEDS = [u.strip() for u in os.getenv("ONB_SEEDS", "").split(",") if u.strip()]

OUT_AUDIO = ROOT / "audio"; OUT_AUDIO.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities (DSP, embed, etc.) ----------------
def write_wav_mono16(path: Path, sr: int, samples: Iterable[float]) -> None:
    import wave
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

# embeddings (subword hashing)
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
        x=sha_to_u64(g); d=u_hash(x,D=D); s=sign_hash(sha_to_u64(g,"sign"))
        v[d]+=s
    nrm=np.linalg.norm(v)+1e-9
    return v/nrm

# annealing pieces
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

# ---------------- DB ----------------
class Memory:
    def __init__(self, path:str, D:int=512):
        self.path=path; self.D=D; self._init()
    def _init(self):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, lang TEXT, text TEXT, src TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeds(id INTEGER PRIMARY KEY, vec BLOB)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, type TEXT, json TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS energetics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits REAL, sfield REAL, L REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS captions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, caption TEXT, meta TEXT)""")
        # online exploration
        cur.execute("""CREATE TABLE IF NOT EXISTS seeds(url TEXT PRIMARY KEY, last REAL, ok INTEGER, robots TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS crawled(url TEXT PRIMARY KEY, ts REAL, sha TEXT, title TEXT)""")
        # persona
        cur.execute("""CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY CHECK (id=1), ts REAL, schema TEXT, avatar_svg TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS identity_history(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, schema TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS coherence(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, C REAL, comp TEXT)""")
        con.commit(); con.close()
    def teach(self, text:str, lang:str, src:str=""):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("INSERT INTO facts(ts,lang,text,src) VALUES(?,?,?,?)",(time.time(), lang, text, src))
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
    # identity
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

# ---------- Persona helpers ----------
def _u32(s:str)->int:
    import hashlib
    return int.from_bytes(hashlib.sha1(s.encode("utf-8","ignore")).digest()[:4],"little")
def _rng(seed:str): 
    import numpy as _np
    return _np.random.RandomState(_u32(seed))
def _palette(seed:str):
    import colorsys
    r=_rng("pal/"+seed); h=r.uniform(0,360)
    def hsl(h,s,l):
        r,g,b=colorsys.hls_to_rgb(h/360,l/100,s/100); return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return {"bg": hsl((h+180)%360, 25, 12), "fg": hsl(h, 35, 92),
            "accents": [hsl((h+30)%360,60,58), hsl((h+140)%360,60,58), hsl((h+230)%360,60,58)]}
def _avatar_svg(ident:dict, size:int=256)->str:
    pal=ident["style"]["palette"]; sh=ident["style"]["shapes"]; W=H=size
    def rect(x,y,w,h,fill): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" ry="24" fill="{fill}"/>'
    def circ(cx,cy,r,fill): return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'
    def path(d,fill): return f'<path d="{d}" fill="{fill}"/>'
    faceR={"round":92,"oval":82,"square":88,"hex":86}[sh["face"]]
    eye={"dot":(6,7),"almond":(10,5),"wide":(12,4)}[sh["eyes"]]
    brow=sh["brows"]; mouth=sh["mouth"]; hair=sh["hair"]; acc=sh["accessory"]
    g=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',rect(0,0,W,H,pal["bg"]), circ(W*0.5,H*0.52,faceR,pal["fg"])]
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
        g += [f'<rect x="{W*0.28}" y="{by-6}" width="{W*0.20}" height="6" fill="{pal["accents"][2]}"/>' ,
              f'<rect x="{W*0.52}" y="{by-6}" width="{W*0.20}" height="6" fill="{pal["accents"][2]}"/>']
    else:
        p(f"M{W*0.28},{by} Q{W*0.38},{by-16} {W*0.48},{by}"); p(f"M{W*0.52},{by} Q{W*0.62},{by-16} {W*0.72},{by}")
    my=H*0.67; amp={"smile":12,"neutral":5,"serif":2}[mouth]
    if mouth=="smile": g.append(path(f"M{W*0.40},{my} Q{W*0.50},{my+amp} {W*0.60},{my}", pal["bg"]))
    elif mouth=="serif": g.append(f'<rect x="{W*0.45}" y="{my-2}" width="{W*0.10}" height="3" fill="{pal["bg"]}"/>' )
    else: g.append(f'<rect x="{W*0.435}" y="{my-2}" width="{W*0.13}" height="4" fill="{pal["bg"]}"/>')
    if acc=="visor": g.append(f'<rect x="{W*0.30}" y="{H*0.40}" width="{W*0.40}" height="16" fill="{pal["accents"][0]}"/>')
    if acc=="antenna": g += [f'<path d="M{W*0.5},{H*0.12} L{W*0.5},{H*0.36}" fill="{pal["accents"][0]}"/>' , circ(W*0.5,H*0.12,6,pal["accents"][0])]
    if acc=="mono": g += [circ(W*0.62,H*0.45,18,pal["accents"][0]), circ(W*0.62,H*0.45,12,pal["bg"])]
    if acc=="ear": g += [circ(W*0.18,H*0.52,14,pal["accents"][1]), circ(W*0.82,H*0.52,14,pal["accents"][1])]
    g.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="14" fill="{pal["fg"]}" opacity="0.75">{ident["core"]["signature_emoji"]} {ident["core"]["name"]}</text>')
    g.append("</svg>"); return "".join(g)

# ---------- Voice + coherence (simplified) ----------
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
    t=np.arange(y.size)/sr
    y += 0.05*np.sin(2*np.pi*f0*t)
    y /= (np.max(np.abs(y))+1e-9)
    return y

def spectral_stats(y:np.ndarray, sr:int=22050)->Dict[str,float]:
    if y.size==0: return {"flatness":0.0,"centroid":0.0,"zcr":0.0}
    X = stft_mag(y, sr, 1024, 256)
    gmean = np.exp(np.mean(np.log(np.maximum(X,1e-12))))
    amean = np.mean(X)
    flat = float(gmean/ (amean+1e-9))
    freqs = np.linspace(0, sr/2, X.shape[0])
    centroid = float(np.sum(freqs[:,None]*X)/ (np.sum(X)+1e-9)) / (sr/2)
    zcr = float(np.mean(np.abs(np.diff(np.sign(y+1e-12)))))
    return {"flatness":flat, "centroid":centroid, "zcr":zcr}

# ---------- Web explore (safe) ----------
class Explorer:
    def __init__(self, mem:Memory, bus):
        self.mem=mem; self.bus=bus
        self.allow_online=ALLOW_ONLINE
        self.stop_flag=False
        self.allowed_hosts=set()
        # priming seeds
        for u in DEFAULT_SEEDS:
            self.add_seed(u)

    def set_allowed(self, hosts:List[str]):
        self.allowed_hosts = set(h.lower().strip() for h in hosts if h.strip())

    def add_seed(self, url:str):
        try:
            u = urllib.parse.urlparse(url)
            if not u.scheme.startswith("http"): return False
            con=sqlite3.connect(self.mem.path); cur=con.cursor()
            cur.execute("INSERT OR IGNORE INTO seeds(url,last,ok,robots) VALUES(?,?,?,?)",(url,0.0,1,""))
            con.commit(); con.close()
            return True
        except Exception:
            return False

    def robots_ok(self, url:str)->bool:
        try:
            u=urllib.parse.urlparse(url)
            if self.allowed_hosts and u.hostname and u.hostname.lower() not in self.allowed_hosts:
                return False
            base=f"{u.scheme}://{u.netloc}"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urllib.parse.urljoin(base, "/robots.txt"))
            rp.read()
            return rp.can_fetch("OnBrainBot/1.0", url)
        except Exception:
            return False

    def fetch_html(self, url:str, timeout=10, max_bytes=600_000)->str:
        req = urllib.request.Request(url, headers={"User-Agent":"OnBrainBot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type","")
            if "text/html" not in ct: return ""
            data = resp.read(max_bytes+1)
            if len(data)>max_bytes: return ""
            return data.decode(resp.headers.get_content_charset() or "utf-8", "ignore")

    def extract_text(self, html:str)->Tuple[str,List[str]]:
        soup=BeautifulSoup(html,"html.parser")
        for s in soup(["script","style","noscript"]): s.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        txt = " ".join(s.strip() for s in soup.get_text(" ").split())
        # simple paragraph-ish splits
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', txt) if len(p.strip())>40][:60]
        return title, parts

    async def loop(self):
        if not self.allow_online:
            return
        while not self.stop_flag:
            try:
                con=sqlite3.connect(self.mem.path); cur=con.cursor()
                cur.execute("SELECT url,last FROM seeds WHERE ok=1 ORDER BY last ASC LIMIT 1")
                row=cur.fetchone()
                con.close()
                if not row:
                    await asyncio.sleep(3.0); continue
                url,last=row; now=time.time()
                # rate limit per seed (>= 30s)
                if now - float(last) < 30.0:
                    await asyncio.sleep(3.0); continue
                if not self.robots_ok(url):
                    con=sqlite3.connect(self.mem.path); cur=con.cursor()
                    cur.execute("UPDATE seeds SET ok=0,last=? WHERE url=?", (now, url))
                    con.commit(); con.close()
                    continue
                html=""
                try:
                    html=self.fetch_html(url)
                except Exception:
                    html=""
                title, parts = ("","")
                if html:
                    title, parts = self.extract_text(html)
                    # teach top sentences
                    for p in parts[:10]:
                        try:
                            lang = detect(p)
                        except Exception:
                            lang = "en"
                        self.mem.teach(p, lang, src=url)
                    # mark crawled
                    sha=str(sha_to_u64(html,"sha"))
                    con=sqlite3.connect(self.mem.path); cur=con.cursor()
                    cur.execute("INSERT OR REPLACE INTO crawled(url,ts,sha,title) VALUES(?,?,?,?)",(url, now, sha, title))
                    con.commit(); con.close()
                # bump last
                con=sqlite3.connect(self.mem.path); cur=con.cursor()
                cur.execute("UPDATE seeds SET last=? WHERE url=?", (now, url))
                con.commit(); con.close()
                # publish event
                await self.bus.pub({"type":"crawl","data":{"url":url,"taught":len(parts) if html else 0,"title":title}})
            except Exception as e:
                await self.bus.pub({"type":"error","data":{"where":"explore","error":str(e),"trace":traceback.format_exc()}})
            await asyncio.sleep(2.0)

# ---------- Orchestrator (simplified vs GB) ----------
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
        self.explorer = Explorer(self.mem, self.bus)
        # identity
        self._ensure_identity_boot()

    def _ensure_identity_boot(self):
        if not self.mem.get_identity():
            ident=self._identity_from_seed("boot")
            self.mem.save_identity(ident, _avatar_svg(ident))

    def _identity_from_seed(self, affix:str)->dict:
        base=f"seed|{affix}|{time.time():.3f}"
        r=_rng(base)
        ident={
          "version":"2",
          "seed":base,
          "core":{"name": f"OnBrain {abs(_u32(base))%10000:04d}",
                  "motto": r.choice(["crystallize the noise","curiosity is compression","tension reveals shape","listen ↔ reflect"]),
                  "signature_emoji": r.choice(["🧠","🔷","✨","🔭","🌐"])},
          "style":{"palette": _palette(base),
                   "shapes":{"face": r.choice(["round","oval","square","hex"]),
                             "eyes": r.choice(["dot","almond","wide"]),
                             "brows": r.choice(["soft","straight","arched"]),
                             "mouth": r.choice(["smile","neutral","serif"]),
                             "hair": r.choice(["none","wave","spike","curl"]),
                             "accessory": r.choice(["none","mono","ear","visor","antenna"])
                            }},
          "voice":{"f0": 130 + 30*r.rand(),
                   "speaking_rate": 0.9 + 0.3*r.rand(),
                   "breathiness": 0.12 + 0.2*r.rand()},
          "locks":{"frozen": False}
        }
        return ident

    def _audio_maps(self, H:float, S:float):
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

    def _anneal_round(self):
        E, ids = self.mem.embeddings(max_items=192)
        if E.size==0: 
            return None
        N=E.shape[0]
        edges = ring_edges(N, k=max(4,min(12,N-1)))
        S=np.zeros(N)
        for i in range(N):
            var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self.rng)
            S[i]=stability(var)
        en=energetics(E,S,edges,self.state.sigma); en["sigma"]=self.state.sigma
        self.mem.log_energy(self.state.tick, self.state.sigma, en)
        # audio
        maps=self._audio_maps(en["H_bits"], en["S_field"])
        sig=self._synth_signal(1.6, 22050, maps)
        wav_path=OUT_AUDIO/f"onb_{self.state.tick}.wav"; write_wav_mono16(wav_path,22050,sig)
        # caption
        facts=self.mem.fact_text(ids[:4]); cap="; ".join([v.split("\n")[0][:60] for v in facts.values()]) if facts else "Exploring and crystallizing knowledge."
        # basic voice params
        f0 = 110 + 70*(1.0-en["H_bits"])
        rate = 0.9 + 0.5*(1.0-en["S_field"])
        breath = 0.12 + 0.25*en["H_bits"]
        y = synth_voice_from_memory(cap, f0, rate, breath, 22050, np.array(sig,dtype=np.float64))
        out = OUT_AUDIO/f"say_{self.state.tick}.wav"; write_wav_mono16(out, 22050, y.tolist())
        # small feedback
        st = spectral_stats(y, 22050); target_flat = 0.25 + 0.35*(1.0-en["H_bits"]); delta = 0.05*(target_flat - st["flatness"])
        self.state.sigma = float(np.clip(self.state.sigma + delta, SIGMA_MIN, 1.0))
        self.mem.log_caption(self.state.tick, cap, {"f0":f0,"rate":rate,"breath":breath})
        return en, cap, {"f0":f0,"rate":rate,"breath":breath}

    async def loop(self):
        while True:
            try:
                self.state.tick += 1
                if self.state.tick % REFLECT_EVERY == 0:
                    out=self._anneal_round()
                    if out:
                        en, cap, voice = out
                        await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en}})
            except Exception as e:
                await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
            await asyncio.sleep(TICK_SEC)

    # viseme calculation for browser avatar
    def visemes_from_audio(self, y:np.ndarray, sr:int=22050, fps:int=30)->List[Dict[str,float]]:
        win=int(sr/fps)
        N = max(1, len(y)//win)
        out=[]
        for i in range(N):
            seg=y[i*win:(i+1)*win]
            if seg.size==0: seg=np.zeros(win)
            amp=float(np.sqrt(np.mean(seg*seg)))
            # rough vowel likelihoods from spectral centroid
            stats=spectral_stats(seg, sr)
            c=stats["centroid"]
            v="rest"
            if amp<0.05: v="rest"
            elif c<0.25: v="O"
            elif c<0.45: v="A"
            else: v="E"
            out.append({"t": i/fps, "v": v, "amp":amp, "centroid":c})
        return out

# ---------------- API ----------------
app = FastAPI(title="OnBrain (Online Explorer + 3D Avatar)")
app.add_middleware(CORSIMiddleware:=CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

brain = OnBrain()

@app.on_event("startup")
async def _boot():
    asyncio.create_task(brain.loop())
    if ALLOW_ONLINE:
        asyncio.create_task(brain.explorer.loop())

@app.get("/", response_class=HTMLResponse)
def home():
    # 3D avatar with Three.js + WebSocket visemes
    return """<!doctype html><html><head><meta charset="utf-8"><title>OnBrain Online + 3D</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
body{font-family:system-ui,Segoe UI,Inter,sans-serif;margin:0;background:#0b0b0f;color:#eee}
.wrapper{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;padding:18px}
.card{background:#111;border:1px solid #222;border-radius:14px;padding:12px}
input,textarea{width:100%;padding:10px;margin:8px 0;border:1px solid #2a2a2a;border-radius:10px;background:#0e0e10;color:#eee}
button{padding:10px 16px;border:0;border-radius:10px;background:#4a7fff;color:#fff;cursor:pointer}
small{color:#aaa}
canvas{display:block;width:100%;height:420px;background:#050508;border-radius:12px}
.badge{padding:2px 8px;background:#1b1b22;border-radius:999px;border:1px solid #2a2a32;font-size:12px}
.flex{display:flex;gap:12px;align-items:center}
</style>
</head><body>
<div class="wrapper">
  <div class="card">
    <div class="flex"><div class="badge">3D Avatar</div><small id="state">connecting…</small></div>
    <canvas id="c"></canvas>
    <div class="flex">
      <button id="speak">Speak last caption</button>
      <button id="seed">Add Wikipedia Seed</button>
      <a href="/persona/avatar.svg" target="_blank">SVG avatar</a>
      <a href="/recent?table=captions" target="_blank">captions</a>
    </div>
  </div>
  <div class="card">
    <h3>Teach</h3>
    <form id="fTeach"><textarea id="teach" rows="4" placeholder="Teach a fact"></textarea><button>Teach</button></form>
    <div id="teachOut"></div>
    <h3>Think</h3>
    <form id="fThink"><textarea id="q" rows="4" placeholder="Ask anything (math, logic, design)"></textarea><button>Think</button></form>
    <pre id="ans"></pre>
    <h3>Explore</h3>
    <div class="flex">
      <input id="url" placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence"/>
      <button id="add">Add Seed</button>
    </div>
    <small>Online mode is <b>""" + ("ON" if ALLOW_ONLINE else "OFF") + """</b>. (set ONB_ALLOW_ONLINE=0 to force offline)</small>
  </div>
</div>
<script src="https://unpkg.com/three@0.158.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
<script>
const canvas = document.getElementById('c');
const stateEl = document.getElementById('state');
// Three.js scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050508);
const camera = new THREE.PerspectiveCamera(55, canvas.clientWidth/canvas.clientHeight, 0.1, 100);
camera.position.set(0, 1.2, 2.2);
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor=0.06;

const light = new THREE.DirectionalLight(0xffffff, 1.2); light.position.set(2,3,4);
scene.add(light, new THREE.AmbientLight(0x222244, 0.6));

const headGeo = new THREE.SphereGeometry(0.9, 64, 48);
const headMat = new THREE.MeshStandardMaterial({color:0x9cc2ff, metalness:0.1, roughness:0.6});
const head = new THREE.Mesh(headGeo, headMat); head.position.y = 1.0; scene.add(head);

const mouthCurve = new THREE.CatmullRomCurve3([new THREE.Vector3(-0.35,0.4,0.75), new THREE.Vector3(0,0.4,0.76), new THREE.Vector3(0.35,0.4,0.75)]);
let mouthGeom = new THREE.TubeGeometry(mouthCurve, 20, 0.02, 8, false);
let mouth = new THREE.Mesh(mouthGeom, new THREE.MeshStandardMaterial({color:0x111111, metalness:0.0, roughness:1.0}));
mouth.position.y = 0.35; scene.add(mouth);

function setViseme(v, amp){
  let mid = 0.4;
  let a = (v==="A") ? 0.09 : (v==="O" ? 0.04 : (v==="E" ? 0.02 : 0.005));
  let off = (v==="O") ? 0.02 : 0.0;
  const p0=new THREE.Vector3(-0.35, mid - a*0.6, 0.75+off);
  const p1=new THREE.Vector3(0,     mid + a*0.5, 0.76+off);
  const p2=new THREE.Vector3(0.35,  mid - a*0.6, 0.75+off);
  mouthCurve.points=[p0,p1,p2];
  mouth.geometry.dispose();
  mouth.geometry = new THREE.TubeGeometry(mouthCurve, 20, 0.02 + 0.02*Math.min(1,amp*8), 8, false);
}

let pal = {bg:0x050508, fg:0xffffff};
function setPalette(hex){ head.material.color.setHex(hex); }

function animate(){
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

// WebSocket for live energetics + visemes
let ws;
function connectWS(){
  ws = new WebSocket((location.protocol==="https:"?"wss":"ws")+"://"+location.host+"/ws");
  ws.onopen = ()=>{ stateEl.textContent = "connected"; };
  ws.onmessage = (ev)=>{
    try{
      const msg = JSON.parse(ev.data);
      if(msg.type==="visemes"){
        // play viseme timeline
        let i=0;
        const list=msg.data.list;
        const t0=performance.now();
        function step(){
          const t=(performance.now()-t0)/1000;
          while(i<list.length && list[i].t < t+0.03){
            setViseme(list[i].v, list[i].amp);
            i++;
          }
          if(i<list.length) requestAnimationFrame(step);
        }
        step();
      }else if(msg.type==="energetics"){
        const H = msg.data.H_bits;
        // color shift by H
        const hue = (0.6 + 0.4*(1.0-H)) % 1.0;
        const c = new THREE.Color().setHSL(hue, 0.55, 0.55);
        setPalette(c.getHex());
      }
    }catch(e){}
  }
  ws.onclose = ()=>{ stateEl.textContent = "disconnected"; setTimeout(connectWS, 1000); };
}
connectWS();

// UI handlers
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
 const text = caps.rows && caps.rows[0] ? (caps.rows[0][3]||"Crystallizing knowledge.") : "Crystallizing knowledge.";
 const r=await fetch('/persona/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
}
document.getElementById('add').onclick=async()=>{
 const u=document.getElementById('url').value.trim(); if(!u) return;
 await fetch('/explore/seed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u})})
}
document.getElementById('seed').onclick=async()=>{
 await fetch('/explore/seed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:"https://en.wikipedia.org/wiki/Artificial_intelligence"})})
}
</script>
</body></html>"""

@app.get("/persona/avatar.svg", response_class=HTMLResponse)
def persona_avatar():
    svg = brain.mem.get_avatar_svg()
    if not svg:
        ident=brain._identity_from_seed("api"); brain.mem.save_identity(ident, _avatar_svg(ident)); svg=brain.mem.get_avatar_svg()
    return svg

@app.post("/persona/say")
async def persona_say(payload: Dict[str,Any] = Body(...)):
    text = str(payload.get("text","" )).strip()
    if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
    # use last anneal wav
    sr=22050
    try:
        paths = sorted(list((OUT_AUDIO).glob("onb_*.wav")), key=lambda p:p.stat().st_mtime)
        last = paths[-1]
        import wave
        with wave.open(str(last),'rb') as wf:
            n=wf.getnframes(); data=wf.readframes(n)
            y=np.frombuffer(data, dtype=np.int16).astype(np.float32)/32767.0
    except Exception:
        y=np.zeros(int(sr*0.8),dtype=np.float32)
    # derive current voice
    f0=128.0; rate=1.0; breath=0.15
    synth = synth_voice_from_memory(text, f0, rate, breath, sr, y)
    out = OUT_AUDIO/"persona_say.wav"; write_wav_mono16(out, sr, synth.tolist())
    # push visemes to WS
    timeline = brain.visemes_from_audio(synth, sr)
    await brain.bus.pub({"type":"visemes","data":{"list":timeline}})
    return {"ok":True,"path":str(out),"frames":len(timeline)}

@app.post("/teach")
def teach(payload: Dict[str, str] = Body(...)):
    text = payload.get("text","").strip()
    if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    fid = brain.mem.teach(text, lang, src="user")
    brain.mem.log(brain.state.tick, "teach", {"id":fid,"lang":lang})
    return {"ok":True, "id": fid, "lang": lang}

@app.post("/think")
def think(payload: Dict[str,str] = Body(...)):
    text = payload.get("text","").strip()
    if not text: return JSONResponse({"ok":False,"error":"empty"}, status_code=400)
    # math / simple reasoning
    try:
        if "=" in text:
            left,right=text.split("=",1)
            expr_l=sympify(left); expr_r=sympify(right); sol=solve(Eq(expr_l,expr_r))
            ans=f"solutions: {sol}"
        else:
            ans=str(simplify(sympify(text)))
    except Exception:
        ans="(freeform reasoning stub) break into steps: clarify → recall → draft → critique → finalize."
    # log + retrieve context
    brain.mem.log(brain.state.tick, "think", {"q":text})
    return {"ok":True,"answer":ans}

@app.get("/status")
def status():
    return {"ok":True, "tick":brain.state.tick, "sigma":brain.state.sigma, "online": ALLOW_ONLINE}

@app.get("/recent")
def recent(table: str = Query("facts"), limit: int = Query(50)):
    con=sqlite3.connect(brain.mem.path); cur=con.cursor()
    try:
        cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
        rows=cur.fetchall()
    except Exception:
        rows=[]
    con.close()
    return {"ok": True, "rows": rows}

# ---------- Online exploration endpoints ----------
@app.post("/explore/seed")
def explore_seed(payload: Dict[str, str] = Body(...)):
    if not ALLOW_ONLINE:
        return JSONResponse({"ok":False,"error":"online_disabled"}, status_code=403)
    url = payload.get("url","").strip()
    ok = brain.explorer.add_seed(url)
    return {"ok":ok}

@app.post("/explore/allow")
def explore_allow(payload: Dict[str, Any] = Body(...)):
    if not ALLOW_ONLINE:
        return JSONResponse({"ok":False,"error":"online_disabled"}, status_code=403)
    hosts = payload.get("hosts",[])
    brain.explorer.set_allowed(hosts)
    return {"ok":True,"hosts":list(brain.explorer.allowed_hosts)}

@app.get("/explore/status")
def explore_status():
    return {"ok":True, "online": ALLOW_ONLINE, "allowed": list(brain.explorer.allowed_hosts)}

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
    print(f"Open: http://{HOST}:{PORT}/ (online={'ON' if ALLOW_ONLINE else 'OFF'})")
    uvicorn.run(app, host=HOST, port=PORT)
