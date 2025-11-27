#!/usr/bin/env python3
# Seed-Crystal Brain - Autonomous, Offline-First, Single File
# - Continuous loop: ingest -> anneal/crystallize -> energetics -> sonify -> STFT->attention -> captions -> reflect
# - 18,000-node avatar rendered from mind state, streamed over /avatar (binary WS)
# - Optional UI at / (Three.js viewer + controls). No interaction required to run.
# - Optional Ollama refine (falls back to heuristics).
# - No SciPy dependency; uses NumPy FFT + wave module.
#
# Run: python seed_crystal_brain.py
# Env (optional):
#   ALLOW_ONLINE=0|1 (default 0) | SC_TICK_SEC=0.8 | SC_REFLECT_EVERY=5
#   OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=llama3
#   SC_PORT=8767 SC_HOST=0.0.0.0
import os, sys, venv, subprocess, json, time, asyncio, math, sqlite3, base64, traceback, random, struct, threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

ROOT = Path.cwd() / "seed_crystal_agi"
VENV = ROOT / ".venv"
REQ = [
    "fastapi==0.115.5",
    "uvicorn==0.32.0",
    "requests==2.32.3",
    "beautifulsoup4==4.12.3",
    "numpy==1.26.4",
    "networkx==3.3",
    "starlette==0.41.3"
]

def ensure_venv_and_reexec():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "state").mkdir(exist_ok=True, parents=True)
    if os.environ.get("SC_BOOTED") == "1":
        return
    if not VENV.exists():
        print("[setup] Creating venv:", VENV)
        venv.create(VENV, with_pip=True)
    pip = VENV / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print("[setup] Installing deps...")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"], stdout=sys.stdout)
    subprocess.check_call([str(pip), "install"] + REQ, stdout=sys.stdout)
    env = os.environ.copy()
    env["SC_BOOTED"] = "1"
    print("[setup] Relaunching inside venv...")
    os.execvpe(str(py), [str(py), __file__], env)

# Allow running from /mnt/data sandbox by copying into project dir when started via stdin
if __file__ == "<stdin>":
    script_target = ROOT / "seed_crystal_brain.py"
    script_target.write_text(sys.stdin.read(), encoding="utf-8")
    __file__ = str(script_target)

ensure_venv_and_reexec()

# --- Imports after bootstrap ---
import numpy as np
import networkx as nx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

# --- Config ---
PORT = int(os.getenv("SC_PORT", "8767"))
HOST = os.getenv("SC_HOST", "0.0.0.0")
DB_PATH = os.getenv("SC_DB_PATH", str(ROOT / "seed_crystal.db"))
SC_TICK_SEC = float(os.getenv("SC_TICK_SEC", "0.8"))
SC_REFLECT_EVERY = int(os.getenv("SC_REFLECT_EVERY", "5"))
ALLOW_ONLINE = int(os.getenv("ALLOW_ONLINE", "1")) # allow crawl by default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
AUTONOMOUS_INGEST_EVERY = int(os.getenv("SC_AUTONOMOUS_INGEST_EVERY", "20"))
SIGMA0 = float(os.getenv("SC_SIGMA0", "0.8"))
GAMMA = float(os.getenv("SC_GAMMA", "0.92"))
SIGMA_MIN = float(os.getenv("SC_SIGMA_MIN", "0.12"))

OUT_AUDIO = ROOT / "audio"; OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_SHAPES = ROOT / "shapes"; OUT_SHAPES.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "state" / "corpus"; CORPUS.mkdir(parents=True, exist_ok=True)

# --- Seed offline corpus (only once) ---
_seed = CORPUS / "00_manifesto.txt"
if not _seed.exists():
    _seed.write_text(
        "Seed-Crystal Manifesto:\n"
        "We anneal noisy bits into stable memory facets, sonify our state, and speak captions.\n"
        "The avatar is not a mask; it is the field itself rendered as 18k nodes.\n",
        encoding="utf-8"
    )

# --- Utils: WAV writing (mono 16-bit) ---
def write_wav_mono16(path: Path, sr: int, samples: List[float]) -> None:
    import wave, array
    x = np.asarray(samples, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:


        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(array.array("h", x).tobytes())

# --- Math / Embedding ---
_P = (1 << 61) - 1
_A = 1371731309
_B = 911382323

def sha_to_u64(s: str, salt: str = "") -> int:
    import hashlib
    h = hashlib.sha256((salt + s).encode("utf-8", "ignore")).digest()
    return int.from_bytes(h[:8], "little")

def u_hash(x: int, a: int = _A, b: int = _B, p: int = _P, D: int = 512) -> int:
    return ((a * x + b) % p) % D

def sign_hash(x: int) -> int:
    return 1 if (x ^ (x >> 1) ^ (x >> 2)) & 1 else -1

def tokenize(text: str) -> List[str]:
    return [w for w in text.replace("\n", " ").lower().split() if w.strip()]

def signed_hash_embedding(tokens: List[str], D: int = 512, eps: float = 1e-8) -> np.ndarray:
    v = np.zeros(D, dtype=np.float64)
    for t in tokens:
        x = sha_to_u64(t)
        d = u_hash(x, D=D)
        s = sign_hash(sha_to_u64(t, salt="sign"))
        v[int(d)] += s
    nrm = np.linalg.norm(v) + eps
    return v / nrm

def embed_text(text: str, D: int = 512) -> np.ndarray:
    return signed_hash_embedding(tokenize(text), D=D)

# --- Annealing / Energetics ---
class Phase:
    RAW="raw"; GEL="gel"; CRYSTAL="crystal"

def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def knn_indices(E: np.ndarray, i: int, k: int = 8) -> List[int]:
    x = E[i]
    sims = (E @ x) / (np.linalg.norm(E, axis=1) * (np.linalg.norm(x) + 1e-9) + 1e-12)
    order = np.argsort(-sims)
    return [j for j in order if j != i][:k]

def monte_carlo_variance(E: np.ndarray, i: int, k: int, sigma: float, M: int = 6, rng=None) -> float:
    if rng is None:
        rng = np.random.RandomState(7)
    idx = knn_indices(E, i, k=max(1, min(k, E.shape[0]-1)))
    vals = []
    D = E.shape[1]
    for _ in range(M):
        ei = E[i] + sigma * rng.normal(0.0, 1.0, size=D)
        ei = ei / (np.linalg.norm(ei) + 1e-9)
        sims = []
        for j in idx:
            ej = E[j] + sigma * rng.normal(0.0, 1.0, size=D)
            ej = ej / (np.linalg.norm(ej) + 1e-9)
            sims.append(cos_sim(ei, ej))
        vals.append(max(sims) if sims else 0.0)
    return float(np.var(vals))

def stability_score(var_sigma: float) -> float:
    return 1.0 / (1.0 + var_sigma)

def anneal_schedule(sigma0: float, gamma: float, step: int, sigma_min: float) -> float:
    return max(sigma0 * (gamma ** step), sigma_min)

def expected_cos_with_noise(ei: np.ndarray, ej: np.ndarray, sigma: float, M: int = 4) -> float:
    rng = np.random.RandomState(11)
    sims = []
    for _ in range(M):
        ei_n = ei + sigma * rng.normal(0.0, 1.0, size=ei.shape); ei_n /= (np.linalg.norm(ei_n)+1e-9)
        ej_n = ej + sigma * rng.normal(0.0, 1.0, size=ej.shape); ej_n /= (np.linalg.norm(ej_n)+1e-9)
        sims.append(float(np.dot(ei_n, ej_n)))
    return float(np.mean(sims))

def weights_tension(E: np.ndarray, edges: np.ndarray, sigma: float, M: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    w = np.zeros(len(edges), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        w[k] = expected_cos_with_noise(E[i], E[j], sigma=sigma, M=M)
    tau = 1.0 - w
    return w, tau

def energetics(E: np.ndarray, S: np.ndarray, edges: np.ndarray, sigma: float) -> dict:
    N = E.shape[0]
    if len(edges) == 0:
        return {"H_bits": float(np.mean(1.0 - S) if N else 0.0), "S_field": 0.0, "L": 0.0}
    w, tau = weights_tension(E, edges, sigma=sigma)
    H_bits = float(np.mean(1.0 - S) if N else 0.0)
    S_field = float(np.mean(tau))
    L = float(np.sum(tau * tau))
    return {"H_bits": H_bits, "S_field": S_field, "L": L}

# --- Sonification / STFT / Attention -> Captions ---
def synth_signal(seconds: float, sr: int, a_fn, m_fn, rho_fn, fc_fn, alpha: float = 0.8, beta: float = 0.4) -> List[float]:
    n = int(seconds * sr)
    out = []
    for i in range(n):
        t = i / sr
        a = a_fn(t); m = m_fn(t); rho = rho_fn(t); fc = max(5.0, fc_fn(t))
        y = a * (1.0 + beta * math.sin(2.0 * math.pi * m * t)) * math.sin(2.0 * math.pi * fc * t + alpha * math.sin(2.0 * math.pi * rho * t))
        out.append(y)
    return out



def default_maps(H_bits: float, S_field: float, latency: float, fitness: float, fmin: float = 110.0, fdelta: float = 440.0):
    H = max(0.0, min(1.0, H_bits))
    S = max(0.0, min(1.0, S_field))
    L = max(0.0, min(1.0, latency))
    F = max(0.0, min(1.0, fitness))
    def a_fn(t): return 0.25 + 0.5 * (1.0 - H) * (1.0 - S)
    def m_fn(t): return 2.0 + 10.0 * S
    def rho_fn(t): return 0.2 + 3.0 * (1.0 - L)
    def fc_fn(t): return fmin + fdelta * F
    return {"a": a_fn, "m": m_fn, "rho": rho_fn, "fc": fc_fn}

def stft_mag(x: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
    # Simple STFT mag using NumPy
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    w = np.hanning(win)
    T = 1 + (len(x) - win) // hop
    F = win // 2 + 1
    X = np.zeros((F, T), dtype=np.float64)
    for t in range(T):
        s = t * hop
        seg = x[s:s+win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.fft.rfft(seg * w)
        X[:, t] = np.abs(spec)
    return X

def make_bands(F: int, H: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, F, H + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(H)]

def head_features(X: np.ndarray, bands: List[Tuple[int, int]]) -> np.ndarray:
    F, T = X.shape
    H = len(bands)
    E = np.zeros((H, T), dtype=np.float64)
    for h, (a, b) in enumerate(bands):
        if b <= a: b = min(a + 1, F)
        E[h] = X[a:b].mean(axis=0)
    d1 = np.pad(np.diff(E, axis=1), ((0,0),(1,0)))
    d2 = np.pad(np.diff(d1, axis=1), ((0,0),(1,0)))
    return np.stack([E, d1, d2], axis=-1)

def project_and_attention(V: np.ndarray, E_mem: np.ndarray, d: int, sigma_temp: float) -> Dict[str, Any]:
    H, T, F3 = V.shape
    D = E_mem.shape[1]
    rng = np.random.RandomState(1234)
    Wk = rng.normal(0, 1.0 / math.sqrt(D), size=(D, d))
    Wqs = rng.normal(0, 1.0, size=(H, F3, d))
    K = E_mem @ Wk
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    shapes = []
    tau = max(1e-3, sigma_temp)
    for h in range(H):
        Q = V[h] @ Wqs[h]
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        S = (Q @ K.T) / (d * tau)
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= (P.sum(axis=1, keepdims=True) + 1e-12)
        for t in range(V.shape[1]):
            w = P[t]
            top = np.argsort(-w)[:8]
            shapes.append({"head": int(h), "t": int(t), "ids": top.astype(int).tolist(), "weights": w[top].astype(float).tolist()})
    return {"shapes": shapes}

def fetch_summaries(db_path: str) -> Dict[int, str]:
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute("SELECT id, summary FROM facets")
    out = {int(i): (s or "") for (i, s) in cur.fetchall()}
    con.close(); return out

def kw(text: str, k: int = 10) -> str:
    return " ".join(text.replace("\n", " ").split()[:k])

def captions_from_shapes(db_path: str, shapes: Dict[str, Any], top_k: int = 3, window: int = 5, stride: int = 5, hbits: float = None, sfield:
float = None) -> Dict[str, Any]:
    id2sum = fetch_summaries(db_path)
    by_t: Dict[int, List[Tuple[List[int], List[float]]]] = {}
    T = 0
    for rec in shapes["shapes"]:
        t = int(rec["t"]); T = max(T, t+1)
        by_t.setdefault(t, []).append((rec["ids"], rec["weights"]))
    caps = []; t0 = 0
    while t0 < T:
        t1 = min(T-1, t0 + window - 1)
        score: Dict[int, float] = {}; denom = 0.0
        for t in range(t0, t1+1):
            for ids, wts in by_t.get(t, []):
                for i, w in zip(ids, wts):
                    score[i] = score.get(i, 0.0) + float(w); denom += float(w)
        if denom > 0:
            for i in list(score.keys()):
                score[i] /= denom
        top = sorted(score.items(), key=lambda x: -x[1])[:top_k]
        top_ids = [i for i,_ in top]
        phrases = [kw(id2sum.get(i, ""), 10) for i in top_ids]
        cap = {"t_start_frame": t0, "t_end_frame": t1, "top_ids": top_ids, "weights": [float(w) for _, w in top], "caption": "; ".join([p for p
in phrases if p])}
        if hbits is not None: cap["H_bits"] = float(hbits)
        if sfield is not None: cap["S_field"] = float(sfield)
        caps.append(cap); t0 += stride
    return {"captions": caps, "meta": {"window": window, "stride": stride, "top_k": top_k}}

# --- Memory Store (SQLite) ---
class MemoryStore:
    def __init__(self, path: str, D: int = 512):
        self.path = path; self.D = D; self._init()

    def _init(self):


        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, tension REAL, energy
REAL, size INTEGER)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, text TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, json TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, url TEXT, title TEXT, text TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS facets (id INTEGER PRIMARY KEY, summary TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, vec BLOB)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS energetics (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits
REAL, sfield REAL, L REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS captions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, caption TEXT, top_ids
TEXT, weights TEXT)""")
        con.commit(); con.close()

    def add_state(self, tick: int, tension: float, energy: float, size: int):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO states(ts, tick, tension, energy, size) VALUES(?,?,?,?,?)", (time.time(), tick, tension, energy, size))
        con.commit(); con.close()

    def add_reflection(self, tick: int, text: str):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO reflections(ts, tick, text) VALUES(?,?,?)", (time.time(), tick, text))
        con.commit(); con.close()

    def add_suggestion(self, tick: int, js: Dict[str, Any]):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO suggestions(ts, tick, json) VALUES(?,?,?)", (time.time(), tick, json.dumps(js)))
        con.commit(); con.close()

    def add_doc_with_embed(self, url: str, title: str, text: str):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO docs(ts, url, title, text) VALUES(?,?,?,?)", (time.time(), url, title, text))
        doc_id = cur.lastrowid
        summary = (text.strip().split("\n")[0] if text else title)[:280]
        e = embed_text(text or title, D=self.D).astype(np.float32)
        cur.execute("INSERT OR REPLACE INTO facets(id, summary) VALUES(?,?)", (doc_id, summary))
        cur.execute("INSERT OR REPLACE INTO embeddings(id, vec) VALUES(?,?)", (doc_id, e.tobytes()))
        con.commit(); con.close()
        return doc_id

    def add_energetics(self, tick: int, sigma: float, H_bits: float, S_field: float, L: float):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO energetics(ts, tick, sigma, hbits, sfield, L) VALUES(?,?,?,?,?,?)", (time.time(), tick, sigma, H_bits, S_field,
L))
        con.commit(); con.close()

    def add_caption(self, tick: int, caption: str, top_ids: List[int], weights: List[float]):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO captions(ts, tick, caption, top_ids, weights) VALUES(?,?,?, ?, ?)", (time.time(), tick, caption,
json.dumps(top_ids), json.dumps(weights)))
        con.commit(); con.close()

    def get_embeddings(self, max_items: int | None = None) -> Tuple[np.ndarray, List[int]]:
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("SELECT id, vec FROM embeddings ORDER BY id ASC")
        rows = cur.fetchall(); con.close()
        if not rows:
            return np.zeros((0, 0), dtype=np.float64), []
        ids = [int(r[0]) for r in rows]
        arrs = [np.frombuffer(r[1], dtype=np.float32).astype(np.float64) for r in rows]
        E = np.stack(arrs, axis=0)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        if max_items and len(ids) > max_items:
            idx = np.random.RandomState(123).choice(len(ids), size=max_items, replace=False)
            E = E[idx]; ids = [ids[i] for i in idx]
        return E, ids

    def recent(self, table: str, limit: int = 50):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall(); con.close()
        return rows

# --- Cube Simulation (physical analogy) ---
@dataclass
class Node:
    id: int; pos: np.ndarray; fixed: bool=False

@dataclass
class Bond:
    a: int; b: int; k: float=0.15; rest: float=1.0

class Cube:
    def __init__(self, n_per_edge: int = 6, seed: int = 42):
        np.random.seed(seed)
        self.G = nx.Graph(); self.tick = 0
        idc = 0
        for x in range(n_per_edge):
            for y in range(n_per_edge):
                for z in range(n_per_edge):
                    p = np.array([x, y, z], dtype=float)
                    p = 2 * (p / (n_per_edge - 1)) - 1
                    self.G.add_node(idc, node=Node(id=idc, pos=p, fixed=False)); idc += 1
        def idx(x, y, z): return x * (n_per_edge**2) + y * n_per_edge + z
        for x in range(n_per_edge):
            for y in range(n_per_edge):
                for z in range(n_per_edge):
                    u = idx(x, y, z)
                    for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
                        nx_ = x+dx; ny_ = y+dy; nz_ = z+dz
                        if nx_ < n_per_edge and ny_ < n_per_edge and nz_ < n_per_edge:
                            v = idx(nx_, ny_, nz_)
                            self.G.add_edge(u, v, bond=Bond(a=u, b=v, k=0.15, rest=2/(n_per_edge-1)))
        corners = [0, idx(n_per_edge-1,0,0), idx(0,n_per_edge-1,0), idx(0,0,n_per_edge-1),
                idx(n_per_edge-1,n_per_edge-1,0), idx(n_per_edge-1,0,n_per_edge-1),
                idx(0,n_per_edge-1,n_per_edge-1), idx(n_per_edge-1,n_per_edge-1,n_per_edge-1)]
        for c in corners: self.G.nodes[c]['node'].fixed = True

    def step(self, dt: float = 0.1, damp: float = 0.9):
        forces = {i: np.zeros(3) for i in self.G.nodes}


        for u,v,data in self.G.edges(data=True):
            b: Bond = data['bond']
            pu = self.G.nodes[u]['node'].pos; pv = self.G.nodes[v]['node'].pos
            d = pv - pu; L = float(np.linalg.norm(d) + 1e-8)
            F = b.k * (L - b.rest) * (d / L)
            forces[u] += F; forces[v] -= F
        for i, data in self.G.nodes(data=True):
            n: Node = data['node']
            if n.fixed: continue
            n.pos += dt * forces[i]; n.pos *= damp
        self.tick += 1

    def metrics(self) -> Dict[str, float]:
        tension=0.0; energy=0.0
        for u,v,data in self.G.edges(data=True):
            b: Bond = data['bond']
            pu = self.G.nodes[u]['node'].pos; pv = self.G.nodes[v]['node'].pos
            L = float(np.linalg.norm(pv - pu))
            tension += abs(L - b.rest)
            energy += 0.5 * b.k * (L - b.rest)**2
        m = max(1, self.G.number_of_edges())
        return {"tension": tension/m, "energy": energy/m, "size": self.G.number_of_nodes()}

# --- Reflection / Heuristic adjust ---
def make_reflection(tick: int, m: Dict[str, float]) -> str:
    t = m.get("tension", 0.0); e = m.get("energy", 0.0); n = m.get("size", 0)
    mood = "calm" if t < 0.01 else "strained" if t < 0.04 else "overstretched"
    return f"[tick={tick}] Tension={t:.5f} Energy={e:.5f} Size={n}. State feels {mood}. Strategy: {'tighten springs' if t < 0.02 else 'loosen a bit' if t > 0.05 else 'hold steady'}."

def heuristic_adjust(m: Dict[str, float]) -> Dict[str, float]:
    t = m.get("tension", 0.0)
    if t < 0.015: return {"k_scale": 1.08, "rest_scale": 0.98}
    if t > 0.050: return {"k_scale": 0.95, "rest_scale": 1.03}
    return {"k_scale": 1.00, "rest_scale": 1.00}

def ask_ollama_refine(metrics: Dict[str, float], reflection: str) -> Dict[str, Any]:
    sys_p = "You optimize a spring-mesh cube. Reply STRICT JSON only: {\"k_scale\":float, \"rest_scale\":float}."
    user_p = f"Metrics: {metrics}\nReflection: {reflection}\nReturn only JSON."
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                        json={"model": OLLAMA_MODEL, "messages":[{"role":"system","content":sys_p},{"role":"user","content":user_p}],
"stream": False},
                        timeout=8)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "").strip()
        data = json.loads(content)
        if not all(k in data for k in ("k_scale","rest_scale")): raise ValueError("missing keys")
        return {"ok": True, "adjust": data, "raw": content}
    except Exception as e:
        return {"ok": False, "adjust": heuristic_adjust(metrics), "error": str(e)}

# --- Simple edges helper ---
def simple_edges(N: int, k: int = 6) -> np.ndarray:
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % N))
    if not edges: return np.zeros((0,2), dtype=np.int32)
    return np.array(sorted({tuple(sorted(e)) for e in edges}), dtype=np.int32)

# --- Broadcaster for /ws ---
class Broadcaster:
    def __init__(self): self._subs: List[asyncio.Queue] = []
    def subscribe(self): q = asyncio.Queue(maxsize=200); self._subs.append(q); return q
    async def publish(self, msg: Dict[str, Any]):
        dead = []
        for q in self._subs:
            try: await q.put(msg)
            except asyncio.QueueFull: dead.append(q)
        if dead:
            for d in dead:
                try: self._subs.remove(d)
                except: pass

# --- 18k-node Avatar Engine ---
class Avatar18k:
    def __init__(self, n=18000, seed=123):
        self.n = int(n)
        rng = np.random.RandomState(seed)
        r = 0.8 * np.cbrt(rng.rand(self.n))
        theta = 2*np.pi*rng.rand(self.n)
        phi = np.arccos(2*rng.rand(self.n)-1)
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        self.pos = np.stack([x,y,z], axis=1).astype(np.float32)
        self.vel = np.zeros_like(self.pos, dtype=np.float32)
        self._bytes = None
        self._lock = threading.Lock()
        self.shape_id = 0

    @staticmethod
    def _hash_str(s: str) -> int:
        h = 1469598103934665603
        for ch in s.encode("utf-8","ignore"):
            h ^= ch; h *= 1099511628211; h &= (1<<64)-1
        return h if h>=0 else -h

    def _target_field(self, shape_id:int) -> np.ndarray:
        N = self.n; P = self.pos
        t = shape_id % 4
        if t == 0: # sphere
            radius = 0.9
            return radius * P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-6)
        if t == 1: # torus
            R, r = 0.8, 0.25
            x, y, z = P[:,0], P[:,1], P[:,2]
            q = np.sqrt(x*x + y*y)+1e-6


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

# --- Orchestrator ---
class Orchestrator:
    def __init__(self):
        self.cube = Cube(n_per_edge=6)
        self.mem = MemoryStore(DB_PATH)
        self.tick = 0
        self.bus = Broadcaster()
        self.anneal_step = 0
        self.sigma = SIGMA0
        self.theta_gel = 0.25
        self.theta_crystal = 0.08
        self.rng = np.random.RandomState(101)
        self.hbits = 0.5; self.sfield = 0.5
        self.last_caption_text = ""
        self.avatar = Avatar18k(n=18000)

    def snapshot(self) -> Dict[str, Any]:
        m = self.cube.metrics()
        return {"tick": self.tick, **m, "sigma": self.sigma, "H_bits": self.hbits, "S_field": self.sfield}

    def _ingest_local(self) -> Optional[int]:
        files = sorted(CORPUS.glob("**/*"))
        if not files: return None
        pick = random.choice([f for f in files if f.is_file()])
        try:
            text = pick.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        title = pick.name
        return self.mem.add_doc_with_embed(pick.as_uri(), title, text)

    def _ingest_online(self) -> Optional[int]:
        # Minimal, safe fetch (e.g., wikipedia page); only if allowed.
        url = random.choice([
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Information_theory",
            "https://en.wikipedia.org/wiki/Signal_processing"
        ])
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"SeedCrystal/1.0"}); r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","noscript"]): tag.decompose()
            title = (soup.title.text.strip() if soup.title else url)[:200]
            import re
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:10000]
            return self.mem.add_doc_with_embed(url, title, text)
        except Exception:
            return None

    def _anneal_and_process(self):
        E, ids = self.mem.get_embeddings(max_items=128)
        if E.size == 0: return None
        N = E.shape[0]
        edges = simple_edges(N, k=max(4, min(12, N - 1)))
        S = np.zeros(N, dtype=np.float64)
        for i in range(N):
            var = monte_carlo_variance(E, i, k=min(8, N - 1), sigma=self.sigma, M=4, rng=self.rng)
            S[i] = stability_score(var)
        en = energetics(E, S, edges, self.sigma)
        self.mem.add_energetics(self.tick, self.sigma, en["H_bits"], en["S_field"], en["L"])
        self.hbits = float(en["H_bits"]); self.sfield = float(en["S_field"])

        # Sonify -> STFT -> Attention -> Captions
        maps = default_maps(H_bits=self.hbits, S_field=self.sfield, latency=0.2, fitness=max(0.0, 1.0 - self.hbits))
        sig = synth_signal(seconds=1.6, sr=22050, a_fn=maps["a"], m_fn=maps["m"], rho_fn=maps["rho"], fc_fn=maps["fc"], alpha=0.8, beta=0.4)
        wav_path = OUT_AUDIO / f"sonification_{self.tick}.wav"
        write_wav_mono16(wav_path, 22050, sig)

        X = stft_mag(np.asarray(sig, dtype=np.float64), sr=22050, win=1024, hop=256)
        bands = make_bands(X.shape[0], H=4)
        V = head_features(X, bands)
        shapes = project_and_attention(V, E_mem=E, d=24, sigma_temp=max(self.sigma, SIGMA_MIN))


        caps = captions_from_shapes(DB_PATH, shapes, top_k=3, window=6, stride=6, hbits=self.hbits, sfield=self.sfield)
        if caps["captions"]:
            last = caps["captions"][-1]
            self.last_caption_text = last.get("caption", "") or self.last_caption_text
            self.mem.add_caption(self.tick, self.last_caption_text, last.get("top_ids", []), last.get("weights", []))

        with open(OUT_SHAPES / f"shapes_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(shapes, f, ensure_ascii=False)
        with open(OUT_SHAPES / f"captions_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(caps, f, ensure_ascii=False)

        self.anneal_step += 1
        self.sigma = anneal_schedule(SIGMA0, GAMMA, self.anneal_step, SIGMA_MIN)
        return {"energetics": en, "caption": (caps["captions"][-1] if caps["captions"] else None)}

    async def run(self):
        while True:
            try:
                self.cube.step(); self.tick += 1
                m = self.cube.metrics()
                self.mem.add_state(self.tick, m["tension"], m["energy"], m["size"])
                await self.bus.publish({"type":"metrics","data":{"tick":self.tick, **m, "sigma": self.sigma, "H_bits":self.hbits,
"S_field":self.sfield}})

                # Autonomous ingest
                if self.tick % AUTONOMOUS_INGEST_EVERY == 0:
                    doc_id = self._ingest_local() or (self._ingest_online() if ALLOW_ONLINE else None)
                    if doc_id:
                        await self.bus.publish({"type":"ingest","data":{"doc_id":int(doc_id)}})

                # Reflection + adjust + anneal pipeline
                if self.tick % SC_REFLECT_EVERY == 0:
                    ref = make_reflection(self.tick, m); self.mem.add_reflection(self.tick, ref)
                    await self.bus.publish({"type":"reflection","data":{"tick":self.tick,"text":ref}})
                    r = ask_ollama_refine(m, ref)
                    adjust = r["adjust"]; self.mem.add_suggestion(self.tick, adjust)
                    # apply to cube
                    ks = float(adjust.get("k_scale",1.0)); rs = float(adjust.get("rest_scale",1.0))
                    ks = max(0.25, min(ks, 4.0)); rs = max(0.5, min(rs, 1.5))
                    for _,_,data in self.cube.G.edges(data=True):
                        b: Bond = data["bond"]; b.k *= ks; b.rest *= rs
                    await self.bus.publish({"type":"suggestion","data":{"tick":self.tick, **adjust, "heuristic": not r.get("ok")}})

                    out = self._anneal_and_process()
                    if out:
                        await self.bus.publish({"type":"energetics","data":{"tick":self.tick, **out["energetics"], "sigma": self.sigma}})
                        if out["caption"]:
                            await self.bus.publish({"type":"caption","data":{"tick":self.tick, **out["caption"]}})

                # Update avatar every tick regardless of UI
                self.avatar.update(self.hbits, self.sfield, self.last_caption_text)

            except Exception as e:
                await self.bus.publish({"type":"error","data":{"tick":self.tick, "error": str(e), "trace": traceback.format_exc()}})
            await asyncio.sleep(SC_TICK_SEC)

# --- API & Server ---
app = FastAPI(title="Seed-Crystal Brain")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
orch = Orchestrator()

@app.on_event("startup")
async def boot():
    asyncio.create_task(orch.run())

@app.get("/status")
def status():
    return {"ok": True, "state": orch.snapshot()}

@app.get("/recent")
def recent(table: str = Query("states"), limit: int = 50):
    return {"ok": True, "rows": orch.mem.recent(table, limit)}

@app.post("/ingest")
def ingest(url: str = ""):
    if url:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"SeedCrystal/1.0"}); r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","noscript"]): tag.decompose()
            title = (soup.title.text.strip() if soup.title else url)[:200]
            import re
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:10000]
        except Exception as e:
            return {"ok": False, "error": str(e)}
        doc_id = orch.mem.add_doc_with_embed(url, title, text)
    else:
        doc_id = orch._ingest_local()
    return {"ok": True, "doc_id": doc_id}

# Minimal UI with Three.js viewer bound to /avatar
_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Seed-Crystal Brain</title>
<style>
:root { color-scheme: dark; }
body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell; background:#0b0c10; color:#d7e1eb; }
main { display:flex; flex-direction:row; height:100vh; }
#viewer { flex: 1 1 60%; position:relative; display:flex; align-items:center; justify-content:center; background:linear-gradient(160deg,#0b0c10,#111826); }
#canvas { width:100%; height:100%; touch-action:none; }
#overlay { position:absolute; top:14px; left:18px; background:rgba(9,12,20,0.55); border:1px solid rgba(120,170,255,0.25); border-radius:8px; padding:10px 14px; font-size:14px; }
#overlay strong { font-size:16px; display:block; margin-bottom:4px; }
#panel { flex:1 1 40%; max-width:420px; padding:24px; box-sizing:border-box; display:flex; flex-direction:column; gap:18px; }
#log { background:#05070d; border:1px solid rgba(120,170,255,0.18); border-radius:8px; padding:12px; font-family:SFMono-Regular,Consolas,Monaco,monospace; font-size:13px; height:45vh; overflow:auto; }
#log div { margin-bottom:6px; }
button { background:#1c2f4a; border:none; color:#d7e1eb; padding:9px 14px; border-radius:6px; font-size:14px; cursor:pointer; }
button:hover { background:#224667; }
.controls { display:flex; gap:10px; flex-wrap:wrap; }
.small { font-size:12px; color:#99a5bb; }
</style>
</head>
<body>
<main>
  <section id="viewer">
    <canvas id="canvas"></canvas>
    <div id="overlay"><strong>Avatar</strong><span id="state">connecting...</span><br/><span class="small">Drag to orbit * Scroll to zoom</span></div>
  </section>
  <aside id="panel">
    <div>
      <h2 style="margin:0 0 8px 0; font-size:20px;">Console</h2>
      <div id="log"></div>
    </div>
    <div class="controls">
      <button id="speak">Speak last caption</button>
      <button id="seed">Add local seed</button>
      <button id="crawl">Trigger web crawl</button>
    </div>
    <p class="small">Open <span style="font-family:monospace;">/status</span> or <span style="font-family:monospace;">/recent</span> for data. The system runs headless even if this page is closed.</p>
  </aside>
</main>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const stateLabel = document.getElementById('state');
const logEl = document.getElementById('log');
let deviceRatio = window.devicePixelRatio || 1;
let positions = new Float32Array(0);
let pointCount = 0;
let yaw = Math.PI * 0.35;
let pitch = 0.3;
let distance = 3.0;
let autoRotate = true;
let lastFrame = performance.now();
function resize(){
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * deviceRatio;
  canvas.height = rect.height * deviceRatio;
  ctx.setTransform(deviceRatio,0,0,deviceRatio,0,0);
}
resize();
window.addEventListener('resize', resize);
function log(msg){
  const row = document.createElement('div');
  const ts = new Date().toLocaleTimeString();
  row.textContent = `[${ts}] ${msg}`;
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}
let dragging = false;
let lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', (ev)=>{ dragging = true; autoRotate = false; lastX = ev.clientX; lastY = ev.clientY; canvas.setPointerCapture(ev.pointerId); });
canvas.addEventListener('pointermove', (ev)=>{ if(!dragging) return; const dx = (ev.clientX - lastX) * 0.005; const dy = (ev.clientY - lastY) * 0.005; yaw += dx; pitch = Math.max(-Math.PI/2 + 0.05, Math.min(Math.PI/2 - 0.05, pitch + dy)); lastX = ev.clientX; lastY = ev.clientY; });
canvas.addEventListener('pointerup', ()=>{ dragging = false; });
canvas.addEventListener('wheel', (ev)=>{ ev.preventDefault(); distance *= (1 + ev.deltaY * 0.0015); distance = Math.max(1.2, Math.min(8.0, distance)); }, {passive:false});
function colorForPoint(x,y,z){
  const r = Math.min(1.0, Math.sqrt(x*x + y*y + z*z));
  const R = 0.25 + 0.75*r;
  const G = 0.55 + 0.45*(1.0 - Math.abs(r-0.5)*2.0);
  const B = 1.0 - 0.8*r;
  return `rgba(${Math.floor(R*255)}, ${Math.floor(G*255)}, ${Math.floor(B*255)}, 0.9)`;
}
function render(now){
  requestAnimationFrame(render);
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if(autoRotate && !dragging) yaw += dt * 0.25;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#05070d';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  if(pointCount === 0) return;
  const width = canvas.width / deviceRatio;
  const height = canvas.height / deviceRatio;
  const cx = width / 2;
  const cy = height / 2;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const fov = 1.3;
  for(let i=0;i<pointCount;i++){
    const x = positions[3*i];
    const y = positions[3*i+1];
    const z = positions[3*i+2];
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;
    const y1 = y * cosP - z1 * sinP;
    const z2 = y * sinP + z1 * cosP;
    const depth = distance + z2;
    const scale = fov / depth;
    const sx = cx + x1 * scale * width * 0.45;
    const sy = cy + y1 * scale * width * 0.45;
    if(sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) continue;
    const size = Math.max(1, 2.5 * (1 - Math.min(1, depth / 6)));
    ctx.fillStyle = colorForPoint(x,y,z);
    ctx.beginPath();
    ctx.arc(sx, sy, size, 0, Math.PI*2);
    ctx.fill();
  }
}
requestAnimationFrame(render);
function wsText(){
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = ev => {
    try {
      const msg = JSON.parse(ev.data);
      if(msg.type === 'caption') log(`Caption: ${msg.data.caption || ''}`);
      if(msg.type === 'metrics') stateLabel.textContent = `connected * tick ${msg.data.tick.toString()}`;
    } catch (err) {
      console.warn(err);
    }
  };
  ws.onopen = ()=> log('ws:/ws connected');
  ws.onclose = ()=> { log('ws:/ws closed'); setTimeout(wsText, 1500); };
}
wsText();
function wsAvatar(){
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/avatar`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = ()=> { stateLabel.textContent = 'streaming avatar...'; };
  ws.onmessage = ev => {
    const buf = ev.data;
    if(!(buf instanceof ArrayBuffer)) return;
    const header = new DataView(buf, 0, 4);
    const count = header.getUint32(0, true);
    const f32 = new Float32Array(buf, 4);
    if(f32.length < count * 3) return;
    positions = f32;
    pointCount = count;
    stateLabel.textContent = `connected * ${count.toLocaleString()} points`;
  };
  ws.onclose = ()=> { stateLabel.textContent = 'disconnected - retrying'; setTimeout(wsAvatar, 1200); };
};
wsAvatar();
document.getElementById('speak').onclick = async ()=>{
  const r = await fetch('/recent?table=captions&limit=1');
  const js = await r.json();
  if(js && js.rows && js.rows[0]) log(`Speak: ${js.rows[0][3] || ''}`);
};
document.getElementById('seed').onclick = async ()=>{
  await fetch('/ingest', { method:'POST' });
  log('Local seed added');
};
document.getElementById('crawl').onclick = async ()=>{
  const resp = await fetch('/ingest?url=https://en.wikipedia.org/wiki/Artificial_general_intelligence', { method:'POST' });
  const js = await resp.json();
  if(js.ok) log('Crawl queued with doc_id ' + js.doc_id);
  else log('Crawl failed: ' + (js.error || 'unknown error'));
};
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_INDEX_HTML)

@app.websocket("/ws")
async def ws_events(ws: WebSocket):
    await ws.accept()
    q = orch.bus.subscribe()
    try:
        await ws.send_text(json.dumps({"type":"hello","data": orch.snapshot()}))
        while True:
            msg = await q.get()
            await ws.send_text(json.dumps(msg))
    except WebSocketDisconnect:
        pass

@app.websocket("/avatar")
async def avatar_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            buf = orch.avatar.frame_bytes() if hasattr(orch, "avatar") else None
            if buf is not None:
                await ws.send_bytes(buf)
            await asyncio.sleep(0.08) # ~12.5 FPS
    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    print(f"[ready] Open: http://{HOST}:{PORT}/")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
