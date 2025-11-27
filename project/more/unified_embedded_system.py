"""
unified_embedded_system.py

Self-contained Unified Cognitive System with embedded components:
- Local (offline) Autonomous Crawler operating over a local corpus directory
- Embedded lightweight LLM-like reflection module (small PyTorch Transformer)
- Local embedding using TF-IDF + PCA (no external model calls)
- Organic AI core integrating emotion, crystalline memory, and node dynamics
- Quart-based API + WebSocket broadcasts for front-end visualization

This file is designed to run fully offline. No external network calls are made.
Place local documents in ./corpus/ as .txt files to enable crawling. If none exist, sample docs will be generated.
"""

import os
import asyncio
import logging
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque, defaultdict
from datetime import datetime

# Lightweight ML stack (standard libs)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from quart import Quart, websocket, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Utilities
# -----------------------------

def ensure_corpus(path='./corpus'):
    os.makedirs(path, exist_ok=True)
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    if not files:
        # create small sample docs
        samples = {
            'ai.txt': 'Artificial intelligence studies algorithms that learn from data.',
            'cognition.txt': 'Cognition explores perception, attention, memory, and reasoning.',
            'agent.txt': 'An agent observes, decides and acts in an environment to achieve goals.'
        }
        for name, text in samples.items():
            with open(os.path.join(path, name), 'w') as fh:
                fh.write(text)
        logging.info('Sample corpus created in ./corpus')

# -----------------------------
# Local Embedder (TF-IDF + PCA)
# -----------------------------
class LocalEmbedder:
    def __init__(self, n_components=64):
        self.vectorizer = TfidfVectorizer(max_features=1024)
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, texts: List[str]):
        X = self.vectorizer.fit_transform(texts).toarray()
        if X.shape[1] < self.pca.n_components:
            # pad columns
            pad = np.zeros((X.shape[0], self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        self.pca.fit(X)
        self.fitted = True
        logging.info('LocalEmbedder fitted on corpus')

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            # fall back to hashing-based embedding
            h = np.frombuffer(hashlib_sha256(text.encode()).digest()[:64], dtype=np.uint8).astype(np.float32)
            return (h - h.mean()) / (h.std() + 1e-9)
        X = self.vectorizer.transform([text]).toarray()
        if X.shape[1] < self.pca.n_components:
            pad = np.zeros((1, self.pca.n_components - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        emb = self.pca.transform(X)[0]
        # normalize
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm

# small helper sha256
import hashlib

def hashlib_sha256(b: bytes):
    return hashlib.sha256(b)

# -----------------------------
# Embedded Mini-LLM (small Transformer)
# -----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        # x: LongTensor (batch, seq)
        e = self.embed(x) * math.sqrt(self.d_model)
        out = self.encoder(e)
        logits = self.head(out)
        return logits

class EmbeddedLLM:
    """A tiny deterministic 'reflection' model. It is not a true LLM but provides consistent, local reflections.
    Uses a byte-level tokenizer and a small transformer. Runs entirely offline.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = MiniTransformer(vocab_size=256, d_model=128, nhead=4, num_layers=2).to(self.device)
        # initialize with deterministic seed
        torch.manual_seed(0)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model.eval()

    def _tokenize(self, text: str, max_len=128):
        b = text.encode('utf-8', errors='ignore')[:max_len]
        ids = [c for c in b]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def reflect(self, prompt: str, max_len=128) -> str:
        """Generate a short deterministic reflection text based on prompt"""
        tok = self._tokenize(prompt, max_len=max_len)
        with torch.no_grad():
            logits = self.model(tok)
            # take mean over sequence and sample top tokens deterministically
            avg = logits.mean(dim=1).squeeze(0)
            topk = torch.topk(avg, k=16).indices.cpu().numpy().tolist()
            # convert bytes back to readable text by mapping to chars
            chars = ''.join(chr((t % 94) + 32) for t in topk)
            # craft a concise reflection
            reflection = f"Reflection: {chars}\nSummary: {prompt[:120]}"
            return reflection

# -----------------------------
# Emotional Crystalline Memory (lightweight)
# -----------------------------
class CrystallineMemory:
    def __init__(self, dim=64, capacity=1024):
        self.capacity = capacity
        self.dim = dim
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.meta = []

    def add(self, emb: np.ndarray, meta: Dict[str, Any]):
        if emb.shape[0] != self.dim:
            # project or pad
            if emb.shape[0] > self.dim:
                emb = emb[:self.dim]
            else:
                pad = np.zeros(self.dim - emb.shape[0], dtype=np.float32)
                emb = np.concatenate([emb, pad])
        emb = emb.astype(np.float32)
        if self.embeddings.shape[0] >= self.capacity:
            # FIFO
            self.embeddings = np.roll(self.embeddings, -1, axis=0)
            self.embeddings[-1] = emb
            self.meta = self.meta[1:] + [meta]
        else:
            self.embeddings = np.vstack([self.embeddings, emb]) if self.embeddings.size else emb.reshape(1, -1)
            self.meta.append(meta)

    def query(self, emb: np.ndarray, k=5):
        if self.embeddings.size == 0:
            return []
        # cosine similarity
        dots = np.dot(self.embeddings, emb)
        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(emb) + 1e-9)
        sims = dots / (norms + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(self.meta[i], float(sims[i])) for i in idx]

# -----------------------------
# Local Autonomous Crawler (offline) - uses local corpus files
# -----------------------------
@dataclass
class Document:
    url: str
    text: str
    embedding: Optional[np.ndarray] = None
    links: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    score: float = 0.0

class LocalCrawler:
    def __init__(self, ai_core, corpus_dir='./corpus', embedder: Optional[LocalEmbedder]=None):
        self.ai = ai_core
        self.corpus_dir = corpus_dir
        self.queue = deque()
        self.visited = set()
        self.embedder = embedder or LocalEmbedder(n_components=64)
        self.running = False
        ensure_corpus(self.corpus_dir)
        self._load_corpus()

    def _load_corpus(self):
        files = [os.path.join(self.corpus_dir, f) for f in os.listdir(self.corpus_dir) if f.endswith('.txt')]
        texts = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fh:
                texts.append(fh.read())
        if texts:
            self.embedder.fit(texts)
        # seed queue with file paths as 'urls'
        for f in files:
            self.queue.append('file://' + os.path.abspath(f))
        logging.info(f'LocalCrawler loaded {len(files)} documents')

    def _read_doc(self, url: str) -> Optional[Document]:
        if not url.startswith('file://'):
            return None
        path = url[len('file://'):]
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        return Document(url=url, text=text)

    async def crawl_once(self, max_docs=5):
        processed = 0
        while self.queue and processed < max_docs:
            url = self.queue.popleft()
            if url in self.visited:
                continue
            doc = self._read_doc(url)
            if not doc:
                continue
            emb = self.embedder.embed(doc.text)
            doc.embedding = emb
            doc.score = float(np.random.rand())
            self.visited.add(url)
            # integrate into AI
            self.ai.memory.add(emb, {'url': doc.url, 'ts': doc.timestamp})
            # notify AI core
            await self.ai.on_new_document(doc)
            processed += 1
        return processed

    async def start(self, interval=5.0):
        self.running = True
        logging.info('LocalCrawler started')
        while self.running:
            try:
                n = await self.crawl_once(max_docs=3)
                if n == 0:
                    await asyncio.sleep(interval)
                else:
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f'Crawler error: {e}')
                await asyncio.sleep(2.0)

    def stop(self):
        self.running = False
        logging.info('LocalCrawler stopped')

# -----------------------------
# Organic Core that integrates components
# -----------------------------
class OrganicCore:
    def __init__(self, device='cpu'):
        self.device = device
        self.embedder = LocalEmbedder(n_components=64)
        self.llm = EmbeddedLLM(device=device)
        self.memory = CrystallineMemory(dim=64, capacity=2048)
        self.nodes = {}
        self.metrics = {'energy_efficiency': 1.0, 'coherence': 0.5}
        self.knowledge_points = []
        self.listeners = []  # websocket listeners
        # initialize simple nodes
        for i in range(16):
            self.nodes[i] = {'awareness': random.random(), 'energy': random.random(), 'pos': np.random.rand(3).tolist()}

    async def on_new_document(self, doc: Document):
        # Query nearest memories
        q = self.memory.query(doc.embedding, k=3)
        # Compose a local context
        context_texts = ' | '.join([m.get('url','') for m, s in q])
        prompt = f"New document ingested: {os.path.basename(doc.url)}\nContext: {context_texts}\nContentSnippet: {doc.text[:200]}"
        reflection = self.llm.reflect(prompt)
        # update memory and knowledge points
        self.memory.add(doc.embedding, {'url': doc.url, 'snippet': doc.text[:200], 'reflection': reflection})
        self.knowledge_points.append({'url': doc.url, 'coords': doc.embedding[:3].tolist(), 'score': doc.score})
        # adjust nodes slightly based on doc
        for n in self.nodes.values():
            n['awareness'] = min(1.0, n['awareness'] + 0.01 * doc.score)
            n['energy'] = max(0.0, n['energy'] - 0.005 * doc.score)
        # broadcast to listeners
        await self.broadcast({
            'type': 'ingest',
            'url': doc.url,
            'reflection': reflection,
            'score': doc.score,
            'timestamp': doc.timestamp
        })

    async def broadcast(self, message: Dict[str, Any]):
        # send to websocket listeners
        to_remove = []
        for ws in list(self.listeners):
            try:
                await ws.send(json.dumps(message))
            except Exception:
                to_remove.append(ws)
        for r in to_remove:
            try: self.listeners.remove(r)
            except: pass

    def register_ws(self, ws):
        self.listeners.append(ws)

# -----------------------------
# API Server and Orchestration
# -----------------------------
app = Quart(__name__)
core = OrganicCore()
crawler = LocalCrawler(core, corpus_dir='./corpus', embedder=core.embedder)

# Background task holder
bg_tasks = []

@app.before_serving
async def startup():
    # no external calls; start idle
    logging.info('Unified embedded system starting up')

@app.route('/api/status')
async def status():
    return jsonify({
        'state': 'idle',
        'nodes': len(core.nodes),
        'memory_size': len(core.memory.meta),
        'knowledge_points': len(core.knowledge_points)
    })

@app.route('/api/start_crawl', methods=['POST'])
async def start_crawl():
    # start crawler in background
    task = asyncio.create_task(crawler.start(interval=2.0))
    bg_tasks.append(task)
    return jsonify({'status': 'crawler_started'})

@app.route('/api/stop_crawl', methods=['POST'])
async def stop_crawl():
    crawler.stop()
    # cancel background tasks
    for t in list(bg_tasks):
        t.cancel()
    return jsonify({'status': 'crawler_stopped'})

@app.websocket('/ws/updates')
async def ws_updates():
    ws = websocket._get_current_object()
    core.register_ws(ws)
    try:
        while True:
            msg = await ws.receive()
            # simple echo or control
            if msg == 'ping':
                await ws.send(json.dumps({'type': 'pong'}))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.info(f'WS closed: {e}')

@app.route('/api/ingest_local', methods=['POST'])
async def ingest_local():
    data = await request.get_json()
    path = data.get('path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'path not found'}), 400
    url = 'file://' + os.path.abspath(path)
    # add to crawler queue
    crawler.queue.appendleft(url)
    return jsonify({'status': 'queued', 'url': url})

@app.route('/api/query_reflect', methods=['POST'])
async def query_reflect():
    data = await request.get_json()
    text = data.get('text', '')
    reflection = core.llm.reflect(text)
    return jsonify({'reflection': reflection})

# Utility to run the server and an autonomous loop if desired
async def autonomous_loop():
    while True:
        try:
            # perform small maintenance: run one crawl step and sleep
            await crawler.crawl_once(max_docs=1)
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f'Autonomous loop error: {e}')
            await asyncio.sleep(2.0)

if __name__ == '__main__':
    # run Quart with asyncio event loop
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--autonomous', action='store_true')
    args = parser.parse_args()

    # ensure corpus exists and embedder is fitted
    ensure_corpus('./corpus')
    # fit embedder on corpus
    texts = []
    for f in os.listdir('./corpus'):
        if f.endswith('.txt'):
            with open(os.path.join('./corpus', f), 'r', encoding='utf-8') as fh:
                texts.append(fh.read())
    if texts:
        core.embedder.fit(texts)

    loop = asyncio.get_event_loop()
    if args.autonomous:
        loop.create_task(autonomous_loop())
    # start Quart
    app.run(host=args.host, port=args.port)
