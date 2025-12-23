# crystal_brain/core.py
from __future__ import annotations
import hashlib
import numpy as np
from typing import Dict, Any
from events import EchoEvent, BrainMetrics, now_ts
from .store import MemoryStore
from . import math as brain_math

class CrystalBrain:
    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim
        self.store = MemoryStore()
        self._last_caption: str = ""
        self._rng = np.random.default_rng()

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding for offline operation."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        reps = (self.embedding_dim + len(raw) - 1) // len(raw)
        tiled = np.tile(raw, reps)[:self.embedding_dim]
        norm = tiled / (np.linalg.norm(tiled) + 1e-9)
        return norm.astype("float32")

    def log_echo_event(self, event: EchoEvent) -> None:
        emb = self._hash_embed(event.text_clean)
        tags: Dict[str, Any] = {
            "duration_s": event.duration_s,
            "lang": event.lang,
            "meta": event.meta,
        }
        self.store.store_memory(event.text_clean, emb, event.timestamp, tags)

    def anneal_and_measure(self) -> BrainMetrics:
        embs = self.store.load_recent_embeddings()
        if embs.shape[0] == 0:
            embs = np.zeros((1, self.embedding_dim), dtype="float32")
            
        bits = (embs > 0.0).astype(np.int8)
        H_bits = brain_math.information_energy(bits)
        S_field = brain_math.field_stability(embs)
        L = brain_math.lyapunov_loss(H_bits, S_field)
        coherence = brain_math.coherence_metric(embs)
        phi = brain_math.integrated_information(embs)
        
        t = now_ts()
        self.store.store_energetics(t, H_bits, S_field, L, coherence, phi)
        
        return BrainMetrics(
            timestamp=t,
            H_bits=H_bits,
            S_field=S_field,
            L=L,
            coherence=coherence,
            phi=phi,
        )

    def generate_caption(self) -> str:
        rows = self.store.get_recent_energetics(limit=4)
        if not rows:
            self._last_caption = "I feel quiet and steady."
            return self._last_caption
            
        latest = rows[0]
        _, H_bits, S_field, L, coherence, phi = latest
        
        parts = []
        if H_bits < 0.4:
            parts.append("my thoughts are very focused")
        elif H_bits < 0.8:
            parts.append("my thoughts are balanced")
        else:
            parts.append("my thoughts are busy")

        if coherence > 0.7:
            parts.append("and everything fits together well")
        elif coherence < 0.3:
            parts.append("and things feel scattered")
        else:
            parts.append("and I am still sorting things out")

        if phi > 0.6:
            parts.append("with a strong sense of connection")
        else:
            parts.append("with a softer sense of connection")

        self._last_caption = "I feel like " + ", ".join(parts) + "."
        return self._last_caption

    def log_voice_profile(self, clarity: float, stability: float, t: float | None = None) -> int:
        """
        Persist a single point on the voice growth curve.
        clarity: how close raw vs normalized text are (0..1)
        stability: macro-emotional stability derived from Heart metrics (0..1)
        """
        ts = t if t is not None else now_ts()
        return self.store.store_voice_profile(ts, clarity, stability)
