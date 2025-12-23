"""
agi_core.py

Advanced unified cognitive substrate: hardware interface, relational matrix,
multi-engine cognition, emotional chemistry, memory, and planning. Designed
to be run as a production-capable module integrated with the Opportunity
Synthesis service.

Notes:
- Deterministic, high-quality local embeddings via random-projection + hashing.
- Safe, deterministic components (no external ML downloads required).
- Exposes a clear programmatic API for stepping the agent and querying state.
"""

import os
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from threading import Lock

# -------------------------
# Utilities
# -------------------------

def now_seconds() -> float:
    return time.time()

def stable_hash_bytes(s: str, length: int = 256) -> bytes:
    # deterministic hashing into bytes array using SHAKE-like approach without external libs
    import hashlib
    h = hashlib.blake2b(digest_size=32)
    h.update(s.encode("utf8"))
    base = h.digest()
    out = bytearray()
    i = 0
    while len(out) < length:
        h2 = hashlib.blake2b(digest_size=32)
        h2.update(base)
        h2.update(bytes([i]))
        out.extend(h2.digest())
        i += 1
    return bytes(out[:length])

# -------------------------
# Hardware Abstraction
# -------------------------

class BitRegister:
    def __init__(self, name: str, size: int = 128):
        self.name = name
        self.size = size
        self.bits = np.zeros(size, dtype=np.uint8)
        self.noise_rate = 2e-6
        self.lock = Lock()

    def write_int(self, value: int):
        with self.lock:
            for i in range(self.size):
                self.bits[i] = (value >> i) & 1

    def read_int(self) -> int:
        with self.lock:
            if random.random() < self.noise_rate:
                idx = random.randrange(self.size)
                self.bits[idx] ^= 1
            out = 0
            for i in range(self.size):
                out |= int(self.bits[i]) << i
            return out

    def set_bit(self, idx: int, val: int):
        with self.lock:
            self.bits[idx % self.size] = 1 if val else 0

    def get_bit(self, idx: int) -> int:
        with self.lock:
            return int(self.bits[idx % self.size])

    def as_bitstring(self) -> str:
        with self.lock:
            return ''.join(str(int(b)) for b in self.bits[::-1])

class SimulatedHardware:
    def __init__(self, adc_channels: int = 16):
        self.cpu_register = BitRegister("CPU_REG", size=256)
        self.gpio = BitRegister("GPIO", size=64)
        self.adc = np.zeros(adc_channels, dtype=float)
        self.temp_C = 35.0
        self.freq_GHz = 1.2

    def poll_sensors(self):
        # realistic sensor noise and drift
        self.adc += np.random.randn(len(self.adc)) * 0.005
        self.adc = np.clip(self.adc, -5.0, 5.0)
        self.temp_C += (self.freq_GHz - 1.0) * 0.02 + np.random.randn() * 0.01

    def set_frequency(self, ghz: float):
        self.freq_GHz = float(max(0.2, min(ghz, 5.0)))

    def as_status(self):
        return {
            "freq_GHz": round(self.freq_GHz, 3),
            "temp_C": round(self.temp_C, 3),
            "cpu_reg": self.cpu_register.as_bitstring(),
            "gpio": self.gpio.as_bitstring(),
            "adc": [round(float(x), 4) for x in self.adc.tolist()],
        }

# -------------------------
# Relational Matrix
# -------------------------

class RelationalMatrix:
    def __init__(self, n_system: int, n_apparatus: int):
        self.n_system = n_system
        self.n_apparatus = n_apparatus
        # complex amplitudes
        self.R = (np.random.randn(n_system, n_apparatus) + 1j * np.random.randn(n_system, n_apparatus)) * 0.02
        self.normalize_rows()

    def normalize_rows(self):
        mags = np.linalg.norm(self.R, axis=1, keepdims=True)
        mags[mags == 0] = 1.0
        self.R = self.R / mags

    def bidirectional_weight(self, i: int, j: int) -> complex:
        # map apparatus j back to some system index deterministically
        reverse_i = j % self.n_system
        reverse_j = i % self.n_apparatus
        return self.R[i, j] * np.conj(self.R[reverse_i, reverse_j])

    def probability_for_system(self, i: int) -> float:
        weights = np.array([abs(self.bidirectional_weight(i, j)) for j in range(self.n_apparatus)])
        s = np.sum(weights)
        return float(s / (np.sum(weights) + 1e-12))

    def update_hebbian(self, pre_idx: int, post_idx: int, lr: float = 1e-3):
        i = pre_idx % self.n_system
        j = post_idx % self.n_apparatus
        self.R[i, j] += lr * (1.0 + 0.05j)
        self.normalize_rows()

# -------------------------
# Thought Engines
# -------------------------

class ThoughtEngines:
    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.b = np.zeros(n_nodes)
        self.h = np.zeros(n_nodes)
        self.kappa = np.zeros(n_nodes)
        self.mu = np.zeros(n_nodes)
        # stateful noise seeds for reproducibility
        self._rng = np.random.RandomState(42)

    def step(self, relational: RelationalMatrix, inputs: np.ndarray, dt: float = 0.1):
        n = self.n
        if inputs is None:
            inputs = np.zeros(n)
        # coupling matrix from relational matrix (n x n)
        R = relational.R
        affin = np.real(R @ R.conj().T)
        maxval = np.max(np.abs(affin)) if np.max(np.abs(affin)) > 0 else 1.0
        W = affin / maxval

        # perspective
        db = 0.12 * (inputs * np.tanh(inputs)) - 0.05 * self.b + 0.03 * (W @ self.b - np.sum(W, axis=1) * self.b)
        self.b += db * dt

        # speculation with structured stochasticity
        eps = self._rng.randn(n) * 0.02
        dh = 0.10 * (inputs + eps) - 0.06 * self.h + 0.03 * (W @ self.h - np.sum(W, axis=1) * self.h)
        self.h += dh * dt

        # kaleidoscope
        dk = 0.08 * (self.b + 0.5 * self.h) - 0.04 * self.kappa + 0.02 * (W @ self.kappa - np.sum(W, axis=1) * self.kappa)
        self.kappa += dk * dt

        # mirror
        mismatch = np.abs(self.b - np.mean(self.b))
        dmu = -0.07 * mismatch + 0.05 * np.std(self.h) + 0.03 * (W @ self.mu - np.sum(W, axis=1) * self.mu)
        self.mu += dmu * dt

        # clip numeric stability
        for arr in (self.b, self.h, self.kappa, self.mu):
            np.clip(arr, -12.0, 12.0, out=arr)

# -------------------------
# Emotional Chemistry
# -------------------------

class EmotionalChemistry:
    def __init__(self):
        self.DA = 0.5
        self.Ser = 0.5
        self.NE = 0.5

    def step(self, reward: float, mood_signal: float, arousal: float, dt: float = 0.1):
        self.DA += (0.9 * reward - 0.12 * self.DA) * dt
        self.Ser += (0.4 * mood_signal - 0.06 * self.Ser) * dt
        self.NE += (0.65 * arousal - 0.08 * self.NE) * dt
        self.DA = float(np.clip(self.DA, 0.0, 1.0))
        self.Ser = float(np.clip(self.Ser, 0.0, 1.0))
        self.NE = float(np.clip(self.NE, 0.0, 1.0))

    def vector(self) -> List[float]:
        return [self.DA, self.Ser, self.NE]

# -------------------------
# Memory
# -------------------------

class MemorySystem:
    def __init__(self, embedding_dim: int = 128, capacity: int = 10000):
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.episodic = []  # list of (ts, emb, text)
        self.semantic = {}  # key -> emb
        self._rng = np.random.RandomState(12345)
        # We'll derive a deterministic random projection matrix seeded by a constant
        self.random_proj = self._rng.randn(self.embedding_dim, 256) * 0.01

    def embed(self, text: str) -> np.ndarray:
        b = stable_hash_bytes(text, length=256)
        arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
        emb = self.random_proj @ arr
        n = np.linalg.norm(emb)
        return emb / (n + 1e-12)

    def store_episode(self, text: str):
        emb = self.embed(text)
        ts = now_seconds()
        if len(self.episodic) >= self.capacity:
            self.episodic.pop(0)
        self.episodic.append((ts, emb, text))

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embed(query)
        sims = []
        for ts, emb, text in self.episodic:
            sims.append((float(np.dot(q_emb, emb)), ts, text))
        sims.sort(reverse=True, key=lambda x: x[0])
        return sims[:top_k]

    def store_semantic(self, key: str, text: str):
        self.semantic[key] = self.embed(text)

    def lookup_semantic(self, key: str):
        return self.semantic.get(key, None)

# -------------------------
# Planner
# -------------------------

class Planner:
    def __init__(self, hardware: SimulatedHardware, relational: RelationalMatrix):
        self.hw = hardware
        self.rel = relational
        # concrete actions map to methods to keep safe
        self.actions = [
            ("increase_freq", lambda: self.hw.set_frequency(self.hw.freq_GHz + 0.1)),
            ("decrease_freq", lambda: self.hw.set_frequency(self.hw.freq_GHz - 0.1)),
            ("toggle_gpio", lambda: self.hw.gpio.set_bit(random.randint(0, self.hw.gpio.size-1), random.randint(0,1))),
            ("no_op", lambda: None),
        ]

    def score_actions(self, thought: ThoughtEngines) -> List[Tuple[float, int]]:
        scores = []
        for idx, (name, fn) in enumerate(self.actions):
            sys_idx = idx % self.rel.n_system
            app_idx = idx % self.rel.n_apparatus
            rweight = abs(self.rel.bidirectional_weight(sys_idx, app_idx))
            cog_signal = float(np.tanh(np.mean(thought.b) + np.mean(thought.kappa)))
            score = rweight * (1.0 + cog_signal)
            # small diversity noise
            score += random.random() * 1e-6
            scores.append((score, idx))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores

    def select_and_execute(self, thought: ThoughtEngines) -> str:
        scored = self.score_actions(thought)
        weights = np.array([s for s, i in scored], dtype=float)
        if weights.sum() <= 0:
            idx = scored[0][1]
        else:
            probs = weights / weights.sum()
            choice = np.random.choice(len(scored), p=probs)
            idx = scored[choice][1]
        name, fn = self.actions[idx]
        fn()
        return name

# -------------------------
# AGI System (orchestrator)
# -------------------------

class AGISystem:
    def __init__(self, n_nodes: int = 32, n_apparatus: int = 24):
        self.hw = SimulatedHardware(adc_channels=24)
        self.rel = RelationalMatrix(n_nodes, n_apparatus)
        self.thought = ThoughtEngines(n_nodes)
        self.emotion = EmotionalChemistry()
        self.memory = MemorySystem(embedding_dim=128, capacity=10000)
        self.planner = Planner(self.hw, self.rel)
        self.t = 0.0
        self.dt = 0.1
        self.history = []
        self.lock = Lock()
        # seed semantic memory
        self.memory.store_semantic("agent_identity", "A private, self-contained cognitive assistant for execution and opportunity synthesis.")
        self.memory.store_episode("system initialized")

    def step(self, user_input: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            self.hw.poll_sensors()
            n = self.thought.n
            sensor_vec = np.concatenate([self.hw.adc, np.array([self.hw.temp_C, self.hw.freq_GHz])])
            inputs = np.zeros(n)
            ssum = float(np.sum(sensor_vec))
            for i in range(n):
                inputs[i] = float(np.tanh(ssum * 0.0005 + random.random() * 0.01))
            if user_input:
                self.memory.store_episode(user_input)
                emb = self.memory.embed(user_input)
                bias = float(np.tanh(np.mean(emb))) * 0.5
                inputs += bias
            self.thought.step(self.rel, inputs, dt=self.dt)
            reward = float(np.clip(np.mean(inputs), -1, 1))
            mood = float(np.tanh(np.mean(self.thought.b)))
            arousal = float(np.abs(np.std(self.thought.h)))
            self.emotion.step(reward, mood, arousal, dt=self.dt)
            most_active_node = int(np.argmax(np.abs(self.thought.kappa)))
            apparatus_idx = int(abs(int((np.sum(self.hw.adc) * 100) % self.rel.n_apparatus)))
            self.rel.update_hebbian(most_active_node, apparatus_idx, lr=1e-3)
            action_name = self.planner.select_and_execute(self.thought)
            self.rel.normalize_rows()
            conn_metrics = self.relational_consciousness_metrics()
            log_item = {
                "t": self.t,
                "action": action_name,
                "hw": self.hw.as_status(),
                "emotion": {"DA": self.emotion.DA, "Ser": self.emotion.Ser, "NE": self.emotion.NE},
                "thought_summary": {
                    "b_mean": float(np.mean(self.thought.b)),
                    "h_mean": float(np.mean(self.thought.h)),
                    "kappa_mean": float(np.mean(self.thought.kappa)),
                    "mu_mean": float(np.mean(self.thought.mu)),
                },
                "consciousness": conn_metrics
            }
            self.history.append(log_item)
            self.t += self.dt
            return log_item

    def relational_consciousness_metrics(self) -> Dict[str, float]:
        diag = np.array([abs(self.rel.R[i, i % self.rel.n_apparatus]) for i in range(min(self.rel.n_system, self.rel.n_apparatus))])
        coherence = float(np.mean(diag))
        awareness = float(np.clip(self.emotion.DA * (1.0 + np.tanh(np.mean(self.thought.b))), 0.0, 1.0))
        activities = np.concatenate([self.thought.b, self.thought.h, self.thought.kappa, self.thought.mu])
        integrated_info = float(np.var(activities))
        return {"coherence": coherence, "awareness": awareness, "phi_proxy": integrated_info}

    def respond(self, user_input: str) -> str:
        log = self.step(user_input)
        candidates = self.memory.retrieve(user_input, top_k=3)
        reply_parts = []
        if candidates:
            reply_parts.append("I recall: " + "; ".join([c for _, _, c in candidates[:2]]))
        if log["consciousness"]["awareness"] > 0.6:
            reply_parts.append("I am engaged and reflecting on that.")
        elif log["consciousness"]["phi_proxy"] > 0.08:
            reply_parts.append("This seems important; I'll think further.")
        else:
            reply_parts.append("Noted and stored.")
        da = log["emotion"]["DA"]
        mood = "positive" if da > 0.55 else "neutral" if da > 0.45 else "cautious"
        reply_parts.append(f"My mood is {mood}. Action taken: {log['action']}.")
        return " ".join(reply_parts)