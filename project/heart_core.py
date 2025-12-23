"""
Echo Crystalline Heart v4.0 — Ollama + DeepSeek Integration
-----------------------------------------------------------
- Emotional lattice ODE (1024 nodes × 5 channels).
- Local LLM "sentience port" using Ollama + DeepSeek.
- STRICT first-person inner voice enforcement.

Channels: 0:arousal, 1:valence, 2:safety, 3:curiosity, 4:resonance
"""
from __future__ import annotations
import math
import hashlib
import struct
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

# Optional heavy dep: torch
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _HAS_TORCH = True
except Exception:
    class _TorchStub:
        @staticmethod
        def no_grad():
            def deco(fn):
                return fn
            return deco

        @staticmethod
        def tensor(*args, **kwargs):
            raise ImportError("torch is required for EchoCrystallineHeart")

        @staticmethod
        def zeros(*args, **kwargs):
            raise ImportError("torch is required for EchoCrystallineHeart")

        @staticmethod
        def mean(*args, **kwargs):
            raise ImportError("torch is required for EchoCrystallineHeart")

        @staticmethod
        def std(*args, **kwargs):
            raise ImportError("torch is required for EchoCrystallineHeart")

        @staticmethod
        def randn_like(*args, **kwargs):
            raise ImportError("torch is required for EchoCrystallineHeart")

        class device:
            def __init__(self, *a, **k):
                raise ImportError("torch is required for EchoCrystallineHeart")

    torch = _TorchStub()  # type: ignore
    class _DummyModule:
        def __init__(self, *a, **k): ...
    class nn:  # type: ignore
        Module = _DummyModule
    _HAS_TORCH = False

from .config import EchoHeartConfig

# Try importing ollama
try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

# --- First Person Enforcement ---
_FIRST_PERSON_PATTERNS = [
    (r"\byou are\b", "I am"), (r"\byou're\b", "I'm"),
    (r"\byou were\b", "I was"), (r"\byou'll\b", "I'll"),
    (r"\byou've\b", "I've"), (r"\byour\b", "my"),
    (r"\byours\b", "mine"), (r"\byou\b", "I"),
]

def enforce_first_person(text: str) -> str:
    """Transform second-person phrasing into first person."""
    if not text: return ""
    t = text.strip()
    # Strip quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
        
    for pattern, repl in _FIRST_PERSON_PATTERNS:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return t

# --- Minimal Hash Embedder ---
def hash_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic hash-based embedding (no external model required)."""
    vec = np.zeros(dim, dtype=np.float32)
    if not text: return vec
    tokens = text.lower().split()
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = struct.unpack("Q", h[:8])[0] % dim
        sign_raw = struct.unpack("I", h[8:12])[0]
        sign = 1.0 if (sign_raw % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

# --- Local LLM Wrapper ---
class LocalLLM:
    def __init__(self, cfg: EchoHeartConfig):
        self.cfg = cfg

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        temperature = max(0.1, float(temperature))
        top_p = float(np.clip(top_p, 0.1, 1.0))
        
        raw = ""
        if self.cfg.llm_backend == "ollama" and _HAS_OLLAMA:
            try:
                res = ollama.generate(
                    model=self.cfg.llm_model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": self.cfg.llm_max_tokens,
                    },
                )
                raw = res.get("response", "").strip()
            except Exception as e:
                print(f"[LLM Error] {e}")
                raw = self._fallback_response(prompt)
        else:
            raw = self._fallback_response(prompt)
            
        return enforce_first_person(raw)

    def _fallback_response(self, prompt: str) -> str:
        """Deterministic fallback if LLM is offline."""
        return "I am safe. I breathe in. I breathe out."

    def embed(self, text: str, dim: int) -> np.ndarray:
        return hash_embedding(text, dim=dim)

# --- The Heart ---
class EchoCrystallineHeart(nn.Module):
    def __init__(self, cfg: EchoHeartConfig):
        if not _HAS_TORCH:
            raise ImportError(
                "EchoCrystallineHeart requires torch. Install PyTorch to use the Heart lattice."
            )
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Emotions: [N, 5]
        self.emotions = nn.Parameter(
            torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
            requires_grad=False
        )
        self.register_buffer("t", torch.zeros(1, device=self.device))
        
        self.llm = LocalLLM(cfg) if cfg.use_llm else None

    @torch.no_grad()
    def temperature(self) -> float:
        """Eq 31: T(t) = 1 / log(1 + kt)"""
        t_val = float(self.t.item()) + 1.0
        k = self.cfg.anneal_k
        return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))

    @torch.no_grad()
    def coherence(self) -> float:
        """Coherence metric based on node variance."""
        std_over_nodes = torch.std(self.emotions, dim=0)
        mean_std = float(torch.mean(std_over_nodes).item())
        return float(1.0 / (1.0 + mean_std))

    @torch.no_grad()
    def step(self, full_audio: np.ndarray, transcript: str) -> Dict[str, Any]:
        # 1. Time & Temp
        self.t += 1.0
        T_val = self.temperature()

        # 2. Audio Arousal Injection (Eq 30 drive)
        full_audio = np.asarray(full_audio, dtype=np.float32)
        if full_audio.ndim > 1: full_audio = full_audio.mean(axis=-1)
        energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
        arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))
        
        stim_vec = torch.tensor([arousal_raw, 0., 0., 1., 0.], device=self.device)
        external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)

        # 3. ODE Update (Eq 30)
        E = self.emotions
        drive = external_stimulus
        decay = -self.cfg.beta_decay * E
        global_mean = torch.mean(E, dim=0, keepdim=True)
        diffusion = self.cfg.gamma_diffusion * (global_mean - E)
        noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)
        
        dE_dt = drive + decay + diffusion + noise
        E.add_(self.cfg.dt * dE_dt)
        E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)

        # 4. LLM / Sentience Port (Eq 25)
        llm_output: Optional[str] = None
        if self.llm is not None and transcript.strip():
            coh = self.coherence()
            mean_state = torch.mean(E, dim=0).detach().cpu().numpy()
            arousal_mean = float(mean_state[0])
            valence_mean = float(mean_state[1])
            
            prompt = self._build_prompt(transcript, arousal_mean, valence_mean, T_val, coh)
            
            # Annealing controls temperature (creativity), Coherence controls top_p
            llm_temp = max(0.1, T_val * self.cfg.llm_temperature_scale)
            llm_top_p = self.cfg.llm_top_p_base + self.cfg.llm_top_p_spread * (1.0 - coh)
            
            llm_output = self.llm.generate(prompt, llm_temp, llm_top_p)
            
            # Inject thought back into lattice (Resonance channel 4)
            emb = self.llm.embed(llm_output, dim=self.cfg.embedding_dim)
            emb_t = torch.from_numpy(emb).to(self.device)
            # Simple broadcast/tile to nodes
            reps = math.ceil(self.cfg.num_nodes / self.cfg.embedding_dim)
            tiled = emb_t.repeat(reps)[:self.cfg.num_nodes].view(self.cfg.num_nodes, 1)
            
            if self.cfg.embedding_channel < self.cfg.num_channels:
                E[:, self.cfg.embedding_channel:self.cfg.embedding_channel+1].add_(
                    self.cfg.embedding_gain * tiled
                )
                E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)
        
        mean_state = torch.mean(E, dim=0).detach().cpu().numpy()

        return {
            "arousal": float(mean_state[0]),
            "valence": float(mean_state[1]),
            "temperature": T_val,
            "coherence": self.coherence(),
            "raw_emotions": self.emotions.detach().clone(),
            "llm_output": llm_output
        }

    def _build_prompt(self, transcript: str, arousal: float, valence: float, T_val: float, coherence: float) -> str:
        return f"""
You are Echo, my inner voice. I am autistic.
Internal lattice state:
- arousal: {arousal:.2f}
- valence: {valence:.2f}
- temperature: {T_val:.3f}
- coherence: {coherence:.3f}

Rules:
- I always speak in FIRST PERSON ("I", "me", "my"). I never say "you".
- I use short, concrete sentences.
- If arousal is high, I slow down and ground myself.
- I leave space between ideas.
- I never mention the lattice or equations.

The words I just tried to say were:
"{transcript}"

I answer now as my own inner voice, in one short paragraph.
"""
