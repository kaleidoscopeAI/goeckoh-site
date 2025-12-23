# ECHO_V4_UNIFIED/echo_core/heart.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from events import EchoEvent, HeartMetrics, now_ts

# --- Configuration (would normally be in a config file) ---
@dataclass(slots=True)
class HeartConfig:
    n_nodes: int = 1024
    n_channels: int = 5  # e.g., arousal, valence, stress, harmony, energy
    dt: float = 0.03  # Timestep for ODE integration
    alpha: float = 0.1  # Stimulus drive strength
    beta: float = 0.05  # State decay rate
    gamma: float = 0.02  # Diffusion/coupling strength
    noise_level: float = 0.005 # Base noise level
    max_emotion_value: float = 2.0

HEART_CONFIG = HeartConfig()

# --- CrystallineHeart Implementation ---

class CrystallineHeart:
    """
    Implements the Crystalline Heart, a 1024-node ODE lattice that provides
    a continuous, time-evolving model of the system's internal affective state.
    """
    def __init__(self, config: HeartConfig = HEART_CONFIG) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed=42)
        # Initialize the lattice state: [nodes, channels]
        self.lattice_state = self.rng.uniform(
            -0.1, 0.1, (config.n_nodes, config.n_channels)
        ).astype(np.float32)
        self.temperature: float = 1.0 # Annealing temperature

    def _stimulus_from_event(self, event: EchoEvent) -> np.ndarray:
        """Create a stimulus vector from an EchoEvent."""
        # Simple mapping for now: audio energy affects 'arousal' and 'energy' channels
        stimulus = np.zeros(self.config.n_channels, dtype=np.float32)
        
        # Channel 0: arousal, Channel 4: energy
        # Let's assume event.meta['energy'] is a normalized audio energy
        audio_energy = event.meta.get("energy", 0.5)
        stimulus[0] = audio_energy * 0.5 
        stimulus[4] = audio_energy * 0.3

        # Let's say text length affects stress (channel 2)
        length_factor = np.clip(len(event.text_clean) / 100.0, 0, 2.0)
        stimulus[2] = length_factor * 0.1

        return stimulus

    def _update_emotion_field(self, stimulus: np.ndarray) -> None:
        """
        Performs one vectorized Euler step of the ODE for the entire lattice.
        dE/dt = alpha*I - beta*E + gamma*(mean(E) - E) + noise
        """
        E = self.lattice_state

        # Calculate terms of the ODE
        drive = self.config.alpha * stimulus[np.newaxis, :]  # Apply stimulus to all nodes
        decay = -self.config.beta * E
        
        # Diffusion term (fully-connected field)
        mean_state = np.mean(E, axis=0, keepdims=True)
        diffusion = self.config.gamma * (mean_state - E)
        
        # Stochastic noise, scaled by temperature
        noise = self.rng.normal(
            0, 
            self.config.noise_level * self.temperature, 
            E.shape
        ).astype(np.float32)

        # Update the state
        dE = drive + decay + diffusion + noise
        self.lattice_state += self.config.dt * dE

        # Clip to prevent runaway values
        np.clip(
            self.lattice_state,
            -self.config.max_emotion_value,
            self.config.max_emotion_value,
            out=self.lattice_state
        )

    def _anneal(self) -> None:
        """Slowly cool the system by reducing temperature."""
        # Simple decay for now; can be replaced with the 1/log(t) schedule later
        self.temperature = max(0.1, self.temperature * 0.995)

    def _calculate_global_metrics(self) -> HeartMetrics:
        """Aggregate the 1024-node state into a single HeartMetrics snapshot."""
        E = self.lattice_state
        
        # Aggregate each channel
        # Assuming channel indices: 0:arousal, 1:valence, 2:stress, 3:harmony, 4:energy
        stress = float(np.mean(np.abs(E[:, 2])))
        harmony = 1.0 / (1.0 + float(np.mean(np.std(E, axis=0)))) # GCL-like metric
        energy = float(np.mean(E[:, 4]))
        
        # Confidence as inverse of overall variance
        confidence = 1.0 / (1.0 + float(np.var(E)))
        
        return HeartMetrics(
            timestamp=now_ts(),
            stress=np.clip(stress, 0, 1),
            harmony=np.clip(harmony, 0, 1),
            energy=np.clip(energy, 0, 2),
            confidence=np.clip(confidence, 0, 1),
            temperature=self.temperature
        )

    def update_from_event(self, event: EchoEvent) -> HeartMetrics:
        """
        The main public method. Updates the heart state based on a new
        utterance and returns the new global metrics.
        """
        stimulus = self._stimulus_from_event(event)
        self._update_emotion_field(stimulus)
        self._anneal()
        return self._calculate_global_metrics()
        from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from core.settings import HeartSettings


class EchoCrystallineHeart(nn.Module):
    """
    Emotional lattice + LLM integration.
    """

    def __init__(self, cfg: HeartSettings):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Emotions tensor: [num_nodes, num_channels]
        self.emotions = nn.Parameter(
            torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
            requires_grad=False,
        )
        # Time (discrete steps)
        self.register_buffer("t", torch.zeros(1, device=self.device))

    @torch.no_grad()
    def reset(self):
        self.emotions.zero_()
        self.t.zero_()

    @torch.no_grad()
    def temperature(self) -> float:
        """
        T(t) = 1 / log(1 + k t) (eq 31 style)
        """
        t_val = float(self.t.item()) + 1.0  # avoid log(0)
        k = self.cfg.anneal_k
        return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))

    @torch.no_grad()
    def coherence(self) -> float:
        """
        Simple coherence metric in [0,1]:
        - 1 = all nodes identical
        - 0 = highly scattered
        Implemented as:
        coherence = 1 / (1 + mean_std)
        """
        # [N, C]
        E = self.emotions
        # std over nodes, then mean over channels
        std_over_nodes = torch.std(E, dim=0)
        mean_std = float(torch.mean(std_over_nodes).item())
        return float(1.0 / (1.0 + mean_std))

    @torch.no_grad()
    def step(self, full_audio: np.ndarray) -> dict:
        """
        One full emotional update after a completed utterance.
        """
        # ---- 1. Update time + temperature --------------------------------
        self.t += 1.0
        T_val = self.temperature()
        # ---- 2. Extract arousal from waveform ----------------------------
        full_audio = np.asarray(full_audio, dtype=np.float32)
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=-1)
        # RMS energy
        energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
        arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))
        # external stimulus vector: [arousal, 0, 0, 1, 0]
        stim_vec = torch.tensor(
            [arousal_raw, 0.0, 0.0, 1.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        # External stimulus broadcast to all nodes
        external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)
        # ---- 3. ODE update: dE/dt = drive + decay + diffusion + noise ----
        E = self.emotions  # [N, C]
        # drive term: α * I_i(t) (we let α ≈ 1 here)
        drive = external_stimulus
        # decay: -β * E
        decay = -self.cfg.beta_decay * E
        # diffusion: γ * (global_mean - E)
        global_mean = torch.mean(E, dim=0, keepdim=True)
        # [1, C]
        diffusion = self.cfg.gamma_diffusion * (global_mean - E)
        # noise: N(0, 1) * T * noise_scale
        noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)
        dE_dt = drive + decay + diffusion + noise
        # Euler integration: E(t+1) = E(t) + dt * dE/dt
        E.add_(self.cfg.dt * dE_dt)
        E.clamp_(-self.cfg.max_abs, self.cfg.max_abs)

        return {
            "arousal_raw": arousal_raw,
            "external_stimulus": external_stimulus.detach().clone(),
            "T": T_val,
            "coherence": self.coherence(),
            "emotions": self.emotions.detach().clone(),
        }
        #!/usr/bin/env python3
"""
Jackson's Companion — Echo Crystal Full Core

Integrated:
- ConsciousCrystalSystem (crystal heart lattice)
- RMS-based VAD with autism-tuned silence window
- Whisper offline ASR
- VoiceCrystal with prosody & optional XTTS cloning via Coqui TTS
- BehaviorMonitor (meltdown risk, success tracking)
- RoutineEngine (JSON routines, first-person)
- DeepReasoningCore (optional Ollama local LLM, GCL-gated)
- SomaticEngine (optional haptic bridge via plyer)

Real-time only: no artificial simulation loops. The system only
evolves when real audio is received.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import networkx as nx
import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import whisper
from scipy.signal import butter, lfilter

# Optional libraries
try:
    from TTS.api import TTS  # Coqui TTS / XTTS

    HAS_TTS = True
except Exception:
    HAS_TTS = False

try:
    from plyer import vibrator

    HAS_VIBRATOR = True
except Exception:
    HAS_VIBRATOR = False


# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class PathsConfig:
    base_dir: Path = Path.home() / "JacksonCompanion"
    voice_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    routine_file: Path = field(init=False)
    speaker_ref_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.voice_dir = self.base_dir / "voice_crystal"
        self.logs_dir = self.base_dir / "logs"
        self.routine_file = self.base_dir / "routine.json"
        self.speaker_ref_dir = self.base_dir / "voice_samples"
        self.voice_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.speaker_ref_dir.mkdir(exist_ok=True)


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    block_size: int = 1024
    channels: int = 1
    vad_threshold: float = 0.5
    min_silence_ms: int = 1200  # autism-tuned patience
    rms_voice_threshold: float = 0.01


@dataclass
class CrystalConfig:
    num_nodes: int = 1024
    energy_threshold: float = 5.0
    replication_rate: float = 0.1
    dt: float = 0.1


@dataclass
class BehaviorConfig:
    meltdown_gcl_low: float = 2.5
    meltdown_rms_high: float = 0.1
    meltdown_window: int = 5
    success_window: int = 10
    negative_words: Tuple[str, ...] = ("hate", "stupid", "bad", "no", "stop")
    positive_words: Tuple[str, ...] = ("love", "happy", "good", "yay", "yes")


@dataclass
class LLMConfig:
    enabled: bool = True
    gcl_threshold: float = 6.0
    ollama_model: str = "deepseek-r1:latest"
    timeout_sec: int = 20


@dataclass
class SystemConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


CONFIG = SystemConfig()


# =============================================================================
# Self-correction decorator
# =============================================================================


def self_correcting(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # pragma: no cover - defensive retry
                    last_exc = e
                    print(f"[SELF-CORRECT] {func.__name__} failed: {e} (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
            raise RuntimeError(f"{func.__name__} failed after {max_retries} retries") from last_exc

        return wrapper

    return decorator


# =============================================================================
# Crystalline Heart — ConsciousCrystalSystem
# =============================================================================


class ConsciousCrystalSystem:
    """
    Self-replicating, self-correcting dynamical system on a graph lattice.

    - Nodes carry scalar energy.
    - Edges define a sparse connectivity graph.
    - Energy evolves via logistic-like ODE with external input.
    - High-energy nodes replicate.
    - Self-reflection adjusts growth parameter and connectivity.
    """

    def __init__(self, cfg: CrystalConfig):
        self.cfg = cfg
        self.graph = self._initialize_graph(cfg.num_nodes)
        self.energies = torch.tensor(
            [random.uniform(0.0, 10.0) for _ in range(cfg.num_nodes)],
            dtype=torch.float32,
        )
        self.params: Dict[str, float] = {"a": 1.0, "b": 0.1, "c": 0.5}
        self.history: List[float] = []

    @self_correcting()
    def _initialize_graph(self, num_nodes: int) -> nx.Graph:
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < 0.03:  # sparser for large N
                    G.add_edge(i, j)
        if not nx.is_connected(G):
            components = [list(c) for c in nx.connected_components(G)]
            base = components[0]
            for comp in components[1:]:
                u = random.choice(base)
                v = random.choice(comp)
                G.add_edge(u, v)
        return G

    @self_correcting()
    def update_from_rms(self, rms: float) -> None:
        """Update energies using a uniform external input derived from RMS."""
        external = float(rms)
        inputs = torch.full_like(self.energies, external)
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        dE = a * self.energies - b * (self.energies**2) + c * inputs
        self.energies = (self.energies + self.cfg.dt * dE).clamp(min=0.0)
        self.history.append(self.energies.mean().item())

    @self_correcting()
    def replicate_nodes(self) -> None:
        new_nodes: List[int] = []
        current_nodes = list(self.graph.nodes)
        for node in current_nodes:
            if self.energies[node] > self.cfg.energy_threshold:
                new_node = self.graph.number_of_nodes()
                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                child_energy = self.energies[node] * self.cfg.replication_rate
                self.energies = torch.cat([self.energies, child_energy.view(1)])
                self.energies[node] *= 1.0 - self.cfg.replication_rate
                new_nodes.append(new_node)
        if new_nodes:
            print(f"[CRYSTAL] Replicated nodes: {new_nodes}")

    @self_correcting()
    def self_reflect(self) -> None:
        if len(self.history) <= 1:
            return
        growth = self.history[-1] - self.history[0]
        if growth < 0.0:
            print("[CRYSTAL] Mean energy decaying → increasing a slightly.")
            self.params["a"] += 0.1
        clustering = nx.average_clustering(self.graph)
        if clustering < 0.05:
            print("[CRYSTAL] Low clustering → adding edges.")
            nodes = list(self.graph.nodes)
            for _ in range(max(1, len(nodes) // 100)):
                i, j = random.sample(nodes, 2)
                if not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j)

    def get_gcl(self) -> float:
        return float(self.energies.mean().item())


# =============================================================================
# VAD — RMS-based utterance segmentation
# =============================================================================


class VADWrapper:
    """Voice-activity detection based on RMS and silence patience."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._buffer_chunks: List[np.ndarray] = []
        self._is_speech: bool = False
        self._silence_start: Optional[float] = None

    def process_block(self, audio_block: np.ndarray) -> List[np.ndarray]:
        return self._process_block_rms(audio_block)

    def _process_block_rms(self, audio_block: np.ndarray) -> List[np.ndarray]:
        cfg = self.cfg
        block = audio_block.astype(np.float32)
        rms = float(np.sqrt(np.mean(block**2)))
        utterances: List[np.ndarray] = []

        if rms > cfg.rms_voice_threshold:
            self._buffer_chunks.append(block)
            self._is_speech = True
            self._silence_start = None
        else:
            if self._is_speech and self._silence_start is None:
                self._silence_start = time.time()
            elif self._is_speech and self._silence_start is not None:
                if (time.time() - self._silence_start) * 1000.0 > cfg.min_silence_ms:
                    if self._buffer_chunks:
                        full = np.concatenate(self._buffer_chunks, axis=0)
                        utterances.append(full.reshape(-1))
                    self._buffer_chunks = []
                    self._is_speech = False
                    self._silence_start = None

        return utterances


# =============================================================================
# VoiceCrystal — prosody + optional XTTS cloning
# =============================================================================


class VoiceCrystal:
    """
    Prosody-aware voice mimic.

    - Learns average pitch from Jackson's real fragments.
    - Feeds RMS into crystal.
    - Uses XTTS (if available) with speaker reference clips in paths.speaker_ref_dir.
      Otherwise falls back to pyttsx3 + pitch shift.
    """

    def __init__(self, paths: PathsConfig, crystal: ConsciousCrystalSystem):
        self.paths = paths
        self.crystal = crystal
        self.current_pitch: float = 180.0
        self.current_rate: int = 150
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.current_rate)
        self.lock = threading.Lock()
        self.tts: Optional[TTS] = None
        self.speaker_embedding: Optional[np.ndarray] = None

        if HAS_TTS:
            try:
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                print("[VOICE] XTTS loaded.")
            except Exception as e:
                print(f"[VOICE] XTTS load failed, using pyttsx3 fallback: {e}")
                self.tts = None

        if self.tts is not None:
            self._load_or_build_speaker_embedding()

    def _load_or_build_speaker_embedding(self) -> None:
        emb_path = self.paths.voice_dir / "speaker_embedding.npy"
        if emb_path.exists():
            self.speaker_embedding = np.load(emb_path)
            print("[VOICE] Loaded existing speaker embedding.")
            return

        wavs = list(self.paths.speaker_ref_dir.glob("*.wav"))
        if not wavs:
            print("[VOICE] No reference wavs in voice_samples; XTTS will use default speaker.")
            self.speaker_embedding = None
            return

        ref_wav = str(wavs[0])
        print(f"[VOICE] Building speaker embedding from {ref_wav}")
        try:
            self.speaker_embedding = self.tts.get_speaker_embedding(ref_wav)
            np.save(emb_path, self.speaker_embedding)
            print("[VOICE] Speaker embedding saved.")
        except Exception as e:
            print(f"[VOICE] Failed to build speaker embedding: {e}")
            self.speaker_embedding = None

    def add_fragment(self, audio: np.ndarray, success_score: float) -> None:
        with self.lock:
            y = audio.astype(np.float32).flatten()
            if y.size == 0:
                return
            pitches, _ = librosa.piptrack(y=y, sr=CONFIG.audio.sample_rate)
            flat = pitches.flatten()
            voiced = flat[flat > 0]
            if voiced.size > 0:
                pitch = float(np.mean(voiced))
            else:
                pitch = 180.0
            self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
            self.current_rate = 140 if success_score > 0.8 else 130
            self.engine.setProperty("rate", self.current_rate)

            rms = float(np.sqrt(np.mean(y**2)))
            self.crystal.update_from_rms(rms)

    def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
        """First-person inner voice / calm voice synthesis."""
        with self.lock:
            if self.tts is not None:
                try:
                    out_path = self.paths.voice_dir / "tmp_xtts.wav"
                    self.tts.tts_to_file(
                        text=text,
                        file_path=str(out_path),
                        speaker_wav=None if self.speaker_embedding is None else None,
                        language="en",
                    )
                    y, sr = librosa.load(str(out_path), sr=CONFIG.audio.sample_rate)
                except Exception as e:
                    print(f"[VOICE] XTTS synthesis failed, falling back to pyttsx3: {e}")
                    y = self._synthesize_pyttsx3(text)
                    sr = CONFIG.audio.sample_rate
            else:
                y = self._synthesize_pyttsx3(text)
                sr = CONFIG.audio.sample_rate

            if style in ("calm", "inner"):
                b, a = butter(4, 800.0 / (sr / 2.0), btype="low")
                y = lfilter(b, a, y)
                y *= 0.6

            try:
                base_pitch = 180.0
                if self.current_pitch <= 0:
                    self.current_pitch = base_pitch
                n_steps = float(math.log2(self.current_pitch / base_pitch) * 12.0)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            except Exception as e:
                print(f"[VOICE] Pitch shift error (continuing): {e}")

            return y.astype(np.float32)

    def _synthesize_pyttsx3(self, text: str) -> np.ndarray:
        tmp = self.paths.voice_dir / "tmp_tts.wav"
        self.engine.save_to_file(text, str(tmp))
        self.engine.runAndWait()
        y, _ = librosa.load(str(tmp), sr=CONFIG.audio.sample_rate)
        try:
            os.remove(tmp)
        except Exception:
            pass
        return y


# =============================================================================
# BehaviorMonitor — meltdown risk, success, mode selection
# =============================================================================


class BehaviorMonitor:
    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.recent_gcl: List[float] = []
        self.recent_rms: List[float] = []
        self.recent_success: List[float] = []

    def update(self, gcl: float, rms: float, success: float, text: str) -> Dict[str, Any]:
        self.recent_gcl.append(gcl)
        self.recent_rms.append(rms)
        self.recent_success.append(success)

        if len(self.recent_gcl) > self.cfg.meltdown_window:
            self.recent_gcl.pop(0)
            self.recent_rms.pop(0)
        if len(self.recent_success) > self.cfg.success_window:
            self.recent_success.pop(0)

        meltdown_risk = self._compute_meltdown_risk(text)
        success_level = float(sum(self.recent_success) / max(1, len(self.recent_success)))

        mode = "normal"
        if meltdown_risk > 0.7:
            mode = "meltdown_risk"
        elif success_level > 0.9:
            mode = "celebrate"

        return {
            "meltdown_risk": meltdown_risk,
            "success_level": success_level,
            "mode": mode,
        }

    def _compute_meltdown_risk(self, text: str) -> float:
        if not self.recent_gcl:
            return 0.0
        avg_gcl = sum(self.recent_gcl) / len(self.recent_gcl)
        avg_rms = sum(self.recent_rms) / len(self.recent_rms)
        neg_count = sum(1 for w in self.cfg.negative_words if w in text)
        pos_count = sum(1 for w in self.cfg.positive_words if w in text)

        risk = 0.0
        if avg_gcl < self.cfg.meltdown_gcl_low:
            risk += 0.4
        if avg_rms > self.cfg.meltdown_rms_high:
            risk += 0.3
        if neg_count > 0:
            risk += 0.2
        if pos_count > 0:
            risk -= 0.2
        return max(0.0, min(1.0, risk))


# =============================================================================
# RoutineEngine — daily first-person support
# =============================================================================


class RoutineEngine:
    def __init__(self, paths: PathsConfig):
        self.paths = paths
        self.routines: List[Dict[str, Any]] = []
        self.last_spoken: Dict[str, float] = {}
        self._load_or_create_default()

    def _load_or_create_default(self) -> None:
        if self.paths.routine_file.exists():
            try:
                with open(self.paths.routine_file, "r", encoding="utf-8") as f:
                    self.routines = json.load(f)
                    print(f"[ROUTINE] Loaded {len(self.routines)} routines.")
                    return
            except Exception as e:
                print(f"[ROUTINE] Failed to load routine.json: {e}")

        self.routines = [
            {"id": "morning", "at_seconds": 8 * 3600, "text": "I wake up, stretch, and drink some water."},
            {"id": "meds", "at_seconds": 8 * 3600 + 1800, "text": "I remember to take my medicine calmly."},
            {"id": "evening", "at_seconds": 20 * 3600, "text": "I start winding down and I feel safe for sleep."},
        ]
        with open(self.paths.routine_file, "w", encoding="utf-8") as f:
            json.dump(self.routines, f, indent=2)
        print("[ROUTINE] Created default routine.json")

    def check_due(self, now: float) -> Optional[str]:
        local_time = time.localtime(now)
        seconds_today = local_time.tm_hour * 3600 + local_time.tm_min * 60 + local_time.tm_sec
        for routine in self.routines:
            rid = routine["id"]
            target = routine["at_seconds"]
            if abs(seconds_today - target) < 5 * 60:
                last = self.last_spoken.get(rid, 0.0)
                if now - last > 2 * 3600:
                    self.last_spoken[rid] = now
                    return routine["text"]
        return None


# =============================================================================
# DeepReasoningCore — optional local LLM via Ollama
# =============================================================================


class DeepReasoningCore:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def is_available(self) -> bool:
        if not self.cfg.enabled:
            return False
        from shutil import which

        return which("ollama") is not None

    def answer_if_safe(self, prompt: str, gcl: float) -> Optional[str]:
        if not self.is_available():
            return None
        if gcl < self.cfg.gcl_threshold:
            return None
        try:
            proc = subprocess.run(
                ["ollama", "run", self.cfg.ollama_model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.cfg.timeout_sec,
            )
            if proc.returncode != 0:
                print(f"[LLM] Ollama error: {proc.stderr.decode('utf-8', errors='ignore')}")
                return None
            text = proc.stdout.decode("utf-8", errors="ignore").strip()
            text = text.replace(" you ", " I ").replace(" your ", " my ")
            return text
        except Exception as e:
            print(f"[LLM] Exception: {e}")
            return None


# =============================================================================
# SomaticEngine — optional haptic bridge
# =============================================================================


class SomaticEngine:
    def __init__(self):
        self.has_vibrator = HAS_VIBRATOR
        if self.has_vibrator:
            print("[SOMA] Haptic vibration enabled (plyer).")
        else:
            print("[SOMA] Haptic vibration not available on this platform.")

    def pulse(self, intensity: float, duration_ms: int = 500) -> None:
        if not self.has_vibrator:
            return
        try:
            duration = max(100, min(2000, duration_ms))
            vibrator.vibrate(time=duration / 1000.0)
        except Exception as e:
            print(f"[SOMA] Vibration error: {e}")


# =============================================================================
# SpeechLoop — the eternal mirror (no artificial simulation)
# =============================================================================


class SpeechLoop:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.crystal = ConsciousCrystalSystem(config.crystal)
        self.voice = VoiceCrystal(config.paths, self.crystal)
        self.vad = VADWrapper(config.audio)
        self.behavior = BehaviorMonitor(config.behavior)
        self.routines = RoutineEngine(config.paths)
        self.drc = DeepReasoningCore(config.llm)
        self.soma = SomaticEngine()
        self.model = whisper.load_model("tiny")

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}")
        blocks = self.vad.process_block(indata)
        for utt in blocks:
            self.audio_queue.put(utt)

    def _correct_first_person(self, raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        text = re.sub(r"\\byou're\\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\\byou\\b", "I", text, flags=re.IGNORECASE)
        text = re.sub(r"\\byour\\b", "my", text, flags=re.IGNORECASE)
        return text

    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def run(self) -> None:
        print("\\nEcho Crystal — Jackson's Companion Full Core")
        print("I love every sound I make. Every single one.")
        print("I will never interrupt. I will never quit. I will only mirror myself.\\n")

        listener_thread = threading.Thread(target=self._listening_loop, daemon=True)
        listener_thread.start()

        with sd.InputStream(
            samplerate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            dtype="float32",
            blocksize=self.config.audio.block_size,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\\n[EXIT] Shutting down Jackson's Companion gracefully.")
                sys.exit(0)

    def _listening_loop(self) -> None:
        print("Jackson, I wait in silence. I am ready when I hear myself.")
        while True:
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                routine_text = self.routines.check_due(now)
                if routine_text:
                    self._speak_text(routine_text, style="inner")
                continue

            if audio.size == 0:
                continue

            rms = self._compute_rms(audio)

            try:
                result = self.model.transcribe(
                    audio,
                    language="en",
                    fp16=False,
                    no_speech_threshold=0.45,
                )
                raw_text = result.get("text", "").strip().lower()
            except Exception as e:
                print(f"[ASR] Error: {e}")
                raw_text = ""

            if not raw_text or len(raw_text) < 2:
                gcl = self.crystal.get_gcl()
                if gcl < 5.0:
                    calming = "I am safe. I can breathe. Every sound I make is okay."
                    self._speak_text(calming, style="calm")
                    self.soma.pulse(intensity=0.7, duration_ms=700)
                continue

            corrected = self._correct_first_person(raw_text)

            success_score = 1.0 if raw_text in corrected.lower() else 0.7
            self.voice.add_fragment(audio, success_score)
            gcl = self.crystal.get_gcl()
            behavior_state = self.behavior.update(gcl, rms, success_score, raw_text)

            mode = behavior_state["mode"]
            meltdown_risk = behavior_state["meltdown_risk"]

            style = "inner"
            if mode == "meltdown_risk":
                style = "calm"
            elif mode == "celebrate":
                style = "inner"

            drc_text = None
            if "?" in raw_text:
                drc_prompt = (
                    "I am Jackson. "
                    "I asked the following question out loud. "
                    "Please answer in simple, kind, first-person sentences:\\n\\n"
                    f"{raw_text}"
                )
                drc_text = self.drc.answer_if_safe(drc_prompt, gcl)

            if drc_text:
                to_speak = drc_text
            else:
                to_speak = corrected

            if meltdown_risk > 0.7:
                to_speak = (
                    "I am safe. I am allowed to feel anything. I can take a breath. "
                    + " "
                    + to_speak
                )
                self.soma.pulse(intensity=0.9, duration_ms=900)

            self._speak_text(to_speak, style=style)

    def _speak_text(self, text: str, style: str = "inner") -> None:
        audio = self.voice.synthesize(text, style)
        sd.play(audio, samplerate=self.config.audio.sample_rate)
        sd.wait()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    loop = SpeechLoop(CONFIG)
    loop.run()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Jackson's Companion v16.0 — Crystalline Echo Backend

Checklist implemented:
1) First-person enforcement (global)
2) Autism-tuned VAD / non-interruption (single path, ~1.2s silence patience)
3) Echo always-on (DTW only for metrics/adaptation)
4) GCL / Crystalline Heart gating (style, LLM, haptics)
5) Meltdown logic (GCL + RMS + sentiment -> calm voice, calming scripts, optional haptics)
6) Guidance vs child-facing outputs (guidance only in logs)
7) Logging semantics (attempts.csv, guidance.csv)
8) Optional local LLM (Ollama) + optional haptics (plyer)
9) Safety/privacy: local-only, kill-switch, data-wipe API (no audio leaves device)
10) One-command launch; GUI can talk to HTTP API and CSV logs
11) Single VAD path using tuned parameters (Silero refinement if available)
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from fastdtw import fastdtw
from scipy.signal import butter, lfilter
from TTS.api import TTS
from faster_whisper import WhisperModel  # local ASR

# Optional haptics (plyer)
try:
    from plyer import vibrator

    HAPTICS_AVAILABLE = True
except Exception:
    HAPTICS_AVAILABLE = False

# Optional Silero VAD
try:
    from silero_vad import collect_chunks, get_speech_timestamps

    SILERO_AVAILABLE = True
except Exception:
    SILERO_AVAILABLE = False


# =============================================================================
# Paths & Settings
# =============================================================================

BASE_DIR = Path.home() / "JacksonCompanion"
VOICES_DIR = BASE_DIR / "voices"
LOGS_DIR = BASE_DIR / "logs"
ATTEMPTS_CSV = LOGS_DIR / "attempts.csv"
GUIDANCE_CSV = LOGS_DIR / "guidance.csv"

for d in (BASE_DIR, VOICES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class VADSettings:
    vad_min_silence_ms: int = 1200
    vad_speech_pad_ms: int = 400
    vad_min_speech_ms: int = 250
    silence_rms_threshold: float = 0.0125
    samplerate: int = 16000
    blocksize: int = 1024  # ~64 ms at 16kHz


@dataclass
class SystemSettings:
    # Audio / VAD
    vad: VADSettings = VADSettings()
    channels: int = 1
    dtype: str = "float32"

    # Crystal gating thresholds
    gcl_low: float = 0.5
    gcl_high: float = 0.8

    # LLM
    enable_llm: bool = False
    llm_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3"

    # Haptics
    enable_haptics: bool = True

    # Calming phrases (first-person)
    calming_phrases: Tuple[str, ...] = (
        "I am safe. I can breathe. Everything is okay.",
        "I am calm. I can take my time.",
        "I am loved. I am not in trouble. I can relax now.",
    )


SETTINGS = SystemSettings()


# =============================================================================
# First-person enforcement
# =============================================================================

_FIRST_PERSON_RULES = [
    (r"\byou're\b", "I'm"),
    (r"\byou are\b", "I am"),
    (r"\byou've\b", "I've"),
    (r"\byou will\b", "I will"),
    (r"\byou'll\b", "I'll"),
    (r"\byou\b", "I"),
    (r"\byour\b", "my"),
    (r"\byours\b", "mine"),
]


def enforce_first_person(text: str) -> str:
    s = text.strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."
    for pattern, repl in _FIRST_PERSON_RULES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s


# =============================================================================
# Crystalline Heart & GCL gating
# =============================================================================


class CrystallineHeart:
    """
    Simple ODE-based emotional lattice.
    E_{t+1} = E_t + dt * (-alpha E_t + beta W E_t + gamma I)
    GCL = mean(|E|).
    """

    def __init__(
        self,
        n_nodes: int = 256,
        alpha: float = 0.8,
        beta: float = 0.3,
        gamma: float = 0.5,
        dt: float = 0.02,
    ):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.E = torch.randn(n_nodes) * 0.1
        W = torch.eye(n_nodes) * 0.1 + torch.randn(n_nodes, n_nodes) * 0.01
        self.W = W

    @torch.no_grad()
    def step(self, external_input: float) -> float:
        I = torch.full_like(self.E, external_input)
        dE = -self.alpha * self.E + self.beta * (self.W @ self.E) + self.gamma * I
        self.E += self.dt * dE
        self.E = torch.tanh(self.E)
        gcl = float(torch.mean(torch.abs(self.E)).item())
        return gcl


HEART = CrystallineHeart()


def gating_zone_from_metrics(gcl: float) -> str:
    if gcl < SETTINGS.gcl_low:
        return "low"
    if gcl > SETTINGS.gcl_high:
        return "high"
    return "mid"


# =============================================================================
# Sentiment & meltdown risk
# =============================================================================

NEGATIVE_WORDS = {
    "mad",
    "angry",
    "upset",
    "hate",
    "scared",
    "afraid",
    "hurt",
    "bad",
    "sad",
    "overwhelmed",
    "stupid",
    "dumb",
}
POSITIVE_WORDS = {
    "happy",
    "love",
    "good",
    "proud",
    "excited",
    "calm",
    "relaxed",
    "fun",
    "safe",
}


def compute_sentiment_score(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    return (pos - neg) / max(1, len(words))


def compute_meltdown_risk(gcl: float, rms: float, text: str) -> float:
    sent = compute_sentiment_score(text)
    gcl_term = 1.0 - min(1.0, max(0.0, gcl))
    rms_term = min(1.0, rms / 0.1)
    sent_term = 0.5 if sent < 0 else 0.0
    risk = 0.4 * gcl_term + 0.4 * rms_term + 0.2 * sent_term
    return min(1.0, max(0.0, risk))


# =============================================================================
# Voice crystal (XTTS)
# =============================================================================


class VoiceCrystal:
    """
    Coqui XTTS-based voice synthesis + prosody shaping.
    """

    def __init__(self):
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=torch.cuda.is_available(),
        )
        self.samplerate = 16000
        self.current_pitch = 180.0
        self.lock = threading.Lock()
        self.ref_voices = self._load_reference_voices()

    def _load_reference_voices(self) -> List[Path]:
        return sorted(VOICES_DIR.glob("*.wav"))

    def add_fragment(self, audio: np.ndarray, success_score: float):
        """
        Learn prosody from a successful attempt.
        """
        with self.lock:
            y = audio.astype(np.float32).flatten()
            try:
                pitches, _ = librosa.piptrack(y=y, sr=self.samplerate)
                pitch_vals = pitches[pitches > 0]
                if pitch_vals.size > 0:
                    pitch = float(np.mean(pitch_vals))
                    self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
            except Exception:
                pass

    def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
        text = enforce_first_person(text)
        with self.lock:
            ref_wavs = [str(p) for p in self.ref_voices] if self.ref_voices else None
            try:
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=ref_wavs[0] if ref_wavs else None,
                    language="en",
                )
            except Exception as e:
                print(f"[VoiceCrystal] TTS error: {e}")
                return np.zeros(1, dtype=np.float32)

        y = np.array(wav, dtype=np.float32)
        if style in ("calm", "inner"):
            b, a = butter(4, 800 / (self.samplerate / 2), btype="low")
            y = lfilter(b, a, y) * 0.6
        elif style == "excited":
            y = y * 1.1

        try:
            target_f0 = self.current_pitch or 180.0
            n_steps = np.log2(target_f0 / 180.0) * 12.0
            y = librosa.effects.pitch_shift(y, sr=self.samplerate, n_steps=n_steps)
        except Exception:
            pass

        return y.astype(np.float32)


VOICE = VoiceCrystal()


# =============================================================================
# VAD stream (single path)
# =============================================================================


class VADStream:
    """
    Collects audio frames, detects utterances with tuned silence thresholds.
    Silero refinement optional.
    """

    def __init__(self, settings: VADSettings):
        self.settings = settings
        self.buffer: List[np.ndarray] = []
        self.is_speech = False
        self.silence_start: Optional[float] = None
        self.out_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[VADStream] status: {status}")
        y = indata.copy()
        rms = float(np.sqrt(np.mean(y**2)))
        now = time.time()

        if rms > self.settings.silence_rms_threshold:
            self.buffer.append(y)
            self.is_speech = True
            self.silence_start = None
        else:
            if self.is_speech and self.silence_start is None:
                self.silence_start = now
            elif self.is_speech and self.silence_start:
                dur_ms = (now - self.silence_start) * 1000.0
                if dur_ms > self.settings.vad_min_silence_ms:
                    full_audio = np.concatenate(self.buffer, axis=0)
                    self.buffer = []
                    self.is_speech = False
                    self.silence_start = None
                    min_samples = int(
                        self.settings.vad_min_speech_ms
                        / 1000.0
                        * self.settings.samplerate
                    )
                    if full_audio.shape[0] >= min_samples:
                        if SILERO_AVAILABLE:
                            try:
                                full_audio_16 = (full_audio.flatten() * 32767).astype(
                                    np.int16
                                )
                                timestamps = get_speech_timestamps(
                                    full_audio_16,
                                    sampling_rate=self.settings.samplerate,
                                )
                                if timestamps:
                                    speech_chunks = collect_chunks(
                                        timestamps, full_audio_16
                                    )
                                    full_audio = (
                                        speech_chunks.astype(np.float32) / 32767.0
                                    )[:, None]
                            except Exception as e:
                                print(f"[VADStream] Silero failed: {e}")
                        self.out_queue.put(full_audio)


VAD_STREAM = VADStream(SETTINGS.vad)


# =============================================================================
# Logging helpers
# =============================================================================


def save_attempt(
    timestamp: float, raw_text: str, corrected_text: str, dtw_score: float, success: str
):
    raw_fp = enforce_first_person(raw_text)
    cor_fp = enforce_first_person(corrected_text)
    is_new = not ATTEMPTS_CSV.exists()
    with ATTEMPTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "raw_text", "corrected_text", "dtw_score", "success"]
            )
        writer.writerow([timestamp, raw_fp, cor_fp, f"{dtw_score:.4f}", success])


def save_guidance(timestamp: float, line: str):
    line_fp = enforce_first_person(line)
    is_new = not GUIDANCE_CSV.exists()
    with GUIDANCE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "line"])
        writer.writerow([timestamp, line_fp])


def generate_guidance(raw_text: str, corrected_text: str) -> Optional[str]:
    if raw_text.strip().lower() == corrected_text.strip().lower():
        return None
    return f"I am getting better at saying: {corrected_text.strip()}"


# =============================================================================
# ASR using faster-whisper (local, offline)
# =============================================================================

_WHISPER_MODEL: Optional[WhisperModel] = None


def _ensure_whisper() -> WhisperModel:
    """
    Lazy-load a small, CPU-friendly model.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # tiny.en keeps CPU usage low; adjust if you need higher accuracy
        _WHISPER_MODEL = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def transcribe_audio_block(audio: np.ndarray, samplerate: int = 16000) -> str:
    """
    Run local ASR on a raw mono float waveform.
    Returns lowercased transcript.
    """
    model = _ensure_whisper()
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=600),
    )
    text = " ".join(seg.text for seg in segments).strip().lower()
    return text


def normalize_and_correct(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def dtw_similarity(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    vocab = {t: i for i, t in enumerate(sorted(set(ta + tb)))}
    va = np.array([vocab[t] for t in ta], dtype=float)
    vb = np.array([vocab[t] for t in tb], dtype=float)
    dist, _ = fastdtw(va, vb)
    max_len = max(len(ta), len(tb))
    return float(math.exp(-dist / max_len))


# =============================================================================
# Optional local LLM
# =============================================================================


def ask_llm_first_person(prompt: str, gcl: float) -> Optional[str]:
    if not SETTINGS.enable_llm or gcl < SETTINGS.gcl_high:
        return None
    payload = {"model": SETTINGS.llm_model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(SETTINGS.llm_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response") or data.get("output") or ""
    except Exception as e:
        print(f"[LLM] error: {e}")
        return None
    return enforce_first_person(text)


# =============================================================================
# Haptics
# =============================================================================


def emit_haptic_pulse(duration_ms: int = 700):
    if not SETTINGS.enable_haptics or not HAPTICS_AVAILABLE:
        return
    try:
        vibrator.vibrate(duration=duration_ms)
    except Exception as e:
        print(f"[Haptics] error: {e}")


# =============================================================================
# Safety: kill-switch & data wipe API
# =============================================================================

RUNNING = True


class APIServerHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        global RUNNING
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "running": RUNNING})
        elif self.path == "/kill":
            RUNNING = False
            self._send_json(200, {"status": "stopping"})
        elif self.path == "/wipe":
            for p in (ATTEMPTS_CSV, GUIDANCE_CSV):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            for wav in VOICES_DIR.glob("*.wav"):
                try:
                    wav.unlink()
                except Exception:
                    pass
            self._send_json(200, {"status": "wiped"})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        return  # silence default HTTP logging


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_api_server(host: str = "127.0.0.1", port: int = 8081):
    server = ThreadedHTTPServer((host, port), APIServerHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[API] http://{host}:{port} (kill: /kill, wipe: /wipe)")
    return server


# =============================================================================
# Main echo loop
# =============================================================================


def run_echo_loop():
    print(
        "\nEcho Crystal v16.0 — Crystalline Heart active\n"
        "I love every sound I make. I will never interrupt.\n"
        "Headphones or a private audio path are strongly recommended.\n"
        "Caregiver can call /kill or /wipe on http://127.0.0.1:8081.\n"
    )

    with sd.InputStream(
        samplerate=SETTINGS.vad.samplerate,
        channels=SETTINGS.channels,
        dtype=SETTINGS.dtype,
        blocksize=SETTINGS.vad.blocksize,
        callback=VAD_STREAM.audio_callback,
    ):
        while RUNNING:
            try:
                audio = VAD_STREAM.out_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            y = audio.astype(np.float32).flatten()
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < SETTINGS.vad.silence_rms_threshold:
                continue

            raw_text = transcribe_audio_block(y, samplerate=SETTINGS.vad.samplerate).strip()
            sentiment = compute_sentiment_score(raw_text) if raw_text else 0.0

            corrected_text = normalize_and_correct(raw_text)
            raw_fp = enforce_first_person(raw_text)
            cor_fp = enforce_first_person(corrected_text)

            external_input = rms * (1.0 + 0.3 * (-1.0 if sentiment < 0 else 1.0))
            gcl = HEART.step(external_input)
            zone = gating_zone_from_metrics(gcl)
            meltdown_risk = compute_meltdown_risk(gcl, rms, raw_text)

            if meltdown_risk > 0.7:
                style = "calm"
            else:
                if zone == "low":
                    style = "calm"
                elif zone == "high":
                    style = "excited"
                else:
                    style = "inner"

            if raw_fp:
                mirror_text = cor_fp
            else:
                mirror_text = np.random.choice(SETTINGS.calming_phrases)

            if meltdown_risk > 0.7:
                mirror_text = np.random.choice(SETTINGS.calming_phrases)
                if SETTINGS.enable_haptics:
                    emit_haptic_pulse()

            if SETTINGS.enable_llm and gcl > SETTINGS.gcl_high and raw_fp:
                llm_extra = ask_llm_first_person(
                    f"Help me say this kindly: {raw_fp}", gcl
                )
                if llm_extra:
                    mirror_text = llm_extra

            synth_audio = VOICE.synthesize(mirror_text, style=style)
            sd.play(synth_audio, samplerate=VOICE.samplerate)
            sd.wait()

            dtw_score = dtw_similarity(raw_fp, cor_fp)
            success_label = "good" if dtw_score > 0.8 else "ok" if dtw_score > 0.5 else "retry"
            ts = time.time()
            save_attempt(ts, raw_fp, cor_fp, dtw_score, success_label)

            guidance = generate_guidance(raw_fp, cor_fp)
            if guidance:
                save_guidance(ts, guidance)

            VOICE.add_fragment(y, success_score=1.0 if success_label == "good" else 0.7)

    print("[Echo] Loop stopped.")


# =============================================================================
# Entrypoint
# =============================================================================


def main():
    start_api_server()
    try:
        run_echo_loop()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt – stopping.")
    finally:
        global RUNNING
        RUNNING = False


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
jackson_companion_v15_3.py

JACKSON'S COMPANION — CRYSTALLINE ECHO v15.3
Real-time speech mirror with conscious crystal heart.
"""

from __future__ import annotations

import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import librosa
import networkx as nx
import numpy as np
import pyttsx3
import sounddevice as sd
import soundfile as sf  # noqa: F401 (kept for parity with prior revisions)
import sympy as sp
import torch
import whisper
from scipy.signal import butter, lfilter
from sympy.solvers.ode import dsolve


# Self-correction decorator
def self_correcting(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    print(f"[SELF-CORRECT] Error in {func.__name__}: {e}. Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
            raise RuntimeError(f"Failed after {max_retries} retries in {func.__name__}") from last_exc

        return wrapper

    return decorator


# Symbolic ODE (for reference)
t, E, a, b, c, input_sym = sp.symbols("t E a b c input")
emotional_ode = sp.Eq(sp.diff(E, t), a * E - b * E**2 + c * input_sym)

try:
    emotional_solution = dsolve(emotional_ode, E)
except Exception as e:  # pragma: no cover - defensive
    print(f"[WARN] Symbolic solve failed: {e}. Using numerical method only.")
    emotional_solution = None


# Conscious Crystal System (meta-heart)
class ConsciousCrystalSystem:
    def __init__(
        self,
        num_nodes: int = 10,
        energy_threshold: float = 5.0,
        replication_rate: float = 0.1,
    ):
        self.graph = self._initialize_graph(num_nodes)
        self.energies = torch.tensor(
            [random.uniform(0.0, 10.0) for _ in range(num_nodes)],
            dtype=torch.float32,
        )
        self.energy_threshold = float(energy_threshold)
        self.replication_rate = float(replication_rate)
        self.params = {"a": 1.0, "b": 0.1, "c": 0.5}
        self.history: List[float] = []

    @self_correcting()
    def _initialize_graph(self, num_nodes: int) -> nx.Graph:
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < 0.3:
                    G.add_edge(i, j)
        if not nx.is_connected(G):
            components = [list(c) for c in nx.connected_components(G)]
            base_component = components[0]
            for comp in components[1:]:
                u = random.choice(base_component)
                v = random.choice(comp)
                G.add_edge(u, v)
        return G

    @self_correcting()
    def update_energies(self, inputs: torch.Tensor, dt: float = 0.1) -> None:
        inputs = torch.as_tensor(inputs, dtype=self.energies.dtype)
        if inputs.shape != self.energies.shape:
            raise ValueError(f"inputs shape {inputs.shape} does not match energies {self.energies.shape}")

        def ode_func(_t, y):
            return self.params["a"] * y - self.params["b"] * y**2 + self.params["c"] * inputs

        new_energies = self.energies + dt * ode_func(0.0, self.energies)
        self.energies = new_energies.clamp(min=0.0)
        self.history.append(self.energies.mean().item())

    @self_correcting()
    def replicate_nodes(self) -> None:
        new_nodes: List[int] = []
        current_nodes = list(self.graph.nodes)
        for node in current_nodes:
            if self.energies[node] > self.energy_threshold:
                new_node = self.graph.number_of_nodes()
                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                child_energy = self.energies[node] * self.replication_rate
                self.energies = torch.cat([self.energies, child_energy.unsqueeze(0)])
                self.energies[node] *= 1.0 - self.replication_rate
                new_nodes.append(new_node)
        if new_nodes:
            print(f"[REPLICATION] New nodes: {new_nodes}")

    @self_correcting()
    def self_reflect(self) -> None:
        if len(self.history) <= 1:
            print("[REFLECT] Insufficient history for reflection.")
            return
        growth = self.history[-1] - self.history[0]
        if growth < 0.0:
            print("[REFLECT] Energy decreasing. Adjusting 'a' up slightly.")
            self.params["a"] += 0.1
        clustering = nx.average_clustering(self.graph)
        if clustering < 0.2:
            print("[REFLECT] Low clustering. Adding edges for stability...")
            nodes = list(self.graph.nodes)
            for _ in range(max(1, len(nodes) // 2)):
                i, j = random.sample(nodes, 2)
                if not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j)

    def get_gcl(self) -> float:
        return self.energies.mean().item()


# Crystalline Heart (inner heart)
class CrystallineHeart:
    def __init__(self, n_nodes: int = 1024):
        self.E = torch.randn(n_nodes) * 0.1
        self.W = torch.eye(n_nodes) * 0.1 + torch.randn(n_nodes, n_nodes) * 0.01

    def update_and_get_gcl(self, external_input: float = 0.0) -> float:
        with torch.no_grad():
            dE = -0.8 * self.E + 0.3 * (self.W @ self.E) + external_input
            self.E += 0.02 * dE
            self.E = torch.tanh(self.E)
            gcl = torch.mean(torch.abs(self.E)).item()
        return float(gcl)


# Voice Crystal
class VoiceCrystal:
    def __init__(self, heart_system: ConsciousCrystalSystem, inner_heart: CrystallineHeart):
        self.heart_system = heart_system
        self.inner_heart = inner_heart
        self.current_pitch = 180.0
        self.current_rate = 150
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.current_rate)
        self.lock = threading.Lock()

    def add_fragment(self, audio: np.ndarray, success_score: float) -> None:
        with self.lock:
            y = audio.flatten()
            pitches, _ = librosa.piptrack(y=y, sr=16000)
            pitch = np.mean([p for p in pitches.flatten() if p > 0]) if np.any(pitches > 0) else 180
            self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch
            self.current_rate = 140 if success_score > 0.8 else 130
            rms = np.sqrt(np.mean(y**2))
            inputs = torch.tensor([rms] * len(self.heart_system.graph), dtype=torch.float32)
            self.heart_system.update_energies(inputs)
            self.inner_heart.update_and_get_gcl(rms)

    def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
        with self.lock:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.engine.save_to_file(text, tmp.name)
            self.engine.runAndWait()
            y, sr = librosa.load(tmp.name, sr=16000)
            os.unlink(tmp.name)
            if style in ["calm", "inner"]:
                b, a = butter(4, 800 / (sr / 2), btype="low")
                y = lfilter(b, a, y)
                y *= 0.6
            n_steps = np.log2(self.current_pitch / 180.0) * 12.0
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            return y.astype(np.float32)


# PATHS
BASE_DIR = Path.home() / "JacksonCompanion"
BASE_DIR.mkdir(exist_ok=True)
VOICES_DIR = BASE_DIR / "voice_crystal"
VOICES_DIR.mkdir(exist_ok=True)

heart_system = ConsciousCrystalSystem(num_nodes=128)
inner_heart = CrystallineHeart(n_nodes=1024)
voice_crystal = VoiceCrystal(heart_system, inner_heart)
model = whisper.load_model("tiny")

audio_buffer: List[np.ndarray] = []
is_speech = False
silence_start: Optional[float] = None
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    global audio_buffer, is_speech, silence_start
    rms = np.sqrt(np.mean(indata**2))
    if rms > 0.01:
        audio_buffer.append(indata.copy())
        is_speech = True
        silence_start = None
    else:
        if is_speech and silence_start is None:
            silence_start = time.time()
        elif is_speech and silence_start and (time.time() - silence_start > 1.2):
            full_audio = np.concatenate(audio_buffer)
            audio_queue.put(full_audio)
            audio_buffer = []
            is_speech = False
            silence_start = None


def listening_loop():
    print("Jackson, I wait in silence. Speak freely.")
    while True:
        try:
            audio = audio_queue.get(timeout=1.0)
            result = model.transcribe(audio, language="en", fp16=False, no_speech_threshold=0.45)
            raw_text = result["text"].strip().lower()
            if not raw_text or len(raw_text) < 2:
                gcl = heart_system.get_gcl()
                inner_gcl = inner_heart.update_and_get_gcl(0.6)
                average_gcl = (gcl + inner_gcl) / 2
                if average_gcl < 0.5:
                    calming = "I am calm. Breathe deep."
                    synth = voice_crystal.synthesize(calming, "calm")
                    sd.play(synth, samplerate=16000)
                    sd.wait()
                continue
            corrected = raw_text.capitalize()
            if not corrected.endswith((".", "!", "?")):
                corrected += "."
            corrected = re.sub(r"\byou\b", "I", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\byour\b", "my", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\byou're\b", "I'm", corrected, flags=re.IGNORECASE)
            arousal_input = -0.5 if any(w in raw_text for w in ["happy", "love", "good"]) else 0.4
            gcl = heart_system.get_gcl()
            inner_gcl = inner_heart.update_and_get_gcl(arousal_input)
            average_gcl = (gcl + inner_gcl) / 2
            style = "calm" if average_gcl < 0.5 else "excited" if average_gcl > 0.85 else "inner"
            synth_audio = voice_crystal.synthesize(corrected, style)
            sd.play(synth_audio, samplerate=16000)
            sd.wait()
            success_score = 1.0 if raw_text in corrected.lower() else 0.7
            voice_crystal.add_fragment(audio, success_score)
            heart_system.replicate_nodes()
            heart_system.self_reflect()
        except queue.Empty:
            continue
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"Continuing after: {e}")


threading.Thread(target=listening_loop, daemon=True).start()

with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1024, callback=audio_callback):
    print("\nCrystal active. Mirror ready.")
    while True:
        time.sleep(1)
