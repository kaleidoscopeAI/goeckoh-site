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

import csv
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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

import librosa
import networkx as nx
import numpy as np
import sounddevice as sd
import torch
import whisper
from scipy.signal import butter, lfilter
from difflib import SequenceMatcher

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

RUNNING = True
STATE: Dict[str, Any] = {
    "gcl": 0.0,
    "risk": 0.0,
    "style": "inner",
    "attempts": 0,
    "avg_similarity": 0.0,
    "voice_counts": {"calm": 0, "inner": 0, "excited": 0},
}

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
    attempts_csv: Path = field(init=False)
    guidance_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.voice_dir = self.base_dir / "voice_crystal"
        self.logs_dir = self.base_dir / "logs"
        self.routine_file = self.base_dir / "routine.json"
        self.speaker_ref_dir = self.base_dir / "voice_samples"
        self.voice_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.speaker_ref_dir.mkdir(exist_ok=True)
        self.attempts_csv = self.logs_dir / "attempts.csv"
        self.guidance_csv = self.logs_dir / "guidance.csv"


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    block_size: int = 1024
    channels: int = 1
    vad_threshold: float = 0.5
    min_silence_ms: int = 1200  # autism-tuned patience
    rms_voice_threshold: float = 0.01


@dataclass
class VoiceConfig:
    mode: str = "clone_only"  # enforce clone-or-silence; no robot fallback
    max_latency_ms: float = 350.0


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
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


CONFIG = SystemConfig()


# =============================================================================
# Self-correction decorator
# =============================================================================


def enforce_first_person(text: str) -> str:
    """
    Force output into first-person form and ensure closing punctuation.
    """
    s = text.strip()
    if not s:
        return ""
    if s and not s.endswith((".", "!", "?")):
        s += "."
    rules = (
        (r"\byou're\b", "I'm"),
        (r"\byou are\b", "I am"),
        (r"\byou've\b", "I've"),
        (r"\byou will\b", "I will"),
        (r"\byou'll\b", "I'll"),
        (r"\byou\b", "I"),
        (r"\byour\b", "my"),
        (r"\byours\b", "mine"),
    )
    for pattern, repl in rules:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s


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
# Logging helpers
# =============================================================================


def log_attempt(paths: PathsConfig, timestamp: float, raw_text: str, corrected: str, dtw_score: float, success: str, gcl: float, risk: float) -> None:
    fp_raw = enforce_first_person(raw_text)
    fp_corr = enforce_first_person(corrected)
    first = not paths.attempts_csv.exists()
    with paths.attempts_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["timestamp", "raw_text", "corrected_text", "dtw", "success", "gcl", "risk"])
        writer.writerow([timestamp, fp_raw, fp_corr, f"{dtw_score:.3f}", success, f"{gcl:.3f}", f"{risk:.3f}"])


def log_guidance(paths: PathsConfig, timestamp: float, phrase: str, risk: float, gcl: float) -> None:
    fp = enforce_first_person(phrase)
    first = not paths.guidance_csv.exists()
    with paths.guidance_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["timestamp", "event", "phrase", "gcl", "risk"])
        writer.writerow([timestamp, "guidance", fp, f"{gcl:.3f}", f"{risk:.3f}"])


def dtw_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def wipe_data(paths: PathsConfig) -> None:
    for p in [paths.attempts_csv, paths.guidance_csv]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    for wav in paths.voice_dir.glob("*.wav"):
        try:
            wav.unlink()
        except Exception:
            pass


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


@dataclass
class VoiceCrystalConfig:
    speaker_ref_dir: Path
    sample_rate: int = 16_000
    mode: str = "clone_only"
    max_latency_ms: float = 350.0


class VoiceCrystal:
    """
    XTTS-first voice clone (clone-or-silence).
    - Precomputes speaker embedding from reference wavs.
    - Returns np.float32 audio at sample_rate.
    - Emits prosody metrics for crystal feedback.
    """

    def __init__(self, cfg: VoiceCrystalConfig):
        self.cfg = cfg
        self.current_pitch: float = 180.0
        self.current_rate: int = 150
        self.lock = threading.Lock()

        self.tts: Optional[TTS] = None
        self.speaker_ref_path: Optional[str] = None
        self.speaker_refs: List[str] = []
        self.spk_embed: Optional[np.ndarray] = None
        self.speaker_id: str = "jackson_clone"
        self.clone_ready: bool = False
        self._prosody_metrics: Dict[str, Any] = {}

        if HAS_TTS:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[VOICE] Loading XTTS on {device}...")
                self.tts = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                    gpu=torch.cuda.is_available(),
                )
            except Exception as e:
                print(f"[VOICE] XTTS load failed; voice cloning unavailable: {e}")
                self.tts = None

        if self.tts is not None:
            self._init_speaker_ref()
            self._warmup()

    def _init_speaker_ref(self) -> None:
        wavs = sorted(self.cfg.speaker_ref_dir.glob("*.wav"))
        if not wavs:
            print(f"[VOICE] No reference wavs in {self.cfg.speaker_ref_dir}; XTTS will use default speaker.")
            self.speaker_ref_path = None
            self.speaker_refs = []
            return
        self.speaker_ref_path = str(wavs[0])
        self.speaker_refs = [str(w) for w in wavs]
        print(f"[VOICE] Using {len(self.speaker_refs)} reference wav(s); primary={self.speaker_ref_path}")

        embeds = []
        for path in self.speaker_refs:
            try:
                if hasattr(self.tts.speaker_manager, "compute_embedding"):
                    emb = self.tts.speaker_manager.compute_embedding(path)
                else:
                    emb = self.tts.speaker_manager.compute_speaker_embedding(path)
                embeds.append(np.asarray(emb, dtype=np.float32))
            except Exception as e:
                print(f"[VOICE] Embedding compute failed for {path}: {e}")
        if embeds:
            self.spk_embed = np.mean(np.stack(embeds, axis=0), axis=0)
            try:
                self.tts.speaker_manager.speaker_embeddings[self.speaker_id] = self.spk_embed
            except Exception as e:
                print(f"[VOICE] Failed to register speaker embedding: {e}")
        self.clone_ready = self.spk_embed is not None

    def add_fragment(self, audio: np.ndarray, success_score: float) -> float:
        """
        Update pitch estimate and return RMS for crystal feedback.
        """
        with self.lock:
            y = audio.astype(np.float32).flatten()
            if y.size == 0:
                return 0.0
            try:
                pitches, _ = librosa.piptrack(y=y, sr=self.cfg.sample_rate)
                flat = pitches.flatten()
                voiced = flat[flat > 0]
                pitch_val = float(np.mean(voiced)) if voiced.size else 180.0
            except Exception as e:
                print(f"[VOICE] Pitch tracking error: {e}")
                pitch_val = 180.0
            self.current_pitch = 0.95 * self.current_pitch + 0.05 * pitch_val
            self.current_rate = 140 if success_score > 0.8 else 130
            rms = float(np.sqrt(np.mean(y**2)))
            return rms

    def synthesize(self, text: str, style: str = "inner") -> np.ndarray:
        """
        Synthesize in child voice (XTTS if available), returning float32 at sample_rate.
        """
        text = enforce_first_person(text)
        if not text:
            return np.zeros((0,), dtype=np.float32)
        with self.lock:
            start = time.time()
            audio: Optional[np.ndarray] = None
            sr = self.cfg.sample_rate

            if self.tts is not None and self.clone_ready:
                try:
                    audio, sr = self._synthesize_xtts(text)
                except Exception as e:
                    print(f"[VOICE] XTTS synthesis failed: {e}")
                    audio = None

            if audio is None:
                print("[VOICE] Clone unavailable; returning silence (clone_only mode).")
                return np.zeros((int(self.cfg.sample_rate * 0.25),), dtype=np.float32)

            if style in ("calm", "inner"):
                try:
                    b, a = butter(4, 800.0 / (sr / 2.0), btype="low")
                    audio = lfilter(b, a, audio)
                    audio *= 0.6
                except Exception as e:
                    print(f"[VOICE] Style filter error: {e}")

            try:
                base_pitch = 180.0
                if self.current_pitch <= 0:
                    self.current_pitch = base_pitch
                n_steps = float(math.log2(self.current_pitch / base_pitch) * 12.0)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            except Exception as e:
                print(f"[VOICE] Pitch shift error: {e}")

            if sr != self.cfg.sample_rate:
                try:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sample_rate)
                except Exception as e:
                    print(f"[VOICE] Resample error: {e}")

            audio = audio.astype(np.float32)
            latency_ms = (time.time() - start) * 1000.0
            avg_rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
            self._prosody_metrics = {
                "latency_ms": latency_ms,
                "average_rms": avg_rms,
                "style": style,
            }
            if latency_ms > self.cfg.max_latency_ms:
                print(f"[VOICE] HARD WARNING latency {latency_ms:.1f} ms > target {self.cfg.max_latency_ms} ms")
            return audio

    def _synthesize_xtts(self, text: str) -> Tuple[np.ndarray, int]:
        if self.tts is None:
            raise RuntimeError("XTTS not loaded")
        try:
            if self.spk_embed is not None:
                try:
                    wav = self.tts.tts(text=text, speaker=self.speaker_id, language="en")
                except Exception:
                    wav = None
            else:
                wav = None
            if wav is None and self.speaker_ref_path:
                wav = self.tts.tts(text=text, speaker_wav=self.speaker_ref_path, language="en")
            if wav is None:
                raise RuntimeError("XTTS returned no audio")
            return np.array(wav, dtype=np.float32), 24_000
        except Exception as e:
            raise RuntimeError(f"XTTS synthesis error: {e}") from e

    def _warmup(self) -> None:
        try:
            _ = self._synthesize_xtts("I am here.")[0]
        except Exception as e:
            print(f"[VOICE] Warmup failed: {e}")

    def get_prosody_metrics(self) -> Dict[str, Any]:
        return dict(self._prosody_metrics)


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
# Safety API: /health /kill /wipe /api/metrics /api/behavior /api/voice-profile
# =============================================================================


class APIServerHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        global RUNNING
        path = self.path.split("?")[0]
        if path == "/health":
            self._send(200, {"status": "healthy" if RUNNING else "stopped", "running": RUNNING})
        elif path == "/api/metrics":
            self._send(200, {"attempts": STATE.get("attempts", 0), "avg_similarity": STATE.get("avg_similarity", 0.0)})
        elif path == "/api/behavior":
            self._send(200, {"gcl": STATE.get("gcl", 0.0), "risk": STATE.get("risk", 0.0), "style": STATE.get("style", "inner")})
        elif path == "/api/voice-profile":
            self._send(200, STATE.get("voice_counts", {}))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global RUNNING
        path = self.path.split("?")[0]
        if path == "/kill":
            RUNNING = False
            self._send(200, {"status": "stopping"})
        elif path == "/wipe":
            wipe_data(CONFIG.paths)
            self._send(200, {"status": "wiped"})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_api_server(host: str = "127.0.0.1", port: int = 8081) -> ThreadedHTTPServer:
    server = ThreadedHTTPServer((host, port), APIServerHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[API] Listening on http://{host}:{port} (/health /kill /wipe /api/*)")
    return server


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


class SpeechLoop:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.crystal = ConsciousCrystalSystem(config.crystal)
        self.voice = VoiceCrystal(
            VoiceCrystalConfig(
                speaker_ref_dir=config.paths.speaker_ref_dir,
                sample_rate=config.audio.sample_rate,
                mode=config.voice.mode,
                max_latency_ms=config.voice.max_latency_ms,
            )
        )
        self.vad = VADWrapper(config.audio)
        self.behavior = BehaviorMonitor(config.behavior)
        self.routines = RoutineEngine(config.paths)
        self.drc = DeepReasoningCore(config.llm)
        self.soma = SomaticEngine()
        self.model = whisper.load_model("tiny")

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.total_attempts = 0
        self.sim_sum = 0.0
        self.voice_usage = {"calm": 0, "inner": 0, "excited": 0}

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
                while RUNNING:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\\n[EXIT] Shutting down Jackson's Companion gracefully.")
                sys.exit(0)

    def _listening_loop(self) -> None:
        print("Jackson, I wait in silence. I am ready when I hear myself.")
        while RUNNING:
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                routine_text = self.routines.check_due(now)
                if routine_text:
                    self._speak_text(enforce_first_person(routine_text), style="inner")
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

            raw_fp = enforce_first_person(raw_text)
            corrected = enforce_first_person(raw_text)

            success_score = 1.0 if raw_text in corrected.lower() else 0.7
            rms_fragment = self.voice.add_fragment(audio, success_score)
            self.crystal.update_from_rms(rms_fragment)
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

            # Stats and logging
            self.voice_usage[style] = self.voice_usage.get(style, 0) + 1
            self.total_attempts += 1
            sim = dtw_similarity(raw_fp, corrected)
            self.sim_sum += sim
            STATE["attempts"] = self.total_attempts
            STATE["avg_similarity"] = self.sim_sum / max(1, self.total_attempts)
            STATE["gcl"] = gcl
            STATE["risk"] = meltdown_risk
            STATE["style"] = style
            STATE["voice_counts"] = dict(self.voice_usage)

            ts = time.time()
            success_label = "good" if sim > 0.8 else "ok" if sim > 0.5 else "retry"
            log_attempt(self.config.paths, ts, raw_fp, corrected, sim, success_label, gcl, meltdown_risk)
            if meltdown_risk > 0.7:
                log_guidance(self.config.paths, ts, to_speak, meltdown_risk, gcl)

    def _speak_text(self, text: str, style: str = "inner") -> None:
        audio = self.voice.synthesize(text, style)
        sd.play(audio, samplerate=self.config.audio.sample_rate)
        sd.wait()
        prosody = self.voice.get_prosody_metrics()
        avg_rms = float(prosody.get("average_rms", 0.0))
        self.crystal.update_from_rms(avg_rms)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    start_api_server()
    loop = SpeechLoop(CONFIG)
    loop.run()


if __name__ == "__main__":
    main()
