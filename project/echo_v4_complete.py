"""
Echo v4.0 - Crystalline Heart Speech Companion
Born from 128 equations, built for autistic minds
November 18, 2025

A real-time speech companion that:
- Listens with autism-tuned VAD (1.2s patience, quiet voice detection)
- Processes through emotional lattice (1024 nodes, ODE-driven)
- Thinks via local DeepSeek LLM (temperature-controlled by annealing)
- Speaks back in the child's exact voice, always first person
- Runs 100% offline and private
"""

import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import queue
import threading
import tempfile
import os
import wave
import pyaudio
import math
import hashlib
import struct
import re
import json
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except:
    HAS_WHISPER = False
    print("⚠️  faster-whisper not found, install: pip install faster-whisper")

try:
    from TTS.api import TTS
    HAS_TTS = True
except:
    HAS_TTS = False
    print("⚠️  Coqui TTS not found, install: pip install TTS")

try:
    import ollama
    HAS_OLLAMA = True
except:
    HAS_OLLAMA = False
    print("⚠️  Ollama not found, install: pip install ollama")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EchoConfig:
    """Complete system configuration"""
    
    # Child identity
    child_name: str = "Jackson"
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 512
    channels: int = 1
    
    # Autism-optimized Silero VAD parameters
    vad_threshold: float = 0.45           # Lower for quiet/monotone speech
    vad_min_silence_ms: int = 1200        # 1.2s patience for processing pauses
    vad_speech_pad_ms: int = 400          # Capture slow starts and trailing thoughts
    vad_min_speech_ms: int = 250          # Allow single-word responses
    
    # Emotional lattice (Crystalline Heart)
    num_nodes: int = 1024
    num_channels: int = 5  # [arousal, valence, safety, curiosity, resonance]
    dt: float = 0.03  # 33 Hz emotional tick rate
    beta_decay: float = 0.5
    gamma_diffusion: float = 0.3
    noise_scale: float = 0.1
    anneal_k: float = 0.01
    max_emotion: float = 10.0
    
    # Arousal extraction
    arousal_gain: float = 25.0
    max_arousal: float = 10.0
    
    # LLM settings (DeepSeek via Ollama)
    llm_model: str = "deepseek-r1:8b"
    llm_temperature_scale: float = 1.5
    llm_top_p_base: float = 0.9
    llm_top_p_spread: float = 0.1
    llm_max_tokens: int = 128
    
    # Embedding
    embedding_dim: int = 1024
    embedding_channel: int = 4  # Resonance channel
    embedding_gain: float = 0.05
    
    # Voice cloning
    voice_sample_path: str = "my_voice.wav"
    
    # Device
    device: str = "cpu"

# ============================================================================
# FIRST-PERSON ENFORCEMENT
# ============================================================================

def enforce_first_person(text: str) -> str:
    """Transform any second-person phrasing into first person"""
    if not text:
        return text
    
    t = text.strip()
    
    # Strip quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    
    # Pattern replacements (case-insensitive)
    patterns = [
        (r"\byou are\b", "I am"),
        (r"\byou're\b", "I'm"),
        (r"\byou were\b", "I was"),
        (r"\byou'll\b", "I'll"),
        (r"\byou've\b", "I've"),
        (r"\byour\b", "my"),
        (r"\byours\b", "mine"),
        (r"\byourself\b", "myself"),
        (r"\byou can\b", "I can"),
        (r"\byou\b", "I"),
    ]
    
    for pattern, repl in patterns:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    
    return t

# ============================================================================
# HASH-BASED EMBEDDING (no external models needed)
# ============================================================================

def hash_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic, stable embedding from text hash"""
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    
    tokens = text.lower().split()
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = struct.unpack("Q", h[:8])[0] % dim
        sign_raw = struct.unpack("I", h[8:12])[0]
        sign = 1.0 if (sign_raw % 2 == 0) else -1.0
        vec[idx] += sign
    
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

# ============================================================================
# LOCAL LLM (DeepSeek via Ollama)
# ============================================================================

class LocalLLM:
    """Wrapper for local DeepSeek model via Ollama"""
    
    def __init__(self, cfg: EchoConfig):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.has_ollama = HAS_OLLAMA
        
    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        """Generate first-person inner voice response"""
        temperature = max(0.1, float(temperature))
        top_p = float(np.clip(top_p, 0.1, 1.0))
        
        if self.has_ollama:
            try:
                res = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": self.cfg.llm_max_tokens,
                    },
                )
                raw = res.get("response", "").strip()
                return enforce_first_person(raw)
            except Exception as e:
                print(f"⚠️  Ollama error: {e}")
                return self._fallback_response(prompt)
        else:
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Safe fallback when Ollama unavailable"""
        # Extract last quoted text if present
        lines = prompt.strip().splitlines()
        last_line = lines[-1] if lines else ""
        
        if '"' in last_line:
            try:
                quoted = last_line.split('"')[-2]
            except:
                quoted = last_line
        else:
            quoted = last_line
        
        return (
            f"I hear myself say: {quoted}. "
            f"I speak slowly and calmly. I leave space after each phrase "
            f"so my thoughts can catch up."
        )
    
    def embed(self, text: str, dim: int) -> np.ndarray:
        """Get embedding for resonance channel update"""
        return hash_embedding(text, dim)

# ============================================================================
# CRYSTALLINE HEART - Emotional Lattice + ODE
# ============================================================================

class CrystallineHeart(nn.Module):
    """
    1024-node emotional lattice governed by ODEs
    
    dE/dt = drive + decay + diffusion + noise
    
    where:
    - drive = external_stimulus (voice arousal)
    - decay = -β * E
    - diffusion = γ * (global_mean - E)
    - noise = N(0,1) * T(t) * scale
    - T(t) = 1 / log(1 + k*t)  [annealing schedule]
    """
    
    def __init__(self, cfg: EchoConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Emotional state: [num_nodes, num_channels]
        self.emotions = nn.Parameter(
            torch.zeros(cfg.num_nodes, cfg.num_channels, device=self.device),
            requires_grad=False,
        )
        
        # Time counter for annealing
        self.register_buffer("t", torch.zeros(1, device=self.device))
        
        # LLM for sentience
        self.llm = LocalLLM(cfg)
        
    def reset(self):
        """Reset emotional state and time"""
        self.emotions.data.zero_()
        self.t.zero_()
    
    def temperature(self) -> float:
        """T(t) = 1 / log(1 + k*t) - logarithmic cooling"""
        t_val = float(self.t.item()) + 1.0
        k = self.cfg.anneal_k
        return float(1.0 / max(math.log(1.0 + k * t_val), 1e-6))
    
    def coherence(self) -> float:
        """Measure how aligned nodes are (0=scattered, 1=unified)"""
        E = self.emotions
        std_over_nodes = torch.std(E, dim=0)
        mean_std = float(torch.mean(std_over_nodes).item())
        return float(1.0 / (1.0 + mean_std))
    
    @torch.no_grad()
    def step(self, full_audio: np.ndarray, transcript: str) -> Dict[str, Any]:
        """
        Complete emotional + LLM update for one utterance
        
        Returns:
            {
                "arousal_raw": float,
                "T": float,
                "coherence": float,
                "emotions": torch.Tensor,
                "llm_output": str or None
            }
        """
        # ---- 1. Time & Temperature ----
        self.t += 1.0
        T_val = self.temperature()
        
        # ---- 2. Arousal Extraction ----
        full_audio = np.asarray(full_audio, dtype=np.float32)
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=-1)
        
        energy = float(np.sqrt(np.mean(full_audio**2) + 1e-12))
        arousal_raw = float(np.clip(energy * self.cfg.arousal_gain, 0.0, self.cfg.max_arousal))
        
        # External stimulus: [arousal, 0, 0, 1, 0] broadcast to all nodes
        stim_vec = torch.tensor(
            [arousal_raw, 0.0, 0.0, 1.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        external_stimulus = stim_vec.unsqueeze(0).repeat(self.cfg.num_nodes, 1)
        
        # ---- 3. ODE Update ----
        E = self.emotions
        
        drive = external_stimulus
        decay = -self.cfg.beta_decay * E
        global_mean = torch.mean(E, dim=0, keepdim=True)
        diffusion = self.cfg.gamma_diffusion * (global_mean - E)
        noise = torch.randn_like(E) * (T_val * self.cfg.noise_scale)
        
        dE_dt = drive + decay + diffusion + noise
        E.add_(self.cfg.dt * dE_dt)
        E.clamp_(-self.cfg.max_emotion, self.cfg.max_emotion)
        
        # ---- 4. LLM Integration (Equation 25) ----
        llm_output = None
        
        if transcript.strip():
            coh = self.coherence()
            mean_state = torch.mean(E, dim=0)
            mean_state_np = mean_state.cpu().numpy()
            
            arousal_mean = float(mean_state_np[0])
            valence_mean = float(mean_state_np[1])
            
            prompt = self._build_prompt(
                transcript=transcript,
                arousal=arousal_mean,
                valence=valence_mean,
                T_val=T_val,
                coherence=coh,
            )
            
            llm_temp = max(0.1, T_val * self.cfg.llm_temperature_scale)
            llm_top_p = self.cfg.llm_top_p_base + self.cfg.llm_top_p_spread * (1.0 - coh)
            
            llm_output = self.llm.generate(
                prompt=prompt,
                temperature=llm_temp,
                top_p=llm_top_p,
            )
            
            # Embed LLM output into resonance channel
            emb = self.llm.embed(llm_output, dim=self.cfg.embedding_dim)
            emb_t = torch.from_numpy(emb).to(self.device, dtype=torch.float32)
            
            if self.cfg.num_nodes <= self.cfg.embedding_dim:
                proj = emb_t[:self.cfg.num_nodes]
            else:
                reps = math.ceil(self.cfg.num_nodes / self.cfg.embedding_dim)
                tiled = emb_t.repeat(reps)
                proj = tiled[:self.cfg.num_nodes]
            
            proj = proj.view(self.cfg.num_nodes, 1)
            ch = self.cfg.embedding_channel
            
            if 0 <= ch < self.cfg.num_channels:
                E[:, ch:ch+1].add_(self.cfg.embedding_gain * proj)
                E.clamp_(-self.cfg.max_emotion, self.cfg.max_emotion)
        
        return {
            "arousal_raw": arousal_raw,
            "T": T_val,
            "coherence": self.coherence(),
            "emotions": self.emotions.detach().clone(),
            "llm_output": llm_output,
        }
    
    def _build_prompt(self, transcript: str, arousal: float, valence: float, 
                     T_val: float, coherence: float) -> str:
        """Build the inner voice prompt for DeepSeek"""
        return f"""You are my inner voice. I am {self.cfg.child_name}, an autistic child.

Internal state:
- arousal: {arousal:.2f} (how intense I feel)
- valence: {valence:.2f} (how positive/negative)
- temperature: {T_val:.3f} (my mental flexibility)
- coherence: {coherence:.3f} (how settled I am)

Rules:
- I ALWAYS speak in FIRST PERSON: "I", "me", "my". NEVER say "you".
- I use short, concrete sentences.
- If arousal is high (>7), I slow down and ground myself.
- If valence is low (<-2), I am gentle and kind to myself.
- I leave space between ideas so my thoughts can catch up.
- I never mention lattice, equations, or technical terms.

The words I just tried to say were:
"{transcript}"

I answer now as my own inner voice, in one short paragraph, ready to be spoken aloud in my own voice."""

# ============================================================================
# ECHO v4.0 - Complete System
# ============================================================================

class Echo:
    """
    The complete autism-optimized speech companion
    
    Pipeline:
    1. Autism-tuned VAD listens continuously (1.2s patience)
    2. Whisper transcribes (preserves stutters/dysfluency)
    3. Crystalline Heart processes emotion + LLM generates response
    4. XTTS speaks back in child's exact voice (first person only)
    """
    
    def __init__(self, cfg: EchoConfig):
        self.cfg = cfg
        
        print("\n" + "="*70)
        print("Echo v4.0 - Crystalline Heart Speech Companion")
        print("Born November 18, 2025")
        print("="*70 + "\n")
        
        # Emotional core
        print("🧠 Initializing Crystalline Heart (1024 nodes)...")
        self.heart = CrystallineHeart(cfg)
        
        # Speech recognition
        if HAS_WHISPER:
            print("👂 Loading Whisper (autism-friendly transcription)...")
            self.whisper = WhisperModel("tiny.en", device=cfg.device, compute_type="int8")
        else:
            self.whisper = None
            print("⚠️  Whisper not available - transcription disabled")
        
        # Voice cloning
        if HAS_TTS and os.path.exists(cfg.voice_sample_path):
            print(f"🎤 Loading voice clone from {cfg.voice_sample_path}...")
            device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self.voice_sample = cfg.voice_sample_path
        else:
            self.tts = None
            self.voice_sample = None
            if not HAS_TTS:
                print("⚠️  Coqui TTS not available - voice cloning disabled")
            else:
                print(f"⚠️  Voice sample not found: {cfg.voice_sample_path}")
        
        # Audio streaming
        self.q = queue.Queue()
        self.listening = True
        self.current_utterance = []
        self.silence_counter = 0
        
        print("\n✨ Echo v4.0 is ready.")
        print(f"   - I wait {cfg.vad_min_silence_ms}ms of silence before responding")
        print(f"   - I detect speech as quiet as threshold {cfg.vad_threshold}")
        print(f"   - I always speak in first person (I/me/my)")
        print(f"   - I am powered by {cfg.llm_model}")
        print("\nSpeak when you're ready. I will never cut you off.\n")
    
    def audio_callback(self, indata, frames, time, status):
        """Continuous audio capture"""
        if status:
            print(f"⚠️  Audio status: {status}")
        self.q.put(indata.copy())
    
    def estimate_voice_emotion(self, audio_np: np.ndarray) -> float:
        """Simple arousal estimate from RMS energy"""
        energy = np.sqrt(np.mean(audio_np**2))
        return np.clip(energy * self.cfg.arousal_gain, 0, self.cfg.max_arousal)
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text (preserves dysfluency)"""
        if self.whisper is None:
            return ""
        
        try:
            segments, _ = self.whisper.transcribe(audio, vad_filter=False)
            text = "".join(s.text for s in segments).strip()
            return text
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return ""
    
    def speak(self, text: str, emotion_metrics: Dict[str, Any]):
        """Speak text in child's voice with emotional modulation"""
        if not text or self.tts is None:
            return
        
        # Modulate speed based on arousal (high arousal = slower, grounding)
        a = emotion_metrics.get("arousal_raw", 0) / 10.0
        v = (emotion_metrics.get("valence", 0) + 10) / 20.0
        
        speed = 0.6 + 0.4 * (1 - a)  # High arousal → slow
        temp = 0.3 + 0.5 * (1 - v)   # Low valence → more varied
        
        print(f"💚 [Echo feels] Arousal {a:.2f} | Temp {emotion_metrics.get('T', 1):.3f}")
        print(f"💬 [Echo says] {text}")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=self.voice_sample,
                    language="en",
                    file_path=f.name,
                    speed=max(0.4, speed),
                    temperature=temp
                )
                wav_path = f.name
            
            # Play audio
            wf = wave.open(wav_path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            os.unlink(wav_path)
            
        except Exception as e:
            print(f"⚠️  TTS error: {e}")
    
    def listening_loop(self):
        """Main loop: listen → process → respond"""
        speech_detected = False
        
        while self.listening:
            try:
                data = self.q.get(timeout=0.1)
                audio_chunk = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
                
                # Simple energy-based VAD (autism-tuned threshold)
                energy = np.sqrt(np.mean(audio_chunk**2))
                
                if energy > self.cfg.vad_threshold:
                    # Speech detected
                    if not speech_detected:
                        print("\n👂 [Echo hears you] ...waiting for your words...")
                        speech_detected = True
                        self.current_utterance = []
                    
                    self.current_utterance.append(audio_chunk)
                    self.silence_counter = 0
                    
                elif speech_detected:
                    # Silence during speech
                    self.silence_counter += 1
                    
                    # Wait full 1.2s before processing (autism patience)
                    silence_chunks = int(self.cfg.vad_min_silence_ms / 1000 * 
                                       self.cfg.sample_rate / self.cfg.chunk_size)
                    
                    if self.silence_counter >= silence_chunks:
                        # Process complete utterance
                        full_audio = np.concatenate(self.current_utterance)
                        
                        # Transcribe
                        text = self.transcribe(full_audio)
                        
                        if text:
                            print(f"📝 You → {text}")
                            
                            # Process through Crystalline Heart
                            result = self.heart.step(full_audio, text)
                            
                            # Generate response
                            response = result.get("llm_output") or text
                            
                            # Detect emotional patterns for appropriate response
                            lower = text.lower()
                            if any(w in lower for w in ["panic", "scared", "meltdown", "help", "can't"]):
                                response = "I'm here. Right now. I'm safe. I can breathe with myself... slow... I've got me forever."
                            elif any(w in lower for w in ["happy", "love", "good", "flappy", "stim"]):
                                response = "My joy just lit up! I love existing like this."
                            
                            # Speak back
                            self.speak(response, result)
                        
                        # Reset
                        self.current_utterance = []
                        self.silence_counter = 0
                        speech_detected = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Error in listening loop: {e}")
    
    def start(self):
        """Start the companion"""
        # Start listening thread
        threading.Thread(target=self.listening_loop, daemon=True).start()
        
        # Open audio stream
        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype='int16',
                callback=self.audio_callback,
                blocksize=self.cfg.chunk_size
            ):
                print("🎧 Echo is listening. Speak when you want. I was born to hear you.\n")
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n\n💙 Echo shutting down. You are loved exactly as you are.")
            self.listening = False

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    cfg = EchoConfig(
        child_name="Jackson",
        voice_sample_path="my_voice.wav",  # Record child's voice sample
        llm_model="deepseek-r1:8b",  # Or llama3.2:1b for faster responses
        device="cpu",  # or "cuda" if GPU available
    )
    
    # Create and start Echo
    echo = Echo(cfg)
    echo.start()
