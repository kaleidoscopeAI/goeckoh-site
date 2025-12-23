"""
expression_gears.py

This module is the "expression" system of the organism. It's the high-level
gear responsible for converting an `AgentDecision` into audible speech.

It orchestrates several lower-level components:
- `voice.py` (VoiceMimic): The actual TTS engine.
- `emotion.py` (extract_prosody): To analyze the user's voice for prosody transfer.
- `VoiceProfile`: To manage and select from a library of the user's own
  voice samples ("Voice Crystal").

The `ExpressionGear` can generate speech in different "modes" (e.g., inner,
outer) and applies the user's own pitch and energy to the synthesized
voice, creating a more natural and empathetic "echo."
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import random
from typing import Dict, List, Literal, Optional
import uuid

from .config import AudioSettings
from .voice import VoiceMimic
from .emotion import extract_prosody, ProsodyProfile
from .gears import AgentDecision, Information, AudioData

Mode = Literal["outer", "inner", "coach"]

@dataclass(slots=True)
class VoiceSample:
    """A single recorded sample of the user's voice."""
    path: Path
    duration_s: float
    rms: float
    quality_score: float = 1.0

@dataclass
class VoiceProfile:
    """Manages the collection of the user's voice samples (the "Voice Crystal")."""
    audio_cfg: AudioSettings
    base_dir: Path
    samples: List[VoiceSample] = field(default_factory=list)
    max_samples: int = 50

    def __post_init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load_existing()

    def load_existing(self):
        """Load all .wav files from the voice directory."""
        for wav_path in sorted(self.base_dir.glob("**/*.wav")):
            try:
                data, sr = sf.read(wav_path, dtype="float32")
                if sr != self.audio_cfg.sample_rate:
                    data = librosa.resample(y=data, orig_sr=sr, target_sr=self.audio_cfg.sample_rate)
                
                duration = len(data) / float(self.audio_cfg.sample_rate)
                rms = float(np.sqrt(np.mean(np.square(data)))) if data.size > 0 else 0.0
                # Quality score can be stored in filename or a metadata file in future
                self.samples.append(VoiceSample(wav_path, duration, rms, 1.0))
            except Exception as e:
                print(f"Failed to load voice sample {wav_path}: {e}")

    def add_sample(self, wav: np.ndarray, quality_score: float) -> Optional[Path]:
        """Adds a new voice sample to the profile if quality is high enough."""
        if quality_score < 0.9:
            return None
        
        path = self.base_dir / f"sample_{int(random.random() * 1e8)}.wav"
        sf.write(path, wav, self.audio_cfg.sample_rate)
        
        duration = len(wav) / float(self.audio_cfg.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(wav))))
        self.samples.append(VoiceSample(path, duration, rms, quality_score))
        
        self._prune()
        return path

    def _prune(self):
        """Keeps only the highest quality samples if count exceeds max_samples."""
        if len(self.samples) > self.max_samples:
            self.samples.sort(key=lambda s: s.quality_score, reverse=True)
            self.samples = self.samples[:self.max_samples]

    def pick_reference(self) -> Optional[Path]:
        """Picks a random, high-quality sample to use for voice cloning."""
        if not self.samples:
            return None
        # Prefer higher quality samples
        high_quality_samples = [s for s in self.samples if s.quality_score > 0.95]
        if high_quality_samples:
            return random.choice(high_quality_samples).path
        return random.choice(self.samples).path

    def maybe_adapt_from_attempt(
        self,
        attempt_wav: np.ndarray,
        style: Literal["neutral", "calm", "excited"] = "neutral",
        quality_score: float = 0.0,
        min_quality: float = 0.8,
    ) -> Optional[Path]:
        """Add a new facet when an attempt is strong enough."""
        if quality_score < min_quality:
            return None
        style_dir = self.base_dir / style
        style_dir.mkdir(parents=True, exist_ok=True)
        path = style_dir / f"{style}_{uuid.uuid4().hex[:8]}.wav"
        sf.write(path, attempt_wav, self.audio_cfg.sample_rate)
        duration = len(attempt_wav) / float(self.audio_cfg.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(attempt_wav)) + 1e-8))
        self.samples.append(VoiceSample(path, duration, rms, quality_score))
        self._prune()
        return path


def _interp_to_num_frames(src: np.ndarray, num_frames: int) -> np.ndarray:
    if src.size == 0: return np.zeros(num_frames, dtype=np.float32)
    if src.size == num_frames: return src.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=src.size)
    x_new = np.linspace(0.0, 1.0, num=num_frames)
    return np.interp(x_new, x_old, src).astype(np.float32)

def apply_prosody_to_tts(
    tts_wav: np.ndarray,
    tts_sample_rate: int,
    prosody: ProsodyProfile,
    strength_pitch: float = 1.0,
    strength_energy: float = 1.0,
) -> np.ndarray:
    """Apply child's prosody onto synthesized waveform (overlap-add)."""
    if tts_wav.ndim > 1:
        tts_wav = np.mean(tts_wav, axis=1)
    tts_wav = np.asarray(tts_wav, dtype=np.float32)

    frame_length = max(int(tts_sample_rate * (prosody.frame_length / prosody.sample_rate)), 256)
    hop_length = max(int(tts_sample_rate * (prosody.hop_length / prosody.sample_rate)), 128)
    
    num_frames = 1 + max(0, (len(tts_wav) - frame_length) // hop_length)
    if num_frames <= 0: return tts_wav
    
    f0_child = _interp_to_num_frames(prosody.f0_hz, num_frames)
    energy_child = _interp_to_num_frames(prosody.energy, num_frames)
    
    # Baseline F0 of TTS
    f0_tts, _, _ = librosa.pyin(tts_wav, fmin=80.0, fmax=600.0, sr=tts_sample_rate, frame_length=frame_length, hop_length=hop_length)
    if np.any(np.isfinite(f0_tts)):
        base_f0 = np.nanmedian(f0_tts)
    else:
        base_f0 = np.median(f0_child)

    out = np.zeros(len(tts_wav) + frame_length, dtype=np.float32)
    window = np.hanning(frame_length).astype(np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if start >= len(tts_wav): break
        
        frame = tts_wav[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")
            
        target_f0 = float(f0_child[i])
        pitch_ratio = (target_f0 / base_f0) if base_f0 > 1 else 1.0
        pitch_ratio = (pitch_ratio - 1.0) * strength_pitch + 1.0
        n_steps = 12.0 * np.log2(max(pitch_ratio, 1e-3))
        
        shifted = librosa.effects.pitch_shift(y=frame, sr=tts_sample_rate, n_steps=n_steps)
            
        # Energy match
        frame_rms = np.sqrt(np.mean(shifted**2) + 1e-6)
        target_rms = float(energy_child[i])
        ratio = (target_rms / frame_rms)
        ratio = (ratio - 1.0) * strength_energy + 1.0
        shifted *= ratio
        
        out[start:end] += shifted * window

    max_val = np.max(np.abs(out)) + 1e-6
    return (out / max_val).astype(np.float32)


@dataclass
class ExpressionGear:
    """High-level gear for generating expressive speech."""
    tts_engine: VoiceMimic
    audio_cfg: AudioSettings
    voice_profile: VoiceProfile
    inner_volume_scale: float = 0.5
    coach_volume_scale: float = 1.1

    def _apply_mode_acoustics(self, wav: np.ndarray, mode: Mode) -> np.ndarray:
        """Applies acoustic effects based on the speech mode."""
        if mode == "inner":
            return (wav * self.inner_volume_scale).astype(np.float32)
        elif mode == "coach":
            return (wav * self.coach_volume_scale).astype(np.float32)
        return wav

    def express(self, decision: AgentDecision, audio_info: Optional[Information]) -> Optional[Information]:
        """
        Generates speech based on an agent's decision.
        Returns an Information object with the final audio, or None.
        """
        text_to_speak = decision.target_text
        if not text_to_speak:
            return None

        # 1. Select a voice reference for cloning
        ref_path = self.voice_profile.pick_reference()
        if ref_path:
            self.tts_engine.update_voiceprint(ref_path)
        
        # 2. Synthesize the base waveform
        base_wav = self.tts_engine.synthesize(text_to_speak)
        if base_wav.size == 0:
            return None

        # 3. Apply prosody transfer if a source is available
        final_wav = base_wav
        if audio_info and isinstance(audio_info.payload, AudioData):
            prosody_source_wav = audio_info.payload.waveform
            prosody_source_sr = audio_info.payload.sample_rate
            try:
                prosody = extract_prosody(prosody_source_wav, prosody_source_sr)
                final_wav = apply_prosody_to_tts(base_wav, self.audio_cfg.sample_rate, prosody)
            except Exception as e:
                print(f"Warning: Prosody transfer failed. Using base TTS. Error: {e}")

        # 4. Apply mode-specific acoustics
        final_wav = self._apply_mode_acoustics(final_wav, decision.mode)

        return Information(
            payload=AudioData(waveform=final_wav, sample_rate=self.audio_cfg.sample_rate),
            source_gear="ExpressionGear",
            metadata={"decision": decision}
        )