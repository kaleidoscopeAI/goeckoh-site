from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import librosa
import numpy as np
import soundfile as sf

from core.settings import AudioSettings
from voice.mimic import VoiceMimic
from voice.profile import VoiceProfile
from voice.prosody import ProsodyProfile, extract_prosody
from loop.decision import AgentDecision, Mode


@dataclass(slots=True)
class AudioData:
    """Raw audio data."""

    waveform: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class Information:
    """Generic wrapper for information passing between gears."""

    payload: AudioData
    source_gear: str
    metadata: dict = field(default_factory=dict)


def _interp_to_num_frames(src: np.ndarray, num_frames: int) -> np.ndarray:
    if src.size == 0:
        return np.zeros(num_frames, dtype=np.float32)
    if src.size == num_frames:
        return src.astype(np.float32)
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

    frame_length = max(
        int(tts_sample_rate * (prosody.frame_length / prosody.sample_rate)), 256
    )
    hop_length = max(
        int(tts_sample_rate * (prosody.hop_length / prosody.sample_rate)), 128
    )

    num_frames = 1 + max(0, (len(tts_wav) - frame_length) // hop_length)
    if num_frames <= 0:
        return tts_wav

    f0_child = _interp_to_num_frames(prosody.f0_hz, num_frames)
    energy_child = _interp_to_num_frames(prosody.energy, num_frames)

    # Baseline F0 of TTS
    f0_tts, _, _ = librosa.pyin(
        tts_wav,
        fmin=80.0,
        fmax=600.0,
        sr=tts_sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if np.any(np.isfinite(f0_tts)):
        base_f0 = np.nanmedian(f0_tts)
    else:
        base_f0 = np.median(f0_child)

    out = np.zeros(len(tts_wav) + frame_length, dtype=np.float32)
    window = np.hanning(frame_length).astype(np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if start >= len(tts_wav):
            break

        frame = tts_wav[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")

        target_f0 = float(f0_child[i])
        pitch_ratio = (target_f0 / base_f0) if base_f0 > 1 else 1.0
        pitch_ratio = (pitch_ratio - 1.0) * strength_pitch + 1.0
        n_steps = 12.0 * np.log2(max(pitch_ratio, 1e-3))

        shifted = librosa.effects.pitch_shift(
            y=frame, sr=tts_sample_rate, n_steps=n_steps
        )

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

    def express(
        self, decision: AgentDecision, audio_info: Optional[Information]
    ) -> Optional[Information]:
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
                final_wav = apply_prosody_to_tts(
                    base_wav, self.audio_cfg.sample_rate, prosody
                )
            except Exception as e:
                print(f"Warning: Prosody transfer failed. Using base TTS. Error: {e}")

        # 4. Apply mode-specific acoustics
        final_wav = self._apply_mode_acoustics(final_wav, decision.mode)

        return Information(
            payload=AudioData(
                waveform=final_wav, sample_rate=self.audio_cfg.sample_rate
            ),
            source_gear="ExpressionGear",
            metadata={"decision": decision},
        )
