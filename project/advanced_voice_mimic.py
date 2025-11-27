from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional
import random
import uuid

import numpy as np

# Optional deps
try:
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
    _HAS_AVM_DEPS = True
except Exception:
    librosa = None  # type: ignore
    sf = None  # type: ignore
    _HAS_AVM_DEPS = False

from .config import AudioSettings
from .prosody import ProsodyProfile, apply_prosody_to_tts, extract_prosody
from .voice_mimic import VoiceMimic

Style = Literal["neutral", "calm", "excited"]
Mode = Literal["outer", "inner", "coach"]


@dataclass(slots=True)
class VoiceSample:
    path: Path
    duration_s: float
    rms: float
    style: Style
    quality_score: float
    added_ts: float


@dataclass(slots=True)
class VoiceProfile:
    audio: AudioSettings
    base_dir: Path
    max_samples_per_style: int = 32
    samples: Dict[Style, List[VoiceSample]] = field(default_factory=lambda: {
        "neutral": [],
        "calm": [],
        "excited": [],
    })

    def __post_init__(self) -> None:
        if not _HAS_AVM_DEPS:
            raise ImportError("advanced_voice_mimic requires librosa and soundfile.")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load_existing()

    # ------------------- helpers -------------------
    def _style_dir(self, style: Style) -> Path:
        return self.base_dir / style

    def _compute_rms(self, wav: np.ndarray) -> float:
        if wav.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(wav))))

    def _register_sample(self, path: Path, style: Style, wav: np.ndarray, quality_score: float) -> None:
        duration = len(wav) / float(self.audio.sample_rate)
        rms = self._compute_rms(wav)
        sample = VoiceSample(
            path=path,
            duration_s=duration,
            rms=rms,
            style=style,
            quality_score=quality_score,
            added_ts=path.stat().st_mtime,
        )
        self.samples.setdefault(style, []).append(sample)
        self._prune(style)

    def load_existing(self) -> None:
        for style in ("neutral", "calm", "excited"):
            dir_path = self._style_dir(style)
            if not dir_path.exists():
                continue
            for wav_path in sorted(dir_path.glob("*.wav")):
                try:
                    data, sr = sf.read(wav_path, dtype="float32")
                except Exception:
                    continue
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.audio.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.audio.sample_rate)
                duration = len(data) / float(self.audio.sample_rate)
                rms = self._compute_rms(np.asarray(data, dtype=np.float32))
                sample = VoiceSample(
                    path=wav_path,
                    duration_s=duration,
                    rms=rms,
                    style=style,  # type: ignore[arg-type]
                    quality_score=1.0,
                    added_ts=wav_path.stat().st_mtime,
                )
                self.samples.setdefault(style, []).append(sample)

    def _prune(self, style: Style) -> None:
        if len(self.samples[style]) <= self.max_samples_per_style:
            return
        self.samples[style].sort(key=lambda s: (s.quality_score, s.added_ts), reverse=True)
        self.samples[style] = self.samples[style][: self.max_samples_per_style]

    def add_sample_from_wav(
        self,
        wav: np.ndarray,
        style: Style,
        name: Optional[str] = None,
        quality_score: float = 1.0,
    ) -> Path:
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        style_dir = self._style_dir(style)
        style_dir.mkdir(parents=True, exist_ok=True)
        suffix = name or uuid.uuid4().hex[:8]
        path = style_dir / f"{style}_{suffix}.wav"
        sf.write(path, wav, self.audio.sample_rate)
        self._register_sample(path, style, wav, quality_score)
        return path

    def pick_reference(self, style: Style = "neutral") -> Optional[VoiceSample]:
        candidates = self.samples.get(style) or []
        if candidates:
            return max(candidates, key=lambda s: s.quality_score)
        if style != "neutral" and self.samples["neutral"]:
            return max(self.samples["neutral"], key=lambda s: s.quality_score)
        for lst in self.samples.values():
            if lst:
                return max(lst, key=lambda s: s.quality_score)
        return None

    def maybe_adapt_from_attempt(
        self,
        attempt_wav: np.ndarray,
        style: Style,
        quality_score: float,
        min_quality_bootstrap: float = 0.8,
        min_quality_refine: float = 0.9,
    ) -> Optional[Path]:
        has_profile = any(self.samples.values())
        threshold = min_quality_bootstrap if not has_profile else min_quality_refine
        if quality_score < threshold:
            return None
        return self.add_sample_from_wav(attempt_wav, style, quality_score=quality_score)


@dataclass(slots=True)
class VoiceCrystalConfig:
    inner_lowpass_window_ms: float = 18.0
    inner_volume_scale: float = 0.45
    coach_volume_scale: float = 1.15
    prosody_strength_pitch: float = 1.0
    prosody_strength_energy: float = 1.0
    sample_rate: int = 16_000


@dataclass(slots=True)
class VoiceCrystal:
    tts: VoiceMimic
    audio: AudioSettings
    profile: VoiceProfile
    config: VoiceCrystalConfig = field(default_factory=VoiceCrystalConfig)

    def _smooth(self, wav: np.ndarray, window_ms: float) -> np.ndarray:
        if wav.size == 0:
            return wav
        window = max(int(window_ms * self.audio.sample_rate / 1000.0), 1)
        if window <= 1:
            return wav
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(wav, kernel, mode="same").astype(np.float32)

    def _apply_mode(self, wav: np.ndarray, mode: Mode) -> np.ndarray:
        if wav.size == 0:
            return wav
        if mode == "inner":
            smoothed = self._smooth(wav, self.config.inner_lowpass_window_ms * 2.0)
            rms = float(np.sqrt(np.mean(smoothed**2) + 1e-6))
            target_rms = rms * 0.8
            if rms > 0:
                smoothed = np.tanh(smoothed / rms) * target_rms
            return (smoothed * self.config.inner_volume_scale).astype(np.float32)
        if mode == "coach":
            return (wav * self.config.coach_volume_scale).astype(np.float32)
        return wav

    def _synthesize_raw(self, text: str, style: Style) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)
        sample = self.profile.pick_reference(style)
        if sample:
            self.tts.update_voiceprint(sample.path)
        return self.tts.synthesize(text)

    def _apply_prosody(
        self,
        wav: np.ndarray,
        prosody_source_wav: Optional[np.ndarray],
        prosody_source_sr: Optional[int],
    ) -> np.ndarray:
        if prosody_source_wav is None or prosody_source_sr is None:
            return wav
        try:
            prosody = extract_prosody(prosody_source_wav, prosody_source_sr)
            return apply_prosody_to_tts(
                wav,
                self.audio.sample_rate,
                prosody,
                strength_pitch=self.config.prosody_strength_pitch,
                strength_energy=self.config.prosody_strength_energy,
            )
        except Exception:
            return wav

    def speak(
        self,
        text: str,
        style: Style = "neutral",
        mode: Mode = "outer",
        prosody_source_wav: Optional[np.ndarray] = None,
        prosody_source_sr: Optional[int] = None,
    ) -> np.ndarray:
        base = self._synthesize_raw(text, style)
        if base.size == 0:
            return base
        prosody_applied = self._apply_prosody(base, prosody_source_wav, prosody_source_sr)
        processed = self._apply_mode(prosody_applied, mode)
        return processed

    def say_inner(
        self,
        text: str,
        style: Style = "calm",
        prosody_source_wav: Optional[np.ndarray] = None,
        prosody_source_sr: Optional[int] = None,
    ) -> np.ndarray:
        return self.speak(
            text=text,
            style=style,
            mode="inner",
            prosody_source_wav=prosody_source_wav,
            prosody_source_sr=prosody_source_sr,
        )

    def say_outer(
        self,
        text: str,
        style: Style = "neutral",
        prosody_source_wav: Optional[np.ndarray] = None,
        prosody_source_sr: Optional[int] = None,
    ) -> np.ndarray:
        return self.speak(
            text=text,
            style=style,
            mode="outer",
            prosody_source_wav=prosody_source_wav,
            prosody_source_sr=prosody_source_sr,
        )

    def say_coach(
        self,
        text: str,
        style: Style = "excited",
        prosody_source_wav: Optional[np.ndarray] = None,
        prosody_source_sr: Optional[int] = None,
    ) -> np.ndarray:
        return self.speak(
            text=text,
            style=style,
            mode="coach",
            prosody_source_wav=prosody_source_wav,
            prosody_source_sr=prosody_source_sr,
        )
