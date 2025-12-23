from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf


@dataclass(slots=True)
class ProsodyFeatures:
    f0_hz: np.ndarray       # shape [T]
    rms: np.ndarray         # shape [T]
    time_s: np.ndarray      # shape [T]
    f0_median: float
    rms_mean: float
    duration_s: float


def _extract_from_float(y: np.ndarray, sr: int) -> ProsodyFeatures:
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)

    n_fft = 1024
    hop_length = 256

    f0 = librosa.yin(
        y,
        fmin=50.0,
        fmax=400.0,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    frames = np.arange(len(f0))
    time_s = frames * hop_length / float(sr)

    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    if f0_valid.size == 0:
        f0_median = 0.0
    else:
        f0_median = float(np.median(f0_valid))

    rms_mean = float(np.mean(rms))
    duration_s = float(len(y) / float(sr))

    return ProsodyFeatures(
        f0_hz=f0,
        rms=rms,
        time_s=time_s,
        f0_median=f0_median,
        rms_mean=rms_mean,
        duration_s=duration_s,
    )


def extract_prosody_from_int16(audio_int16: np.ndarray, sr: int) -> ProsodyFeatures:
    y = audio_int16.astype(np.float32) / 32768.0
    return _extract_from_float(y, sr)


def _safe_ratio(a: float, b: float, default: float = 1.0) -> float:
    if b <= 0 or a <= 0:
        return default
    return a / b


def apply_prosody_to_tts(
    tts_wav_path: Path,
    target: ProsodyFeatures,
    pitch_limit_semitones: float = 5.0,
    tempo_limit_ratio: float = 0.5,  # max +/- 50% change
    loudness_limit_ratio: float = 2.0,
) -> None:
    """
    Match TTS audio's global pitch, tempo, and loudness to child's utterance.
    """
    if not tts_wav_path.exists():
        return

    y_tts, sr_tts = librosa.load(str(tts_wav_path), sr=None, mono=True)
    tts_prosody = _extract_from_float(y_tts, sr_tts)

    # Pitch
    if target.f0_median > 0 and tts_prosody.f0_median > 0:
        ratio = _safe_ratio(target.f0_median, tts_prosody.f0_median, default=1.0)
        n_steps = 12.0 * np.log2(ratio)
        n_steps = float(np.clip(n_steps, -pitch_limit_semitones, pitch_limit_semitones))
        if abs(n_steps) > 1e-3:
            y_tts = librosa.effects.pitch_shift(y_tts, sr_tts, n_steps=n_steps)

    # Tempo
    if target.duration_s > 0 and tts_prosody.duration_s > 0:
        dur_ratio = _safe_ratio(tts_prosody.duration_s, target.duration_s, default=1.0)
        rate = float(np.clip(dur_ratio, 1.0 - tempo_limit_ratio, 1.0 + tempo_limit_ratio))
        if abs(rate - 1.0) > 1e-3:
            y_tts = librosa.effects.time_stretch(y_tts, rate)

    # Loudness
    new_prosody = _extract_from_float(y_tts, sr_tts)
    if new_prosody.rms_mean > 0 and target.rms_mean > 0:
        amp_ratio = _safe_ratio(target.rms_mean, new_prosody.rms_mean, default=1.0)
        amp_ratio = float(
            np.clip(amp_ratio, 1.0 / loudness_limit_ratio, loudness_limit_ratio)
        )
        y_tts = y_tts * amp_ratio

    sf.write(str(tts_wav_path), y_tts, sr_tts)
