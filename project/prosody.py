"""Prosody extraction and transfer helpers.

This module turns raw waveforms into pitch/energy profiles and can
optionally transfer a child's prosody onto synthesized TTS audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Optional dep: allow import without librosa, fail when functions used.
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except Exception:
    librosa = None  # type: ignore
    _HAS_LIBROSA = False


@dataclass(slots=True)
class ProsodyProfile:
    """Container for F0 and energy envelopes."""

    f0_hz: np.ndarray
    energy: np.ndarray
    times_s: np.ndarray
    frame_length: int
    hop_length: int
    sample_rate: int


def extract_prosody(
    wav: np.ndarray,
    sample_rate: int,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    fmin_hz: float = 80.0,
    fmax_hz: float = 600.0,
) -> ProsodyProfile:
    """Extract F0 and RMS energy envelopes from a waveform."""
    if not _HAS_LIBROSA:
        raise ImportError("Prosody extraction requires librosa.")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = np.asarray(wav, dtype=np.float32)

    frame_length = max(int(sample_rate * frame_ms / 1000.0), 256)
    hop_length = max(int(sample_rate * hop_ms / 1000.0), 128)

    f0, voiced_flag, _ = librosa.pyin(
        y=wav,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if np.any(voiced_flag):
        median_f0 = float(np.nanmedian(f0[voiced_flag]))
    else:
        median_f0 = 150.0
    f0 = np.nan_to_num(f0, nan=median_f0).astype(np.float32)

    rms = librosa.feature.rms(
        y=wav, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    rms = np.maximum(rms, 1e-5).astype(np.float32)

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)

    return ProsodyProfile(
        f0_hz=f0,
        energy=rms,
        times_s=times,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )


def _interp_to_num_frames(src: np.ndarray, num_frames: int) -> np.ndarray:
    if src.size == 0:
        return np.zeros(num_frames, dtype=np.float32)
    if src.size == num_frames:
        return src.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=src.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=num_frames, dtype=np.float32)
    return np.interp(x_new, x_old, src).astype(np.float32)


def apply_prosody_to_tts(
    tts_wav: np.ndarray,
    tts_sample_rate: int,
    prosody: ProsodyProfile,
    strength_pitch: float = 1.0,
    strength_energy: float = 1.0,
) -> np.ndarray:
    """Rough prosody transfer by aligning pitch and energy envelopes."""
    if tts_wav.ndim > 1:
        tts_wav = np.mean(tts_wav, axis=1)
    tts_wav = np.asarray(tts_wav, dtype=np.float32)

    frame_length = max(int(tts_sample_rate * (prosody.frame_length / prosody.sample_rate)), 256)
    hop_length = max(int(tts_sample_rate * (prosody.hop_length / prosody.sample_rate)), 128)
    num_frames = 1 + max(0, (len(tts_wav) - frame_length) // hop_length)
    if num_frames <= 0:
        return tts_wav

    f0_child = _interp_to_num_frames(prosody.f0_hz, num_frames)
    energy_child = _interp_to_num_frames(prosody.energy, num_frames)

    try:
        f0_tts = librosa.yin(
            tts_wav,
            fmin=80.0,
            fmax=600.0,
            sr=tts_sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced = f0_tts > 0
        base_f0 = float(np.median(f0_tts[voiced])) if np.any(voiced) else float(np.median(f0_child))
    except Exception:
        base_f0 = float(np.median(f0_child))

    out = np.zeros(len(tts_wav) + frame_length, dtype=np.float32)
    window = np.hanning(frame_length).astype(np.float32)
    eps = 1e-6

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if start >= len(tts_wav):
            break
        frame = tts_wav[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")

        target_f0 = float(f0_child[i])
        raw_ratio = target_f0 / base_f0 if base_f0 > 0 else 1.0
        pitch_ratio = raw_ratio ** strength_pitch
        n_steps = 12.0 * np.log2(max(pitch_ratio, 1e-3))
        try:
            shifted = librosa.effects.pitch_shift(
                frame,
                sr=tts_sample_rate,
                n_steps=n_steps,
            )
        except Exception:
            shifted = frame
        if shifted.shape[0] != frame_length:
            if shifted.shape[0] > frame_length:
                shifted = shifted[:frame_length]
            else:
                shifted = np.pad(shifted, (0, frame_length - shifted.shape[0]))

        frame_rms = float(np.sqrt(np.mean(np.square(shifted)) + eps))
        target_rms = float(energy_child[i])
        ratio = (target_rms / frame_rms) ** strength_energy if frame_rms > 0 else 1.0
        shifted *= ratio
        out[start:end] += shifted * window

    max_abs = float(np.max(np.abs(out)) + eps)
    if max_abs > 1.0:
        out = out / max_abs
    return out.astype(np.float32)
