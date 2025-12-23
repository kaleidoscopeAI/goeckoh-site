"""Enrollment logger to build psychoacoustic fingerprints from real audio."""

import json
import os
from dataclasses import asdict
from typing import List

import numpy as np
from scipy.signal import find_peaks

from .attempt_analysis import analyze_attempt
from .voice_profile import SpeakerProfile, VoiceFingerprint


def log_voice_characteristics(
    audio_samples: List[np.ndarray],
    sr: int,
    user_id: str,
    output_dir: str,
    speaker_embedding: np.ndarray,
) -> SpeakerProfile:
    """
    Enrollment logger using scipy/numpy only.

    Computes jitter/shimmer from audio and builds Θ_u + embedding.
    """
    if len(audio_samples) == 0:
        raise ValueError("No audio samples.")
    emb = np.asarray(speaker_embedding, dtype=np.float32).copy()
    if emb.ndim != 1:
        raise ValueError("Embedding must be 1D.")

    f0_values: List[float] = []
    hnr_values: List[float] = []
    tilt_values: List[float] = []
    zcr_values: List[float] = []
    jitter_values: List[float] = []
    shimmer_values: List[float] = []
    total_duration = 0.0
    num_syllables_est = 0

    for y in audio_samples:
        if len(y) == 0:
            continue
        feats = analyze_attempt(y, sr)

        valid_f0 = feats.f0_attempt[feats.f0_attempt > 0]
        if valid_f0.size > 1:
            f0_values.extend(valid_f0.tolist())
            jitter = np.abs(np.diff(valid_f0)) / (valid_f0[:-1] + 1e-6)
            jitter_values.extend(jitter.tolist())

        mask = feats.energy_attempt > 0.01
        if mask.any():
            hnr_values.extend(feats.hnr_attempt[mask].tolist())
            tilt_values.extend(feats.spectral_tilt[mask].tolist())
            zcr_values.extend(feats.zcr_attempt[mask].tolist())
            energy_valid = feats.energy_attempt[mask]
            if energy_valid.size > 1:
                shimmer = np.abs(np.diff(energy_valid)) / (energy_valid[:-1] + 1e-6)
                shimmer_values.extend(shimmer.tolist())

        peaks, _ = find_peaks(feats.energy_attempt, height=0.5, distance=10)
        num_syllables_est += len(peaks)
        total_duration += len(feats.energy_attempt) * feats.dt

    if not f0_values:
        mu_f0, sigma_f0 = 150.0, 20.0
    else:
        mu_f0, sigma_f0 = np.median(f0_values), np.std(f0_values)
    avg_hnr = np.mean(hnr_values) if hnr_values else 0.8
    avg_tilt = np.mean(tilt_values) if tilt_values else 0.5
    avg_zcr = np.mean(zcr_values) if zcr_values else 0.3
    jitter_base = np.mean(jitter_values) if jitter_values else 0.1
    shimmer_base = np.mean(shimmer_values) if shimmer_values else 0.1

    rate = num_syllables_est / max(total_duration, 1e-6)
    rate = np.clip(rate, 1.0, 6.0)

    fingerprint = VoiceFingerprint(
        mu_f0=float(mu_f0),
        sigma_f0=float(sigma_f0),
        base_roughness=1.0 - avg_hnr,
        base_metalness=avg_tilt,
        base_sharpness=avg_zcr,
        rate=float(rate),
        jitter_base=float(jitter_base),
        shimmer_base=float(shimmer_base),
    )

    profile = SpeakerProfile(user_id, fingerprint, emb)
    print(f"Logged profile for {user_id}: {fingerprint.to_dict()}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{user_id}_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(fingerprint), f, indent=4)
    np.save(os.path.join(output_dir, f"{user_id}_embed.npy"), emb)

    return profile
