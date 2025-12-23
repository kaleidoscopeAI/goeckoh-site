"""
Voice enrollment system - creates the Cloning Bubble from child audio.

This module analyzes multiple audio samples from a child and creates
their unique SpeakerProfile (Θᵤ + embedding) that constrains all future
synthesis and visualization.

goeckoh/psychoacoustic_engine/voice_logger.py
"""

import os
import json
from dataclasses import asdict
from typing import List

import numpy as np
from scipy import signal

from .attempt_analysis import analyze_chunk, AttemptFeatures
from .voice_profile import VoiceFingerprint, SpeakerProfile


def _detect_syllables(energy: np.ndarray, dt: float, sr: int = 22050) -> int:
    """
    Estimate syllable count from energy envelope peaks.
    
    Uses peak detection on the energy contour to approximate
    the number of syllables in the audio.
    """
    if len(energy) == 0:
        return 0
    
    # Smooth energy
    window_size = max(3, int(0.05 / dt))  # 50ms window
    if window_size % 2 == 0:
        window_size += 1
    
    if len(energy) < window_size:
        return 0
    
    smoothed = signal.savgol_filter(energy, window_size, 2)
    
    # Find peaks with minimum distance
    min_distance = max(1, int(0.15 / dt))  # Min 150ms between syllables
    threshold = 0.3 * np.max(smoothed)
    
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold:
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    
    return len(peaks)


def log_voice_characteristics(
    audio_samples: List[np.ndarray],
    sr: int,
    user_id: str,
    output_dir: str,
    speaker_embedding: np.ndarray = None,
) -> SpeakerProfile:
    """
    Enrollment: build Θ_u + embedding for one child.
    
    Analyzes multiple audio clips from a child to extract their unique
    psychoacoustic fingerprint and create their Cloning Bubble identity.
    
    Args:
        audio_samples: List of audio arrays from the child
        sr: Sample rate (Hz)
        user_id: Unique identifier for this child
        output_dir: Where to save fingerprint JSON and embedding NPY
        speaker_embedding: Pre-computed voice embedding (e.g., from OpenVoice)
    
    Returns:
        SpeakerProfile containing the complete Cloning Bubble identity
    
    Raises:
        ValueError: If audio_samples is empty or embedding is invalid
    """
    if len(audio_samples) == 0:
        raise ValueError("audio_samples must contain at least one recording.")
    
    # Handle optional embedding
    if speaker_embedding is not None:
        emb = np.asarray(speaker_embedding, dtype=np.float32).copy()
        if emb.ndim != 1:
            raise ValueError("speaker_embedding must be a 1D vector.")
    else:
        emb = None
    
    # Accumulate features across all samples
    f0_values = []
    hnr_values = []
    tilt_values = []
    zcr_values = []
    total_duration = 0.0
    num_syllables_est = 0
    
    for y in audio_samples:
        # Extract psychoacoustic features
        feats: AttemptFeatures = analyze_chunk(y, sr)
        
        # Collect valid F0 values (ignore unvoiced frames)
        valid_f0 = feats.f0_attempt[feats.f0_attempt > 0]
        if valid_f0.size > 0:
            f0_values.extend(valid_f0)
        
        # Collect features from voiced regions (energy threshold)
        mask = feats.energy_attempt > 0.01
        if mask.any():
            hnr_values.extend(feats.hnr_attempt[mask])
            tilt_values.extend(feats.spectral_tilt[mask])
            zcr_values.extend(feats.zcr_attempt[mask])
        
        # Estimate syllable count
        syllables = _detect_syllables(feats.energy_attempt, feats.dt, sr)
        num_syllables_est += syllables
        total_duration += len(feats.energy_attempt) * feats.dt
    
    # Compute fingerprint statistics
    mu_f0 = float(np.median(f0_values)) if f0_values else 150.0
    sigma_f0 = float(np.std(f0_values)) if f0_values else 20.0
    avg_hnr = float(np.mean(hnr_values)) if hnr_values else 0.8
    avg_tilt = float(np.mean(tilt_values)) if tilt_values else 0.5
    avg_zcr = float(np.mean(zcr_values)) if zcr_values else 0.3
    
    # Compute speaking rate (syllables per second)
    if total_duration <= 0.0:
        rate = 2.5
    else:
        rate = num_syllables_est / total_duration
    rate = float(np.clip(rate, 1.0, 6.0))
    
    # Create the Bubble DNA (Θᵤ)
    fingerprint = VoiceFingerprint(
        mu_f0=mu_f0,
        sigma_f0=sigma_f0,
        base_roughness=1.0 - avg_hnr,  # high HNR → low roughness
        base_metalness=avg_tilt,
        base_sharpness=avg_zcr,
        rate=rate,
        jitter_base=0.1,
        shimmer_base=0.1,
        base_radius=1.0,
    )
    
    # Create complete profile
    profile = SpeakerProfile(
        user_id=user_id,
        fingerprint=fingerprint,
        embedding=emb,
    )
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    
    fingerprint_path = os.path.join(output_dir, f"{user_id}_fingerprint.json")
    with open(fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(asdict(fingerprint), f, indent=4)
    
    if emb is not None:
        embedding_path = os.path.join(output_dir, f"{user_id}_embed.npy")
        np.save(embedding_path, emb)
    
    print(f"✓ Cloning Bubble created for {user_id}")
    print(f"  Pitch: {mu_f0:.1f} Hz ± {sigma_f0:.1f} Hz")
    print(f"  Rate: {rate:.2f} syllables/sec")
    print(f"  Roughness: {fingerprint.base_roughness:.2f}")
    print(f"  Metalness: {fingerprint.base_metalness:.2f}")
    print(f"  Sharpness: {fingerprint.base_sharpness:.2f}")
    print(f"  Saved to: {output_dir}")
    
    return profile


def load_speaker_profile(user_id: str, data_dir: str) -> SpeakerProfile:
    """
    Load a previously created Cloning Bubble from disk.
    
    Args:
        user_id: Unique identifier for the child
        data_dir: Directory containing the saved fingerprint and embedding
    
    Returns:
        SpeakerProfile loaded from disk
    """
    fingerprint_path = os.path.join(data_dir, f"{user_id}_fingerprint.json")
    embedding_path = os.path.join(data_dir, f"{user_id}_embed.npy")
    
    with open(fingerprint_path, "r", encoding="utf-8") as f:
        fp_dict = json.load(f)
    
    fingerprint = VoiceFingerprint(**fp_dict)
    embedding = np.load(embedding_path)
    
    return SpeakerProfile(
        user_id=user_id,
        fingerprint=fingerprint,
        embedding=embedding
    )
