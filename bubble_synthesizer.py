"""
Text → child-shaped control curves → audio + bubble controls.

This module ensures the TTS clone mimics the physics of the child's voice,
preventing generic TTS artifacts by constraining synthesis with Θᵤ.

goeckoh/psychoacoustic_engine/bubble_synthesizer.py
"""

from typing import Dict, List, Tuple

import numpy as np

from .voice_profile import SpeakerProfile


# Phoneme classification for Bouba/Kiki mapping
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_SHARP_CONSONANTS = {"K", "T", "P", "S", "Z", "SH", "CH", "F", "TH", "DH"}
_SOFT_CONSONANTS = {"M", "N", "L", "R", "W", "Y", "B", "D", "G", "JH", "V"}


def _phoneme_sharpness(phoneme: str, base_sharpness: float) -> float:
    """
    Map phoneme to Bouba/Kiki sharpness target.
    
    Vowels → smooth (Bouba)
    Sharp consonants → spiky (Kiki)
    Soft consonants → intermediate
    
    Blends phoneme target with child's baseline sharpness.
    """
    ph = "".join(c for c in phoneme.upper() if c.isalpha())
    
    if ph in _VOWELS:
        target = 0.1  # Smooth/round
    elif ph in _SHARP_CONSONANTS:
        target = 0.9  # Sharp/angular
    elif ph in _SOFT_CONSONANTS:
        target = 0.5  # Intermediate
    else:
        target = 0.4  # Default
    
    # Blend with child's baseline
    alpha = 0.6
    return np.clip(alpha * target + (1.0 - alpha) * base_sharpness, 0.0, 1.0)


class MockVocoder:
    """
    Minimal vocoder stub (no external dependencies).
    
    In production, replace with a real TTS backend like:
    - Coqui TTS
    - OpenVoice
    - XTTS
    - Bark
    
    The real vocoder should accept the control curves (pitch,
    energy, HNR, tilt) and speaker embedding to synthesize
    child-like speech.
    """
    
    def g2p(self, text: str) -> List[str]:
        """
        Grapheme-to-phoneme conversion.
        
        Character-level pseudo-G2P for demonstration.
        In production, use a real G2P system (e.g., phonemizer).
        """
        return [c for c in text.upper() if c.isalpha()]
    
    def synthesize(
        self,
        phonemes: List[str],
        speaker_embedding: np.ndarray,
        pitch_contour: np.ndarray,
        energy_contour: np.ndarray,
        hnr_contour: np.ndarray,
        tilt_contour: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Synthesize audio from control curves.
        
        Stub implementation returns a simple sine wave.
        Real implementation would use the embedding and contours
        to drive a neural vocoder.
        """
        duration = max(len(energy_contour) * dt, 0.25)
        samples = int(22050 * duration)
        t = np.linspace(0.0, duration, samples, endpoint=False)
        
        # Simple sine wave at median pitch
        f0_median = np.median(pitch_contour[pitch_contour > 0]) if np.any(pitch_contour > 0) else 220.0
        
        return (np.sin(2.0 * np.pi * f0_median * t) * 0.3).astype(np.float32)


def feed_text_through_bubble(
    text: str,
    profile: SpeakerProfile,
    vocoder_backend=None,
    dt: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Transform text through the Cloning Bubble.
    
    This is the core synthesis function that:
    1. Converts text to phonemes
    2. Generates child-specific control curves (F0, energy, HNR, tilt, ZCR)
    3. Passes controls + embedding to vocoder
    4. Returns synchronized audio and bubble controls
    
    Args:
        text: Input text to synthesize
        profile: Child's SpeakerProfile (Θᵤ + embedding)
        vocoder_backend: TTS backend (defaults to MockVocoder)
        dt: Time step for control curves (seconds)
    
    Returns:
        (audio, controls) where:
        - audio: Synthesized speech [N_samples]
        - controls: Dict with keys {energy, f0, zcr, hnr, tilt, dt}
                   for driving bubble visualization
    
    The controls dict can be wrapped into AttemptFeatures for
    feeding to compute_bubble_state.
    """
    if vocoder_backend is None:
        vocoder_backend = MockVocoder()
    
    fp = profile.fingerprint
    
    # Convert text to phonemes
    phonemes = vocoder_backend.g2p(text)
    if not phonemes:
        raise ValueError("No phonemes from G2P.")
    
    # Estimate duration based on child's speaking rate
    child_duration = max(len(phonemes) / max(fp.rate, 1e-6), dt)
    num_frames = max(1, int(child_duration / dt))
    t = np.linspace(0.0, child_duration, num_frames)
    
    # Generate pitch contour
    # Base contour follows natural speech prosody
    base_contour = np.clip(np.sin(np.pi * t / child_duration), 0.0, 1.0)
    target_f0 = fp.mu_f0 + base_contour * fp.sigma_f0
    
    # Add jitter (micro-variations in pitch)
    jitter_pert = np.sin(t * 100) * fp.jitter_base
    target_f0 += jitter_pert * fp.sigma_f0 * 0.1
    
    # Generate energy contour
    energy = np.clip(base_contour, 0.1, 1.0)
    
    # Add shimmer (micro-variations in energy)
    shimmer_pert = np.cos(t * 50) * fp.shimmer_base
    energy += shimmer_pert * 0.1
    energy = np.clip(energy, 0.1, 1.0)
    
    # Static HNR and tilt from child's baseline
    target_hnr = np.full(num_frames, 1.0 - fp.base_roughness)
    target_tilt = np.full(num_frames, fp.base_metalness)
    
    # Generate ZCR (Bouba/Kiki) contour from phonemes
    zcr = np.zeros(num_frames)
    frames_per_ph = max(1, num_frames // len(phonemes))
    
    for i, ph in enumerate(phonemes):
        start = i * frames_per_ph
        end = num_frames if i == len(phonemes) - 1 else start + frames_per_ph
        zcr[start:end] = _phoneme_sharpness(ph, fp.base_sharpness)
    
    # Synthesize audio using the vocoder
    audio = vocoder_backend.synthesize(
        phonemes,
        profile.embedding,
        target_f0,
        energy,
        target_hnr,
        target_tilt,
        dt
    )
    
    # Return audio and bubble controls (without 'dt' key for backward compatibility)
    controls = {
        "energy": energy.astype(np.float32),
        "f0": target_f0.astype(np.float32),
        "zcr": zcr.astype(np.float32),
        "hnr": target_hnr.astype(np.float32),
        "tilt": target_tilt.astype(np.float32),
    }
    
    return audio, controls


def controls_to_attempt_features(controls: Dict[str, np.ndarray], dt: float = 0.01):
    """
    Convert bubble controls dict to AttemptFeatures.
    
    Helper function to wrap synthesis controls into the format
    expected by compute_bubble_state.
    
    Args:
        controls: Dict from feed_text_through_bubble
        dt: Time step (default 0.01s)
    """
    from .attempt_analysis import AttemptFeatures
    
    return AttemptFeatures(
        energy_attempt=controls["energy"],
        f0_attempt=controls["f0"],
        zcr_attempt=controls["zcr"],
        spectral_tilt=controls["tilt"],
        hnr_attempt=controls["hnr"],
        dt=dt
    )
