"""
Voice profile data structures for the Cloning Bubble.
Defines the Bubble DNA (Θᵤ) that constrains synthesis and visualization.

goeckoh/psychoacoustic_engine/voice_profile.py
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VoiceFingerprint:
    """
    Static Bubble Constraints Θ_u for one child.
    
    This is the "DNA" of the Cloning Bubble - the psychoacoustic
    constraints that define how this child's voice should sound and look.
    """
    mu_f0: float              # Median pitch (Hz) - voice color
    sigma_f0: float           # Pitch variability (Hz) - expressivity range
    base_roughness: float     # 0..1, from HNR (breathy vs clean)
    base_metalness: float     # 0..1, from spectral tilt (soft vs bright)
    base_sharpness: float     # 0..1, from ZCR (Bouba vs Kiki baseline)
    rate: float               # Syllables per second (idle heartbeat / tempo)
    jitter_base: float        # Micro-variation in pitch for realism
    shimmer_base: float       # Micro-variation in energy for realism
    base_radius: float = 1.0  # Default bubble size


@dataclass
class SpeakerProfile:
    """
    Complete Cloning Bubble identity.
    
    Combines the fingerprint (Θᵤ) with an optional neural embedding
    that captures the unique tone-color of the child's voice.
    """
    user_id: str
    fingerprint: VoiceFingerprint
    embedding: np.ndarray = None  # Optional 1D float32 vector from voice encoder


# Alias for backward compatibility with bubble_foam.py
VoiceProfile = SpeakerProfile
