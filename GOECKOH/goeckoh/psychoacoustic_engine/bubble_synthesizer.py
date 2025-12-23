"""Text → child-shaped control curves → audio + bubble controls."""

from typing import Dict, List, Tuple

import numpy as np

from .voice_profile import SpeakerProfile

_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_SHARP_CONSONANTS = {"K", "T", "P", "S", "Z", "SH", "CH", "F", "TH", "DH"}
_SOFT_CONSONANTS = {"M", "N", "L", "R", "W", "Y", "B", "D", "G", "JH", "V"}


def _phoneme_sharpness(phoneme: str, base_sharpness: float) -> float:
    ph = "".join(c for c in phoneme.upper() if c.isalpha())
    target = 0.1 if ph in _VOWELS else 0.9 if ph in _SHARP_CONSONANTS else 0.5 if ph in _SOFT_CONSONANTS else 0.4
    alpha = 0.6
    return np.clip(alpha * target + (1.0 - alpha) * base_sharpness, 0.0, 1.0)


class MockVocoder:
    """Minimal vocoder stub (no external dependencies)."""

    def g2p(self, text: str) -> List[str]:
        # Character-level pseudo-G2P so Bouba/Kiki responds to sharp vs smooth letters
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
        duration = max(len(energy_contour) * dt, 0.25)
        samples = int(22050 * duration)
        t = np.linspace(0.0, duration, samples, endpoint=False)
        return np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)


def feed_text_through_bubble(
    text: str,
    profile: SpeakerProfile,
    vocoder_backend=MockVocoder(),
    dt: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fp = profile.fingerprint

    phonemes = vocoder_backend.g2p(text)
    if not phonemes:
        raise ValueError("No phonemes from G2P.")

    child_duration = max(len(phonemes) / max(fp.rate, 1e-6), dt)
    num_frames = max(1, int(child_duration / dt))
    t = np.linspace(0.0, child_duration, num_frames)

    base_contour = np.clip(np.sin(np.pi * t / child_duration), 0.0, 1.0)
    target_f0 = fp.mu_f0 + base_contour * fp.sigma_f0
    jitter_pert = np.sin(t * 100) * fp.jitter_base
    target_f0 += jitter_pert * fp.sigma_f0 * 0.1

    energy = np.clip(base_contour, 0.1, 1.0)
    shimmer_pert = np.cos(t * 50) * fp.shimmer_base
    energy += shimmer_pert * 0.1
    energy = np.clip(energy, 0.1, 1.0)

    target_hnr = np.full(num_frames, 1.0 - fp.base_roughness)
    target_tilt = np.full(num_frames, fp.base_metalness)

    zcr = np.zeros(num_frames)
    frames_per_ph = max(1, num_frames // len(phonemes))
    for i, ph in enumerate(phonemes):
        start = i * frames_per_ph
        end = num_frames if i == len(phonemes) - 1 else start + frames_per_ph
        zcr[start:end] = _phoneme_sharpness(ph, fp.base_sharpness)

    audio = vocoder_backend.synthesize(
        phonemes, profile.embedding, target_f0, energy, target_hnr, target_tilt, dt
    )

    return audio, {
        "energy": energy.astype(np.float32),
        "f0": target_f0.astype(np.float32),
        "zcr": zcr.astype(np.float32),
        "hnr": target_hnr.astype(np.float32),
        "tilt": target_tilt.astype(np.float32),
        "dt": np.array([dt], dtype=np.float32),
    }
