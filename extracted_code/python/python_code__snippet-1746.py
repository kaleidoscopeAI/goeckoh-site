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
