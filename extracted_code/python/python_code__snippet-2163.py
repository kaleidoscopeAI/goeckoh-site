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


