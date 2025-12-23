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
