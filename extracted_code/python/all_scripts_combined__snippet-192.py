class ProsodyProfile:
    """
    Represents the prosody of a speech segment.
    """

    f0_hz: np.ndarray
    energy: np.ndarray
    frame_length: int
    hop_length: int
    sample_rate: int


def extract_prosody(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 1024,
    hop_length: int = 256,
