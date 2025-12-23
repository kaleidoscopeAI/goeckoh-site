class Information:
    """Generic wrapper for information passing between gears."""

    payload: AudioData
    source_gear: str
    metadata: dict = field(default_factory=dict)


def _interp_to_num_frames(src: np.ndarray, num_frames: int) -> np.ndarray:
    if src.size == 0:
        return np.zeros(num_frames, dtype=np.float32)
    if src.size == num_frames:
        return src.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=src.size)
    x_new = np.linspace(0.0, 1.0, num=num_frames)
    return np.interp(x_new, x_old, src).astype(np.float32)


def apply_prosody_to_tts(
    tts_wav: np.ndarray,
    tts_sample_rate: int,
    prosody: ProsodyProfile,
    strength_pitch: float = 1.0,
    strength_energy: float = 1.0,
