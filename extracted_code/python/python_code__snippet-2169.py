def g2p(self, text: str) -> List[str]: ...

def synthesize(
    self,
    phonemes: List[str],
    speaker_embedding: np.ndarray,
    pitch_contour: np.ndarray,
    energy_contour: np.ndarray,
    hnr_contour: np.ndarray,
    tilt_contour: np.ndarray,
    dt: float,
) -> np.ndarray: ...


