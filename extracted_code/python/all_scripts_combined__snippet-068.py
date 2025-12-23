class EmotionData:
    """Payload for emotional state information from the Heart."""
    arousal: float
    valence: float
    coherence: float
    temperature: float
    raw_emotions: np.ndarray # The full [N, 5] tensor

