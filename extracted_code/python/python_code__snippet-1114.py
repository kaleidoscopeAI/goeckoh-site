class EmotionalState:
    """5D Emotional Vector for EADS (Emotional Actuation Dial System)"""
    joy: float = 0.0      # Positive affect
    fear: float = 0.0      # Threat response
    trust: float = 0.5     # Safety/receptivity
    anger: float = 0.0     # Confrontation energy
    anticipation: float = 0.0  # Novelty seeking
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.joy, self.fear, self.trust, 
                        self.anger, self.anticipation], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray):
        return cls(*vec.tolist())

