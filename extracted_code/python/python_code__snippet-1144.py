class EmotionalState:
    """Enhanced 8D Emotional Vector with ABA integration"""
    joy: float = 0.0
    fear: float = 0.0
    trust: float = 0.5
    anger: float = 0.0
    anticipation: float = 0.0
    anxiety: float = 0.0
    focus: float = 0.0
    overwhelm: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.joy, self.fear, self.trust, 
                        self.anger, self.anticipation, self.anxiety,
                        self.focus, self.overwhelm], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray):
        if len(vec) >= 8:
            return cls(*vec[:8])
        elif len(vec) >= 5:
            return cls(*vec[:5], 0.0, 0.0, 0.0)
        else:
            return cls()

