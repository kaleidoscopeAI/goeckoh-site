import numpy as np

class EmotionalLattice:
    def __init__(self, nodes=1024):
        self.state = np.random.rand(nodes).astype(np.float32)
        self.arousal = 0.5
        self.valence = 0.5

    def update_from_audio(self, audio: np.ndarray):
        energy = np.mean(np.abs(audio))
        self.arousal = min(1.0, energy * 10)
        self.valence = 0.7 if energy > 0.01 else 0.3

    def get_style(self) -> str:
        if self.arousal > 0.8:
            return "calm"
        elif self.arousal < 0.3:
            return "excited"
        return "neutral"

