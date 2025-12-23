"""Enhanced emotional state with quantum coherence"""
valence: float  # -1 (negative) to +1 (positive)
arousal: float  # 0 (calm) to 1 (excited) 
coherence: float  # 0 (chaotic) to 1 (ordered)
phase: complex  # Quantum phase factor

def to_vector(self) -> np.ndarray:
    return np.array([self.valence, self.arousal, self.coherence, self.phase.real, self.phase.imag])

