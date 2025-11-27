import numpy as np

class EmotionalState:
    def __init__(self):
        self.valence = 0.5      # -1 (negative) to +1 (positive)
        self.arousal = 0.5      # 0 (calm) to 1 (excited)
        self.stance = 0.0       # -1 (defensive) to +1 (receptive)
        self.coherence = 0.8    # Emotional stability

    def update(self, data_ingested, system_state):
        """Emotional dynamics based on ingested data and system state"""
        novelty = self.calculate_novelty(data_ingested)
        stress = 1 - system_state.health  # Real: inverse health
        integration = system_state.coherence

        # Emotional update equations
        self.valence += 0.1 * (novelty - 0.5) - 0.05 * stress
        self.arousal = 0.3 * novelty + 0.7 * system_state.energy_efficiency
        self.coherence = 0.9 * self.coherence + 0.1 * integration

        # Bound emotions
        self.valence = np.clip(self.valence, -1, 1)
        self.arousal = np.clip(self.arousal, 0, 1)
        self.stance = np.clip(self.stance, -1, 1)

    def calculate_novelty(self, data_ingested):
        """Calculate novelty of data (entropy-based)"""
        if not data_ingested:
            return 0.5
        if isinstance(data_ingested, dict):
            data_ingested = list(data_ingested.values())
        counts = np.unique(data_ingested, return_counts=True)[1]
        p = counts / len(data_ingested)
        entropy = -np.sum(p * np.log2(p + 1e-10))
        return np.clip(entropy / np.log2(len(data_ingested) + 1e-10), 0, 1)

class Node:
    def __init__(self, id):
        self.id = id
        self.is_healthy = True
        self.X = np.random.randint(0,2,128)
        self.S = np.random.uniform(0,1,128)
        self.E = 1.0
        self.K = np.random.randn(128)
        self.Psi = np.random.randn(128)
        self.A = 0.5
        self.emotional_state = EmotionalState()
        self.position = np.random.rand(3)
        self.velocity = np.zeros(3)
        self.mass = 1.0
        self.bonds = []

    async def sense_emotional_state(self):
        return self.emotional_state.valence  # Real: full state
