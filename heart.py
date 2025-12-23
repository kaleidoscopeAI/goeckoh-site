import numpy as np

class CrystallineHeart:
    def __init__(self):
        self.state = np.zeros(64, dtype=np.float32)
        self.gcl = 1.0

    def pulse(self, energy: float, latency: float):
        # Physics: Latency/Noise introduce Entropy
        entropy = (energy * 3.5) + (max(0, latency - 300) / 1000.0)
        
        if entropy > 0.05:
            perturb = np.random.normal(0, entropy, self.state.shape)
            self.state += perturb.astype(np.float32)
        
        # Self-Organization (Healing)
        self.state *= 0.95
        
        # Calculate Metric
        variance = float(np.var(self.state))
        self.gcl = float(np.clip(1.0 / (1.0 + (variance * 10.0)), 0.0, 1.0))
        
        return self.gcl, float(np.std(self.state))
