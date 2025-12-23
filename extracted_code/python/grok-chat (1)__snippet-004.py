class CrystallineHeart:
    def __init__(self):
        self.lattice = np.random.choice([-1, 1], size=(CONFIG.lattice_size,) * 3)  # 3D spins
        self.temperature = 1.0

    def energy_delta(self, i, j, k):
        s = self.lattice[i, j, k]
        neighbors = sum(self.lattice[n] for n in self.get_neighbors(i, j, k))
        return 2 * CONFIG.coupling * s * neighbors

    def get_neighbors(self, i, j, k):
        coords = [(i+1, j, k), (i-1, j, k), (i, j+1, k), (i, j-1, k), (i, j, k+1), (i, j, k-1)]
        return [self.lattice[(x % CONFIG.lattice_size, y % CONFIG.lattice_size, z % CONFIG.lattice_size)] for x, y, z in coords]

    def anneal(self, stress: float):
        for _ in range(100):  # Steps per update
            i, j, k = np.random.randint(0, CONFIG.lattice_size, 3)
            delta = self.energy_delta(i, j, k)
            if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
                self.lattice[i, j, k] *= -1
        self.temperature *= 0.99
        self.lattice *= (1 - CONFIG.decay * stress)  # Decay with stress

    def get_emotion(self) -> str:  # Mimicry style from coherence
        coherence = np.mean(self.lattice)
        if coherence > 0.5: return "excited"
        elif coherence < -0.5: return "calm"
        return "neutral"

