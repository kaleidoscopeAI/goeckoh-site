def __init__(self, n_system: int, n_apparatus: int):
    self.n_system = n_system
    self.n_apparatus = n_apparatus
    # complex amplitudes
    self.R = (np.random.randn(n_system, n_apparatus) + 1j * np.random.randn(n_system, n_apparatus)) * 0.02
    self.normalize_rows()

def normalize_rows(self):
    mags = np.linalg.norm(self.R, axis=1, keepdims=True)
    mags[mags == 0] = 1.0
    self.R = self.R / mags

def bidirectional_weight(self, i: int, j: int) -> complex:
    # map apparatus j back to some system index deterministically
    reverse_i = j % self.n_system
    reverse_j = i % self.n_apparatus
    return self.R[i, j] * np.conj(self.R[reverse_i, reverse_j])

def probability_for_system(self, i: int) -> float:
    weights = np.array([abs(self.bidirectional_weight(i, j)) for j in range(self.n_apparatus)])
    s = np.sum(weights)
    return float(s / (np.sum(weights) + 1e-12))

def update_hebbian(self, pre_idx: int, post_idx: int, lr: float = 1e-3):
    i = pre_idx % self.n_system
    j = post_idx % self.n_apparatus
    self.R[i, j] += lr * (1.0 + 0.05j)
    self.normalize_rows()

