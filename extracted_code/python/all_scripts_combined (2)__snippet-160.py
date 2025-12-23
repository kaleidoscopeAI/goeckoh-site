def __init__(self, n_nodes: int):
    self.n = n_nodes
    self.b = np.zeros(n_nodes)
    self.h = np.zeros(n_nodes)
    self.kappa = np.zeros(n_nodes)
    self.mu = np.zeros(n_nodes)
    # stateful noise seeds for reproducibility
    self._rng = np.random.RandomState(42)

def step(self, relational: RelationalMatrix, inputs: np.ndarray, dt: float = 0.1):
    n = self.n
    if inputs is None:
        inputs = np.zeros(n)
    # coupling matrix from relational matrix (n x n)
    R = relational.R
    affin = np.real(R @ R.conj().T)
    maxval = np.max(np.abs(affin)) if np.max(np.abs(affin)) > 0 else 1.0
    W = affin / maxval

    # perspective
    db = 0.12 * (inputs * np.tanh(inputs)) - 0.05 * self.b + 0.03 * (W @ self.b - np.sum(W, axis=1) * self.b)
    self.b += db * dt

    # speculation with structured stochasticity
    eps = self._rng.randn(n) * 0.02
    dh = 0.10 * (inputs + eps) - 0.06 * self.h + 0.03 * (W @ self.h - np.sum(W, axis=1) * self.h)
    self.h += dh * dt

    # kaleidoscope
    dk = 0.08 * (self.b + 0.5 * self.h) - 0.04 * self.kappa + 0.02 * (W @ self.kappa - np.sum(W, axis=1) * self.kappa)
    self.kappa += dk * dt

    # mirror
    mismatch = np.abs(self.b - np.mean(self.b))
    dmu = -0.07 * mismatch + 0.05 * np.std(self.h) + 0.03 * (W @ self.mu - np.sum(W, axis=1) * self.mu)
    self.mu += dmu * dt

    # clip numeric stability
    for arr in (self.b, self.h, self.kappa, self.mu):
        np.clip(arr, -12.0, 12.0, out=arr)

