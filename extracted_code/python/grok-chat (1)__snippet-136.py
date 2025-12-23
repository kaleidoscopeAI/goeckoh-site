n_nodes: int = 256
bit_dim: int = 128
seed: Optional[int] = None

alpha_drive: float = 1.0
beta_decay: float = 0.5
gamma_coupling: float = 0.3
T0: float = 1.0
alpha_T: float = 0.01

bits: np.ndarray = field(init=False)
positions: np.ndarray = field(init=False)
emotions: np.ndarray = field(init=False)
step_count: int = field(init=False, default=1)

def __post_init__(self):
    if self.seed is not None:
        np.random.seed(self.seed)

    self.bits = np.random.randint(0, 2, size=(self.n_nodes, self.bit_dim)).astype(np.float32)

    self.positions = 0.1 * np.random.randn(self.n_nodes, 3).astype(np.float32)

    self.emotions = np.zeros((self.n_nodes, 5), dtype=np.float32)

@property
def temperature(self) -> float:
    t = max(self.step_count, 1)
    denom = math.log1p(self.alpha_T * t)
    if denom <= 0:
        denom = 1e-3
    return float(self.T0 / denom)

def _bond_weights(self) -> np.ndarray:
    sigma = 1.0

    diff_pos = self.positions[:, None, :] - self.positions[None, :, :]

    dist2 = np.sum(diff_pos**2, axis=2)

    spatial = np.exp(-dist2 / (2 * sigma**2))

    diff_bits = self.bits[:, None, :] - self.bits[None, :, :]

    hamming = np.sum(np.abs(diff_bits), axis=2) / self.bit_dim

    w = spatial * (1 - hamming)

    row_sums = w.sum(axis=1, keepdims=True) + 1e-8

    B = w / row_sums

    return B

def _tension(self, B: np.ndarray) -> np.ndarray:
    torch = None  # assume no torch for now
    # Placeholder for tension calculation
    return np.zeros(self.n_nodes, dtype=np.float32)

def _emotional_ode_step(self, B: np.ndarray, input_stimulus: Optional[np.ndarray], dt: float) -> None:
    E = self.emotions

    alpha = 1.0
    beta = 0.5
    gamma = 0.3

    I = input_stimulus if input_stimulus is not None else np.zeros_like(E)

    drive = alpha * I
    decay = -beta * E
    diffusion = gamma * (B @ E - E)

    T = self.temperature
    noise = np.random.normal(0, T, size=E.shape) * 0.1

    dE = drive + decay + diffusion + noise

    self.emotions = E + dt * dE

def step(self, input_stimulus: Optional[np.ndarray] = None, dt: float = 0.02) -> Dict[str, Any]:
    B = self._bond_weights()
    total_tension = self._tension(B)
    self._emotional_ode_step(B, input_stimulus, dt)

    awareness = total_tension  # placeholder

    GCL = float(awareness.mean())
    T = self.temperature

    crit = float(np.var(awareness) * (1.0 + T))

    self.step_count += 1

    metrics = {
        "step": self.step_count,
        "temperature": T,
        "global_coherence": GCL,
        "criticality_proxy": crit,
        "mean_arousal": float(self.emotions[:, 0].mean()),
        "mean_valence": float(self.emotions[:, 1].mean()),
        "mean_awareness": float(awareness.mean()),
    }
    return metrics

