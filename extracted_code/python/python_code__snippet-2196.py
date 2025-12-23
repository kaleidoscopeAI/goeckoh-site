"""
The emotional regulation engine
Implements coupled ODEs for affective state dynamics

dE_i/dt = α*I_i(t) - β*E_i(t) + γ*Σ w_ij(E_j - E_i)
"""

def __init__(self, num_nodes: int = 1024):
    self.num_nodes = num_nodes
    self.nodes = [CrystallineNode(i) for i in range(num_nodes)]
    self.temperature = 1.0  # Annealing temperature T(t)
    self.time_step = 0

    # ODE parameters
    self.alpha = 1.0  # Input sensitivity
    self.beta = 0.7   # Decay rate
    self.gamma = 0.4  # Diffusion coupling
    self.dt = 0.1     # Integration step

    # Initialize sparse random connectivity
    self._initialize_topology(k_neighbors=8)

def _initialize_topology(self, k_neighbors: int):
    """Create sparse random graph topology"""
    for node in self.nodes:
        # Random neighbors
        neighbor_ids = np.random.choice(
            [i for i in range(self.num_nodes) if i != node.id],
            size=min(k_neighbors, self.num_nodes - 1),
            replace=False
        )
        node.neighbors = [self.nodes[i] for i in neighbor_ids]
        node.weights = np.random.rand(len(node.neighbors)) * 0.5

def update(self, external_input: np.ndarray) -> None:
    """
    Euler step for emotional ODEs
    external_input: stimulus vector of shape (5,)
    """
    # Compute derivatives for all nodes
    derivatives = []
    for node in self.nodes:
        # Input term
        dE_input = self.alpha * external_input

        # Decay term
        dE_decay = -self.beta * node.emotion

        # Diffusion term (coupling to neighbors)
        dE_diffusion = np.zeros(5, dtype=np.float32)
        if node.neighbors:
            for neighbor, weight in zip(node.neighbors, node.weights):
                dE_diffusion += self.gamma * weight * (neighbor.emotion - node.emotion)

        # Total derivative
        dE = dE_input + dE_decay + dE_diffusion

        # Add temperature-scaled noise
        noise = np.random.randn(5) * (self.temperature * 0.01)

        derivatives.append((dE + noise) * self.dt)

    # Apply updates
    for node, dE in zip(self.nodes, derivatives):
        node.emotion += dE
        # Clip to reasonable bounds
        node.emotion = np.clip(node.emotion, -2.0, 2.0)

    self.time_step += 1

def get_global_coherence_level(self) -> float:
    """
    GCL: Global Coherence Level
    Measures synchronization across the lattice
    """
    # Variance-based coherence
    emotions = np.array([n.emotion for n in self.nodes])
    variance = np.var(emotions, axis=0).mean()

    # High variance = low coherence
    # Use sigmoid to map to [0, 1]
    coherence = 1.0 / (1.0 + variance)
    return float(np.clip(coherence, 0.0, 1.0))

def get_stress_level(self) -> float:
    """
    System stress: mean tension across all bonds
    """
    stresses = [n.compute_local_stress() for n in self.nodes]
    return float(np.mean(stresses))

def get_metrics(self) -> Dict[str, float]:
    """Export key metrics for telemetry/gating"""
    emotions = np.array([n.emotion for n in self.nodes])
    mean_emotion = emotions.mean(axis=0)

    return {
        "coherence": self.get_global_coherence_level(),
        "stress": self.get_stress_level(),
        "arousal": float(mean_emotion[0]),
        "valence": float(mean_emotion[1]),
        "confidence": float(mean_emotion[3]),
        "temperature": self.temperature,
        "time_step": self.time_step
    }

def adjust_temperature(self, meltdown_index: float):
    """
    Adjust annealing temperature based on crisis state
    T'(t) = T_base(t) + δ_T * M(t)
    """
    # Base annealing schedule: T(t) = T0 / ln(1 + αt)
    t_eff = max(self.time_step, 1)
    base_temp = 1.0 / max(np.log(1.0 + 0.01 * t_eff), 0.01)

    # Meltdown boost
    meltdown_boost = 0.8 * np.clip(meltdown_index, 0.0, 1.0)

    self.temperature = base_temp + meltdown_boost


