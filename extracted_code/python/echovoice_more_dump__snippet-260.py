class GradientFlow:
    """Optimizes the continuous (vector) part of the state."""
    def __init__(self, hamiltonian: SemanticHamiltonian, lr: float = 0.1) -> None:
        self.ham = hamiltonian
        self.lr = float(lr)

    def step(self, state: HybridState, dt: float) -> None:
        grads = self.ham.analytic_gradient(state)
        for n, g in grads.items():
            arr = np.asarray(state.x[n]).astype(float)
            state.x[n] = (arr - dt * self.lr * g).astype(float)

class MetropolisEngine:
    """Optimizes the discrete (bit-string) part of the state via simulated annealing."""
    def __init__(self, hamiltonian: SemanticHamiltonian, anneal_fn: Callable[[int], float]) -> None:
        self.ham = hamiltonian
        self.anneal_fn = anneal_fn
        self.t: int = 0

    def step(self, state: HybridState) -> None:
        T = float(self.anneal_fn(self.t))
        node_id = random.choice(list(state.E.keys()))
        bit_dim = int(np.asarray(state.E[node_id]).size)
        bit_idx = random.randrange(bit_dim)
        
        delta = self.ham.delta_energy_for_bitflip(state, node_id, bit_idx)
        
        accept_prob = min(1.0, math.exp(-delta / (T + 1e-12)))
        if random.random() < accept_prob:
            arr = np.asarray(state.E[node_id]).astype(int).copy()
            arr[bit_idx] = 1 - int(arr[bit_idx])
            state.E[node_id] = arr
            
        self.t += 1
        
