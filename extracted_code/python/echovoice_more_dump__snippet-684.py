class MetropolisEngine:
    def __init__(self, state: HybridState, hamiltonian: SemanticHamiltonian, anneal_fn: Callable[[int], float]) -> None:
        self.state = state
        self.ham = hamiltonian
        self.anneal_fn = anneal_fn
        self.t: int = 0

    def step(self) -> None:
        T = float(self.anneal_fn(self.t))
        node = random.choice(list(self.state.E.keys()))
        d = int(np.asarray(self.state.E[node]).size)
        bit_idx = random.randrange(d)
        delta = self.ham.delta_energy_for_bitflip(self.state, node, bit_idx)
        accept_prob = min(1.0, math.exp(-delta / (T + 1e-12)))
        if random.random() < accept_prob:
            arr = np.asarray(self.state.E[node]).astype(int).copy()
            arr[bit_idx] = 1 - int(arr[bit_idx])
            self.state.E[node] = arr
            # keep packed representation consistent
            if node in self.state.E_packed:
                self.state.E_packed[node] = pack_bits(self.state.E[node])
        self.t += 1

