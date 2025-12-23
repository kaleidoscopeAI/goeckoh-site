def __init__(self, state: HybridState, hamiltonian: SemanticHamiltonian, lr: float = 0.1) -> None:
    self.state = state
    self.ham = hamiltonian
    self.lr = float(lr)

def step(self, dt: float) -> None:
    grads = self.ham.analytic_gradient(self.state)
    for n, g in grads.items():
        arr = np.asarray(self.state.x[n]).astype(float)
        self.state.x[n] = (arr - dt * self.lr * g).astype(float)

