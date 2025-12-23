"""Optimizes the continuous (vector) part of the state."""
def __init__(self, hamiltonian: SemanticHamiltonian, lr: float = 0.1) -> None:
    self.ham = hamiltonian
    self.lr = float(lr)

def step(self, state: HybridState, dt: float) -> None:
    grads = self.ham.analytic_gradient(state)
    for n, g in grads.items():
        arr = np.asarray(state.x[n]).astype(float)
        state.x[n] = (arr - dt * self.lr * g).astype(float)

