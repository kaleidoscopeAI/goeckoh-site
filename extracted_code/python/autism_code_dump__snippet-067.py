"""
Simple ODE-based emotional lattice.
E_{t+1} = E_t + dt * (-alpha E_t + beta W E_t + gamma I)
GCL = mean(|E|).
"""

def __init__(
    self,
    n_nodes: int = 256,
    alpha: float = 0.8,
    beta: float = 0.3,
    gamma: float = 0.5,
    dt: float = 0.02,
):
    self.n_nodes = n_nodes
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.dt = dt
    self.E = torch.randn(n_nodes) * 0.1
    W = torch.eye(n_nodes) * 0.1 + torch.randn(n_nodes, n_nodes) * 0.01
    self.W = W

@torch.no_grad()
def step(self, external_input: float) -> float:
    I = torch.full_like(self.E, external_input)
    dE = -self.alpha * self.E + self.beta * (self.W @ self.E) + self.gamma * I
    self.E += self.dt * dE
    self.E = torch.tanh(self.E)
    gcl = float(torch.mean(torch.abs(self.E)).item())
    return gcl


