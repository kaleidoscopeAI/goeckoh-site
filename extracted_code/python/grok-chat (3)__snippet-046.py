def __init__(self, n_nodes=1024):
    super().__init__()
    self.n = n_nodes
    self.E = torch.randn(n_nodes, requires_grad=True) * 0.1
    self.W = torch.eye(n_nodes) * 0.1 + torch.randn(n_nodes, n_nodes) * 0.01
    self.optimizer = torch.optim.Adam([self.E], lr=0.02)

def forward(self, external_input: float):
    # Equation 11 — Emotional ODE integration (Euler step)
    dE = -0.8 * self.E + 0.3 * (self.W @ self.E) + external_input
    self.E.data += 0.02 * dE
    self.E.data = torch.tanh(self.E.data)  # bounded [-1,1]
    return self.optimizer.step()
    self.optimizer.zero_grad()
    gcl = float(torch.mean(torch.abs(self.E)).item())
    return gcl  # Global Coherence Level

