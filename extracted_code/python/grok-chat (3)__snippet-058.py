"""
Simple ODE-like lattice whose mean |E| ≈ Global Coherence Level (GCL).
"""
def __init__(self, n_nodes: int = 1024):
    self.E = torch.randn(n_nodes) * 0.1
    # Small self-weights + small random connections
    self.W = torch.eye(n_nodes) * 0.1 + torch.randn(n_nodes, n_nodes) * 0.01
def update_and_get_gcl(self, external_input: float = 0.0) -> float:
    with torch.no_grad():
        dE = -0.8 * self.E + 0.3 * (self.W @ self.E) + external_input
        self.E += 0.02 * dE
        self.E = torch.tanh(self.E)
        gcl = torch.mean(torch.abs(self.E)).item()
    return float(gcl)
