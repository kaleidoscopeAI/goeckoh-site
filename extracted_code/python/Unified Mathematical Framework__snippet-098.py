def __init__(self, n_nodes=1024, dim=128):
    super().__init__()
    self.n = n_nodes
    self.bits = nn.Parameter(torch.randint(0, 2, (n_nodes, dim)).float())
    self.positions = nn.Parameter(torch.randn(n_nodes, 3))
    self.emotions = nn.Parameter(torch.zeros(n_nodes, 5))  # [arousal, valence, dominance, coherence, resonance]
    self.t = torch.tensor(0.0)
    self.T0 = 1.0
    self.alpha_t = 0.01

def temperature(self):
    return self.T0 / torch.log1p(self.alpha_t * self.t)

def forward(self, external_stimulus=None):
    self.t += 1.0
    T = self.temperature()
    if external_stimulus is None:
        external_stimulus = torch.zeros_like(self.emotions)

    # Simple but faithful emotional ODEs (eq 30 + diffusion)
    decay = -0.5 * self.emotions
    noise = torch.randn_like(self.emotions) * T * 0.1
    diffusion = 0.3 * (self.emotions.mean(0) - self.emotions)
    dE = external_stimulus.mean(0) + decay + diffusion + noise
    self.emotions.data = self.emotions + 0.03 * dE
    self.emotions.data.clamp_(-10, 10)

    metrics = {
        "arousal": self.emotions[:,0].mean().item(),
        "valence": self.emotions[:,1].mean().item(),
        "temperature": T.item(),
    }
    return metrics

