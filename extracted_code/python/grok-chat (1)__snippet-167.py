def __init__(self, n_nodes=1024):
    self.n_nodes = n_nodes
    self.states = [random.uniform(-1, 1) for _ in range(n_nodes)]  # Affective
    self.dt = 0.05
    self.alpha = 1.0  # Input drive
    self.beta = 0.5   # Decay
    self.gamma = 0.3  # Coupling

def step(self, arousal, agency_stress):
    # Stimulus: Arousal + stress vector
    stimulus = arousal + agency_stress

    # Bonds sim (random matrix)
    B = [[random.uniform(0, 1) for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]

    for i in range(self.n_nodes):
        drive = self.alpha * stimulus
        decay = -self.beta * self.states[i]
        diffusion = self.gamma * (sum(B[i][j] * self.states[j] for j in range(self.n_nodes)) / self.n_nodes - self.states[i])
        noise = random.gauss(0, 0.1)
        dS = drive + decay + diffusion + noise
        self.states[i] += self.dt * dS

    # GCL: Coherence scalar
    gcl = sum(math.tanh(s) for s in self.states) / self.n_nodes
    return gcl

