def __init__(self, n_nodes=1024):
    self.n_nodes = n_nodes
    self.states = [random.uniform(-1, 1) for _ in range(n_nodes)]
    self.dt = 0.05

def step(self, arousal, latency):
    # Stimulus: Arousal from length, latency sim stress
    stimulus = arousal - latency  # Net affect

    # Update: Simplified ODE (no matrix for pure)
    for i in range(self.n_nodes):
        drive = stimulus
        decay = -0.5 * self.states[i]
        noise = random.uniform(-0.1, 0.1)
        ds = drive + decay + noise
        self.states[i] += self.dt * ds

    # GCL: Tanh mean (0-1 normalize)
    gcl = (sum(math.tanh(s) for s in self.states) / self.n_nodes + 1) / 2
    return gcl

