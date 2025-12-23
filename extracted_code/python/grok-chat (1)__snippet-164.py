def __init__(self, n_nodes=1024):
    self.n_nodes = n_nodes
    self.emotions = [random.uniform(-1, 1) for _ in range(n_nodes)]  # Affective states
    self.dt = 0.05
    self.alpha = 1.0
    self.beta = 0.5
    self.gamma = 0.3

def step(self, stimulus):
    # Simple bonds (random for sim)
    B = [[random.uniform(0, 1) for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]

    for i in range(self.n_nodes):
        drive = self.alpha * stimulus
        decay = -self.beta * self.emotions[i]
        diffusion = self.gamma * (sum(B[i][j] * self.emotions[j] for j in range(self.n_nodes)) - self.emotions[i])
        noise = random.gauss(0, 0.1)
        dE = drive + decay + diffusion + noise
        self.emotions[i] += self.dt * dE

    return compute_gcl(self.emotions)

