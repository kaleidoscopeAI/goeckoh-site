def __init__(self, n_nodes=1024):
    self.nodes = [Node() for _ in range(n_nodes)]
    self.dt = 0.05
    self.history = []

def step(self, stimulus):
    # ODE: Update emotions
    for node in self.nodes:
        drive = stimulus
        decay = -0.5 * sum(node.emotion)
        noise = random.gauss(0, 0.1)
        for i in range(5):
            node.emotion[i] += self.dt * (drive + decay + noise)

    # Bonds/Energy: Sim hamiltonian
    energy = sum(bond_energy(random.choice(self.nodes), random.choice(self.nodes)) for _ in range(100)) / 100

    # Life Eq for GCL
    states = [sum(n.emotion) for n in self.nodes]
    env = [random.uniform(-1, 1) for _ in range(self.n_nodes)]
    past = self.history[-1] if self.history else states
    future = [s + random.gauss(0, 0.1) for s in states]  # Sim
    n_copies = self.n_nodes + random.randint(-10, 10)  # Growth sim
    gcl = life_equation(states, env, past, future, n_copies, self.dt)

    self.history.append(states)
    if len(self.history) > 10:
        self.history = self.history[-10:]

    return gcl, energy

