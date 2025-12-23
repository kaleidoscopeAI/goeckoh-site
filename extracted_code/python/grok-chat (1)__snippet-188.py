def __init__(self, n_nodes=1024):
    self.nodes = [Node() for _ in range(n_nodes)]
    self.dt = 0.05
    self.history = []

def step(self, stimulus, text):  # Add text for entropy
    for node in self.nodes:
        drive = stimulus
        decay = -0.5 * sum(node.emotion)
        noise = random.gauss(0, 0.1)
        for i in range(5):
            node.emotion[i] += self.dt * (drive + decay + noise)

    energy = sum(bond_energy(random.choice(self.nodes), random.choice(self.nodes)) for _ in range(100)) / 100

    states = "".join(chr(int(sum(n.emotion) * 10 % 256)) for n in self.nodes)  # Sim "text" from states
    env = "".join(chr(random.randint(0, 255)) for _ in range(self.n_nodes))
    past = self.history[-1] if self.history else states
    future = text + "".join(chr(random.randint(0, 255)) for _ in range(10))  # Sim future with input
    n_copies = self.n_nodes + random.randint(-10, 10)
    gcl = life_equation(states, env, past, future, n_copies, self.dt)

    self.history.append(states)
    if len(self.history) > 10:
        self.history = self.history[-10:]

    therapy_disorder = therapy_entropy(text)  # Entropy as meltdown risk
    return gcl, energy, therapy_disorder

