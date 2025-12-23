def __init__(self, data_vector, energy=1.0):
    self.vector = np.array(data_vector, dtype=float)
    self.energy = energy
    self.tension = 0.0
    self.connections = []

def connect(self, other):
    self.connections.append(other)

def update(self, env):
    influence = np.tanh(env.temperature - self.tension)
    self.vector += influence * np.random.randn(*self.vector.shape)
    self.energy *= np.exp(-self.tension / (env.pressure + 1e-5))

def normalize(self):
    norm = np.linalg.norm(self.vector)
    if norm > 0:
        self.vector /= norm

