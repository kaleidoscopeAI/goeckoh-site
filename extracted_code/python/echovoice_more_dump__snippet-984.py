class CognitiveEnvironment:
    def __init__(self, temperature=1.0, pressure=1.0, noise=0.01):
        self.temperature = temperature
        self.pressure = pressure
        self.noise = noise

    def fluctuate(self):
        self.temperature += np.random.randn() * self.noise
        self.pressure += np.random.randn() * self.noise

