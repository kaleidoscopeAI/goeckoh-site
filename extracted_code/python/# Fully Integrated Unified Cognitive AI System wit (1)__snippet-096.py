def __init__(self, node_id, data_vector=[0]*8, energy=1.0):  # Reduced dim for CPU
    super().__init__(data_vector, energy)
    self.node_id = node_id
    self.position = np.random.rand(3) * 20 - 10
    self.vel = np.random.randn(3) * 0.1
    self.awareness = np.random.rand()
    self.valence = np.random.rand()
    self.arousal = np.random.rand()
    self.isHealthy = True
    self.bit = QuantumBit(self.arousal, self.valence)

def update(self, env):
    super().update(env)
    self.position += self.vel * 0.01
    self.vel += np.random.randn(3) * 0.001
    pos_norm = np.linalg.norm(self.position)
    if pos_norm > 10:
        self.vel -= (self.position / pos_norm) * 0.01
    self.awareness = np.clip(self.awareness + np.random.randn() * 0.01, 0, 1)
    self.valence = np.clip(self.valence + np.random.randn() * 0.01, 0, 1)
    self.arousal = np.clip(self.arousal + np.random.randn() * 0.01, 0, 1)
    self.bit.real = self.arousal
    self.bit.imag = self.valence
    # Hardware feedback
    thermal_cpu = get_sensor_gradient() / 100.0 if hardware_kernel.connected else 0.25
    eta = 0.1
    self.energy += eta * thermal_cpu
    self.energy = np.clip(self.energy, 0, 1)

