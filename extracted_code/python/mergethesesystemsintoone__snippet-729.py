energy_level: float = 0.0
activation: float = 0.0
connections: Dict[int, float] = field(default_factory=dict)

def update_state(self, energy_input: float, tension: float):
    """Update the memory state based on energy input and tension."""
    self.energy_level = energy_input
    self.activation = np.tanh((energy_input * tension) / ACTIVATION_THRESHOLD)

