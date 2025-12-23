def __init__(self, position, energy=1.0):
    self.position = np.array(position)
    self.energy = energy
    self.activation = 0.0
    self.connections = []
    self.history = []

def update_activation(self, tension):
    """Update activation using hyperbolic tangent function"""
    self.activation = np.tanh(self.energy * tension)

def propagate_energy(self, decay_constant):
    """Calculate energy propagation to connected points"""
    energy_transfer = {}
    total_distance = sum(distance.euclidean(self.position, conn.position) 
                       for conn in self.connections)

    if total_distance == 0:
        return energy_transfer

    for connected_point in self.connections:
        dist = distance.euclidean(self.position, connected_point.position)
        energy = self.energy * np.exp(-decay_constant * dist)
        energy_transfer[connected_point] = energy / total_distance

    return energy_transfer

