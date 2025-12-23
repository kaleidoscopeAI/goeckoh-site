def __init__(self, size=10, decay_constant=0.1, merge_threshold=0.8, 
             activation_threshold=0.5):
    self.size = size
    self.decay_constant = decay_constant
    self.merge_threshold = merge_threshold
    self.activation_threshold = activation_threshold
    self.memory_points = []
    self.tension_field = None
    self.string_network = nx.Graph()

def add_memory_point(self, position, energy=1.0):
    """Add new memory point to the cube"""
    point = MemoryPoint(position, energy)
    self.memory_points.append(point)
    self._update_connections()
    self._calculate_tension_field()
    return point

def _update_connections(self):
    """Update connections between memory points using minimum spanning tree"""
    if len(self.memory_points) < 2:
        return

    # Create distance matrix
    n = len(self.memory_points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = distance.euclidean(
                self.memory_points[i].position,
                self.memory_points[j].position
            )
            distances[i,j] = distances[j,i] = dist

    # Calculate minimum spanning tree
    mst = minimum_spanning_tree(distances).toarray()

    # Update connections
    self.string_network.clear()
    for i in range(n):
        self.memory_points[i].connections = []
        for j in range(n):
            if mst[i,j] > 0:
                self.memory_points[i].connections.append(self.memory_points[j])
                self.string_network.add_edge(i, j, weight=mst[i,j])

def _calculate_tension_field(self):
    """Calculate tension field across the cube"""
    if not self.memory_points:
        return

    # Initialize tension field
    grid_points = np.linspace(0, self.size, num=50)
    X, Y, Z = np.meshgrid(grid_points, grid_points, grid_points)
    tension = np.zeros_like(X)

    # Calculate tension at each grid point
    for x_idx, x in enumerate(grid_points):
        for y_idx, y in enumerate(grid_points):
            for z_idx, z in enumerate(grid_points):
                point = np.array([x, y, z])

                # Sum contributions from all memory points
                for mem_point in self.memory_points:
                    dist = distance.euclidean(point, mem_point.position)
                    if dist > 0:
                        tension[x_idx,y_idx,z_idx] += (
                            mem_point.energy * mem_point.activation / 
                            (dist * dist)
                        )

    self.tension_field = {
        'X': X, 'Y': Y, 'Z': Z, 'tension': tension
    }

def propagate_energy(self):
    """Propagate energy through the cube"""
    energy_transfers = []

    # Calculate all energy transfers
    for point in self.memory_points:
        transfers = point.propagate_energy(self.decay_constant)
        energy_transfers.append((point, transfers))

    # Apply energy transfers
    for source, transfers in energy_transfers:
        remaining_energy = source.energy
        for target, energy in transfers.items():
            target.energy += energy
            remaining_energy -= energy
        source.energy = remaining_energy

def merge_points(self):
    """Merge memory points that are too close"""
    i = 0
    while i < len(self.memory_points):
        j = i + 1
        while j < len(self.memory_points):
            dist = distance.euclidean(
                self.memory_points[i].position,
                self.memory_points[j].position
            )

            if dist < self.merge_threshold:
                # Merge points
                point1 = self.memory_points[i]
                point2 = self.memory_points[j]

                # Average position and combine energy
                new_position = (point1.position + point2.position) / 2
                new_energy = point1.energy + point2.energy

                # Create merged point
                merged_point = MemoryPoint(new_position, new_energy)
                merged_point.history = point1.history + point2.history

                # Replace points
                self.memory_points[i] = merged_point
                self.memory_points.pop(j)

                # Update connections
                self._update_connections()
            else:
                j += 1
        i += 1

def update(self):
    """Update the entire cube system"""
    # Update tension field
    self._calculate_tension_field()

    # Update point activations
    for point in self.memory_points:
        # Get local tension
        local_tension = np.interp(
            point.position, 
            [0, self.size], 
            [0, np.max(self.tension_field['tension'])]
        )
        point.update_activation(local_tension)

    # Propagate energy
    self.propagate_energy()

    # Merge close points
    self.merge_points()

def get_state(self):
    """Get current state of the cube system"""
    return {
        'points': [(p.position, p.energy, p.activation) 
                  for p in self.memory_points],
        'connections': [(i, j) for i, j in self.string_network.edges()],
        'tension_field': self.tension_field
    }import numpy as np
