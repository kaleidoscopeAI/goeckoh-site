def __init__(self, cube_size: float = 10.0):
    self.cube_size = cube_size
    self.memory_points: Dict[int, MemoryPoint] = {}
    self.connections: List[Tuple[int, int]] = []
    self.initialize_cube()

def initialize_cube(self):
    """Initialize memory points in a 3D cube."""
    for x, y, z in product(range(int(self.cube_size)), repeat=3):
        point_id = len(self.memory_points)
        self.memory_points[point_id] = MemoryPoint(
            id=point_id, position=np.array([x, y, z])
        )

def calculate_string_tension(self, point1: MemoryPoint, point2: MemoryPoint) -> float:
    """Calculate tension between two points."""
    distance = np.linalg.norm(point1.position - point2.position)
    return np.exp(-distance / DECAY_CONSTANT)

def update_field(self, source_point: MemoryPoint, energy_state: EnergyState):
    """Update energy field for all memory points."""
    for point in self.memory_points.values():
        distance = np.linalg.norm(point.position - source_point.position)
        energy_input = energy_state.propagate(distance)
        point.memory_state.update_state(energy_input, point.tension)

def simulate_connections(self):
    """Create connections and update tensions dynamically."""
    for point_id, point in self.memory_points.items():
        neighbors = self.get_neighbors(point)
        for neighbor_id in neighbors:
            tension = self.calculate_string_tension(point, self.memory_points[neighbor_id])
            point.tension += tension
            self.connections.append((point_id, neighbor_id))

def get_neighbors(self, point: MemoryPoint) -> List[int]:
    """Find neighboring points within a unit distance."""
    neighbors = []
    for other_id, other_point in self.memory_points.items():
        if np.linalg.norm(point.position - other_point.position) <= 1.0 and point.id != other_id:
            neighbors.append(other_id)
    return neighbors

def calculate_system_tension(self) -> float:
    """Calculate total system tension as a sum of all connections."""
    total_tension = 0.0
    for point_id, point in self.memory_points.items():
        total_tension += point.tension
    return total_tension

# Example usage
network = StringNetwork(cube_size=5.0)
energy_state = EnergyState(magnitude=5.0, direction=np.array([1, 0, 0]), frequency=1.0)
source_point = network.memory_points[0]

# Update the field and simulate connections
network.update_field(source_point, energy_state)
network.simulate_connections()

# Log total system tension
total_tension = network.calculate_system_tension()
print(f"Total System Tension: {total_tension}")

