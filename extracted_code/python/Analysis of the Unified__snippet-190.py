"""Base class for both Kaleidoscope and Mirrored engines."""

def __init__(self, num_gears: int = 5, num_paths: int = 4):
    # Initialize gear configuration
    self.gears: List[MemoryGear] = []
    self._setup_gears(num_gears)

    # Initialize logic paths
    self.logic_paths: List[LogicPath] = [LogicPath() for _ in range(num_paths)]

    # Crystallization components
    self.crystallization_chamber = CrystallizationChamber()
    self.current_insights: List[Any] = []
    self.perfect_node: Optional[Node] = None

def _setup_gears(self, num_gears: int):
    """Setup gear network with connections."""
    # Create gears with varying capacities (10 + i * 5)
    capacities = [10 + i * 5 for i in range(num_gears)]
    self.gears = [MemoryGear(weight=0, capacity=cap) for cap in capacities]

    # Create gear connections (each connects to next and previous)
    for i in range(num_gears):
        if i > 0:
            self.gears[i].connected_gears.append(self.gears[i-1])
        if i < num_gears - 1:
            self.gears[i].connected_gears.append(self.gears[i+1])

def process_node_dump(self, node_dump: List[Any]) -> List[Any]:
    """Process data dumped from a node through the mechanical system."""
    insights = []

    # Distribute data across gears
    for data in node_dump:
        data = self._filter_data(data)  # Apply engine-specific filtering

        # Add to first available gear
        for gear in self.gears:
            if len(gear.data) < gear.capacity: # Changed check from weight to len(data)
                if gear.add_data(data):
                    # If gear rotated, update logic paths
                    gear_positions = [g.position for g in self.gears]
                    for path in self.logic_paths:
                        path.shift_position(gear_positions)
                        new_insights = path.generate_insights([data])
                        insights.extend(new_insights)
                break

    self.current_insights.extend(insights)
    return insights

def crystallize(self, nodes: List[Node]) -> Node:
    """
    Crystallize current insights and nodes into a perfect representative node.
    """
    return self.crystallization_chamber.crystallize(
        nodes,
        self.current_insights,
        self.gears,
        self.logic_paths
    )

def _filter_data(self, data: Any) -> Any:
    """Engine-specific data filtering. Implemented by subclasses."""
    raise NotImplementedError

