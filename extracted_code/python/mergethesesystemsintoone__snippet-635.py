"""Represents the current state of a node."""
energy: float
health: float
tasks_completed: int
last_activity: datetime
processing_capacity: float
memory_usage: float
connections: Set[str]
state_hash: str = field(init=False)

def __post_init__(self):
    self.update_state_hash()

def update_state_hash(self):
    """Updates the state hash based on current values."""
    state_values = [
        self.energy,
        self.health,
        self.tasks_completed,
        self.processing_capacity,
        self.memory_usage
    ]
    self.state_hash = str(hash(tuple(state_values)))

