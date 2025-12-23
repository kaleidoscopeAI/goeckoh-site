"""Current state of a QSIN node"""
node_id: str
status: str = "initializing"
energy: float = 0.0
last_active: float = field(default_factory=time.time)
quantum_state: Optional[QuantumState] = None
connected_nodes: Set[str] = field(default_factory=set)
entangled_nodes: Set[str] = field(default_factory=set)
processed_tasks: int = 0
successful_replications: int = 0
total_uptime: float = 0.0

def __post_init__(self):
    """Initialize with defaults if needed"""
    if self.quantum_state is None:
        self.quantum_state = QuantumState()

