def __init__(self, node_id: Optional[str] = None, dna: Optional[GeneticCode] = None, parent_id: Optional[str] = None):
    self.node_id = node_id or str(uuid.uuid4())
    self.parent_id = parent_id
    self.dna = dna or GeneticCode()  # Initialize with default or provided DNA
    self.birth_time = time.time()
    self.state = NodeState(energy=self.dna.initial_energy)
    self.memory = deque(maxlen=self.dna.memory_capacity)
    self.knowledge_base: Dict[str, List] = {}
    self.connections: Set[str] = set()
    self.logs = []  # For basic logging
    self.task_queue = [] # Task queue for each node

def process_data(self, data: Any):
    """Processes a given data unit, consuming energy."""
    if self.state.energy <= 0:
        self.log_event("Failed to process data: Insufficient energy.")
        self.state.status = "Inactive"
        return False

    # Simulate data processing
    print(f"Node {self.node_id} processing: {data}")
    self.state.energy -= self.dna.energy_consumption_rate  # Consume energy
    self.state.data_processed += 1

    # Generate an insight based on processed data
    if isinstance(data, dict) and "task" in data:
        task = data["task"]
        if task not in self.knowledge_base:
            self.knowledge_base[task] = []

        # Simulate insight generation
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        insight = {
            "timestamp": timestamp,
            "insight": f"Insight generated from task '{task}' at {timestamp}."
        }
        self.knowledge_base[task].append(insight)

        # Update memory usage based on data size
        self.state.memory_usage += (len(json.dumps(data)) + len(json.dumps(insight))) / 1024  # in KB

    # Store data in memory
    self.memory.append(data)

    # Update last activity time
    self.state.last_activity = time.time()
    self.state.status = "Active"

    return True

def replicate(self):
    """Replicates the node with a chance of mutation."""
    if self.state.energy >= self.dna.replication_threshold and len(self.memory) >= self.dna.min_memory_for_replication:
        new_dna = self.dna.mutate()
        child_node = Node(dna=new_dna, parent_id=self.node_id)
        child_node.state.energy = self.state.energy / 2
        self.state.energy /= 2
        self.state.last_replication = time.time()
        self.log_event(f"Node {self.node_id} replicated. New node: {child_node.node_id}")
        return child_node
    else:
        self.log_event(f"Node {self.node_id} does not meet replication criteria.")
        return None

def log_event(self, event: str):
    """Logs an event with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    self.logs.append(f"{timestamp} - {event}")

def get_status(self) -> Dict:
    """Returns the current status of the node."""
    return {
        "node_id": self.node_id,
        "parent_id": self.parent_id,
        "dna": self.dna,
        "energy": self.state.energy,
        "memory_usage": self.state.memory_usage,
        "data_processed": self.state.data_processed,
        "last_replication": self.state.last_replication,
        "status": self.state.status,
        "knowledge_base": self.knowledge_base
    }

def should_replicate(self) -> bool:
    """Determine if a node should replicate based on energy and knowledge."""
    return self.energy > 20 and len(self.knowledge_base) > 5

def receive_task(self, task: Dict[str, Any]):
    """Receives a task and adds it to the task queue."""
    self.task_queue.append(task)
    self.log_event(f"Node {self.node_id} received task: {task['id']}")


