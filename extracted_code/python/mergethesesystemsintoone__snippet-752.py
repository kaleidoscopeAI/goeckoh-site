def __init__(self, node_id: Optional[str] = None, energy: float = 10.0, task_load: int = 0,
             recent_performance: Optional[List[float]] = None, stress_level: float = 0.0,
             emotional_state: str = "Calm", traits: Optional[Dict[str, float]] = None,
             memory: Optional[List[Dict]] = None, relationships: Optional[List[Tuple[str, str]]] = None,
             memory_threshold: float = 5.0, c_node_ptr: Optional[ctypes.c_void_p] = None):
    super().__init__(node_id, 0, "SmartEmotionalNode", "", 0, c_node_ptr)  # Initialize base Node attributes
    self.energy = energy
    self.task_load = task_load
    self.recent_performance = recent_performance if recent_performance is not None else []
    self.stress_level = stress_level
    self.emotional_state = emotional_state
    self.traits = traits if traits is not None else {}
    self.memory = memory if memory is not None else []
    self.relationships = relationships if relationships is not None else []
    self.memory_threshold = memory_threshold

    # Load the shared library for node operations if not already loaded
    if not hasattr(SmartEmotionalNode, 'node_lib'):
        SmartEmotionalNode.node_lib = ctypes.CDLL("./c_backend/node_operations.so")
        self._setup_c_functions()

    # Initialize the node in the C backend and store the pointer
    if not self.c_node_ptr:
        self.c_node_ptr = self.node_lib.initialize_node(
            self.id.encode('utf-8'),
            ctypes.c_double(self.energy),
            "SmartEmotionalNode".encode('utf-8'),  # Assuming a default role
            "127.0.0.1".encode('utf-8'),  # Placeholder IP
            0  # Placeholder port
        )

def _setup_c_functions(self):
    """Defines argument types and return types for C functions."""
    self.node_lib.initialize_node.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    self.node_lib.initialize_node.restype = ctypes.c_void_p

    self.node_lib.assign_task.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    self.node_lib.assign_task.restype = None

    self.node_lib.update_node_status.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    self.node_lib.update_node_status.restype = None

    self.node_lib.send_heartbeat.argtypes = [ctypes.c_void_p]
    self.node_lib.send_heartbeat.restype = None

    self.node_lib.free_node.argtypes = [ctypes.c_void_p]
    self.node_lib.free_node.restype = None

def __del__(self):
    """
    Destructor to free the memory allocated to the C Node struct when the object is deleted.
    """
    if self.c_node_ptr:
        self.node_lib.free_node(self.c_node_ptr)

def calculate_stress(self):
    """
    Calculate the stress level based on task load, energy levels, and recent performance.
    """
    task_factor = self.task_load / 10.0  # Normalize task load to a factor out of 10
    energy_factor = (10.0 - self.energy) / 10.0  # Invert energy level to represent stress
    performance_factor = 1.0 - np.mean(self.recent_performance) if self.recent_performance else 1.0  # Use 1.0 as default if no performance data

    # Calculate stress level, ensuring it is clipped between 0 and 1
    self.stress_level = np.clip(
        task_factor * 0.4 + energy_factor * 0.4 + performance_factor * 0.2,
        0.0,
        1.0
    )

def update_emotional_state(self):
    """
    Update the emotional state of the node based on its current stress level.
    """
    if self.stress_level < 0.3:
        self.emotional_state = "Calm"
    elif self.stress_level < 0.6:
        self.emotional_state = "Alert"
    elif self.stress_level < 0.8:
        self.emotional_state = "Anxious"
    else:
        self.emotional_state = "Overwhelmed"

def adjust_memory_threshold(self, cpu_intensity: float, task_complexity: float):
    """
    Dynamically adjust the memory threshold based on CPU intensity and task complexity.
    """
    self.memory_threshold = np.clip(
        self.memory_threshold + (cpu_intensity * 0.1 - task_complexity * 0.05),
        1.0,
        10.0
    )

def dump_memory_to_kaleidoscope(self):
    """
    Dump memory to the Kaleidoscope Engine when the memory threshold is reached.
    """
    if len(self.memory) >= self.memory_threshold:
        print(f"Dumping memory from Node {self.id} to Kaleidoscope Engine: {len(self.memory)} items.")
        # In a real scenario, you would send this data to the Kaleidoscope Engine
        self.memory.clear()  # Clear memory after dumping

def collect_data(self, data: Dict, cpu_intensity: float):
    """
    Collect data and metadata, adjust memory threshold, and dump memory if necessary.
    """
    self.memory.append({
        "data": data,
        "metadata": {
            "size": len(data),
            "type": type(data).__name__,
            "timestamp": datetime.now().isoformat()
        }
    })
    self.adjust_memory_threshold(cpu_intensity, len(self.memory))
    self.dump_memory_to_kaleidoscope()

def process_task(self, task_complexity: float):
    """
    Process a task, affecting energy and stress levels based on task complexity and traits.
    """
    # Call the C function to assign a task
    task_json = json.dumps({"task_id": str(uuid.uuid4()), "complexity": task_complexity})
    self.node_lib.assign_task(self.c_node_ptr, task_json.encode('utf-8'))

    self.energy -= task_complexity * self.traits.get("energy_efficiency", 1.0)
    self.task_load = max(0, self.task_load - 1)
    self.calculate_stress()
    self.update_emotional_state()

def add_performance_record(self, success: bool):
    """
    Add a performance record to the recent performance list.
    """
    self.recent_performance.append(1.0 if success else 0.0)
    if len(self.recent_performance) > 10:
        self.recent_performance.pop(0)  # Maintain a sliding window of 10 entries
    self.calculate_stress()
    self.update_emotional_state()

def should_replicate(self) -> bool:
    """
    Determine if the node should replicate based on memory and stress level.
    """
    return len(self.memory) >= self.memory_threshold and self.stress_level < 0.5

def replicate(self) -> Optional["SmartEmotionalNode"]:
    """
    Replicate the node, creating a new node with mutated traits.
    """
    if not self.should_replicate():
        return None

    new_traits = {
        key: max(0.01, value + np.random.normal(0, 0.05))
        for key, value in self.traits.items()
    }
    self.energy /= 2
    new_node = SmartEmotionalNode(
        id=str(uuid.uuid4()),
        energy=self.energy,
        traits=new_traits,
        memory=self.memory[-3:],  # Inherit the last 3 memories
        relationships=self.relationships.copy(),
        memory_threshold=self.memory_threshold
    )
    return new_node

def visualize_state(self):
    """
    Visualize the node's current state using a bar chart.
    """
    plt.figure(figsize=(10, 6))
    metrics = {
        "Stress Level": self.stress_level,
        "Energy Level": self.energy,
        "Memory Utilization": len(self.memory) / self.memory_threshold
    }
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
    plt.title(f"Node {self.id} State Visualization")
    plt.ylim(0, 1.2)  # Set y-axis limit to accommodate metrics exceeding 1.0
    plt.ylabel("Normalized Metrics")
    plt.show()

def get_state(self) -> dict:
    """Return the current state of the node."""
    return {
        'node_id': self.node_id,
        'capacity': self.capacity,
        'role': self.role,
        'status': self.status,
        'last_heartbeat': self.last_heartbeat,
        'tasks': self.tasks,
        'ip_address': self.ip_address,
        'port': self.port,
        'energy': self.energy,
        'task_load': self.task_load,
        'recent_performance': self.recent_performance,
        'stress_level': self.stress_level,
        'emotional_state': self.emotional_state,
        'traits': self.traits,
        'memory': self.memory,
        'relationships': self.relationships,
        'memory_threshold': self.memory_threshold
    }

def communicate(self, target_node: 'SmartEmotionalNode', message: Dict):
    """Send a message to another node."""
    if target_node.node_id in [relationship[0] for relationship in self.relationships]:
        message_data = {
            'sender_id': self.node_id,
            'recipient_id': target_node.node_id,
            'timestamp': time.time(),
            'message': message
        }
        target_node.receive_message(message_data)
    else:
        raise ValueError(f"Target node {target_node.node_id} is not a known relationship.")

def receive_message(self, message_data: Dict):
    """Receive and process a message from another node."""
    self.short_term_memory.append(message_data)
    # Process the message content as needed
    if message_data['message'].get('type') == 'data_exchange':
        self.process_data_exchange(message_data)
    elif message_data['message'].get('type') == 'task_delegation':
        self.process_task_delegation(message_data)
    # Add more message types as needed

def process_data_exchange(self, message_data: Dict):
    """
    Process data exchange messages from another node.
    """
    data = message_data['message'].get('data')
    if data:
        print(f"Node {self.node_id} received data: {data}")
        # Implement data processing logic here

def process_task_delegation(self, message_data: Dict):
    """
    Process task delegation messages from another node.
    """
    task = message_data['message'].get('task')
    if task:
        print(f"Node {self.node_id} received task: {task}")
        # Implement task acceptance and processing logic here

def negotiate_resources(self, neighboring_nodes: List["SmartEmotionalNode"]):
    """
    Negotiate resource allocation with neighboring nodes based on stress levels and emotional states.
    """
    if self.stress_level > 0.7:
        for neighbor in neighboring_nodes:
            if neighbor.energy > self.energy:
                transfer_amount = min(neighbor.energy * 0.1, self.energy * 0.2)
                self.energy += transfer_amount
                neighbor.energy -= transfer_amount
                print(f"Node {self.id} received energy from Node {neighbor.id}, amount: {transfer_amount:.2f}")
                break

def adjust_behavior_based_on_insights(self, insights: List[Dict]):
    """
    Adjust the node's behavior based on insights from the Kaleidoscope Engine.
    """
    for insight in insights:
        if insight['type'] == 'resource_optimization':
            if 'energy_efficiency' in insight['data']:
                self.traits['energy_efficiency'] = insight['data']['energy_efficiency']
                print(f"Node {self.id} adjusted energy efficiency to {self.traits['energy_efficiency']:.2f}")

        elif insight['type'] == 'task_prioritization':
            if 'task_types' in insight['data']:
                self.prioritize_tasks(insight['data']['task_types'])

def prioritize_tasks(self, task_types: List[str]):
    """
    Adjust task priorities based on insight recommendations.
    """
    for task in self.tasks:
        if task['type'] in task_types:
            task['priority'] = 1  # High priority
        else:
            task['priority'] = 0  # Default priority
    self.tasks.sort(key=lambda x: x['priority'], reverse=True)
    print(f"Node {self.id} adjusted task priorities.")

def analyze_data(self):
    """
    Analyze data in the node's memory to potentially update traits or behaviors.
    """
    for entry in self.memory:
        data = entry["data"]
        if isinstance(data, dict):
            if data.get("type") == "text" and "keyword" in data.get("content", ""):
                self.traits["learning_rate"] = min(1.0, self.traits.get("learning_rate", 0.5) + 0.01)
                print(f"Node {self.id} increased learning rate to {self.traits['learning_rate']:.2f}")
            elif data.get("type") == "numerical" and data.get("value", 0) > 50:
                self.traits["energy_efficiency"] = max(0.01, self.traits.get("energy_efficiency", 1.0) - 0.01)
                print(f"Node {self.id} decreased energy efficiency to {self.traits['energy_efficiency']:.2f}")

def emotional_reaction(self, emotion: str, intensity: float):
    """
    Modify perspective based on an emotional reaction.
    """
    if emotion == "Anxious":
        # Example: Reduce weights of dimensions associated with risk
        self.weights *= (1 - intensity)
    elif emotion == "Confident":
        # Example: Increase weights of dimensions associated with success
        self.weights *= (1 + intensity)
    # Add more emotional reactions as needed

def final_operations_before_recycling(self):


