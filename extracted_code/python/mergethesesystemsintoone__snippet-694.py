def __init__(self, node_id: Optional[str] = None, dna: Optional[GeneticCode] = None, parent_id: Optional[str] = None, memory_graph: Optional[MemoryGraph] = None):
    # ... existing initialization code ...
    self.memory_graph = memory_graph or MemoryGraph()

def process_data(self, data: Any):
    """
    Processes a given data unit, consuming energy and storing it in memory.
    Also, adds data and insights to the memory graph.
    """
    if self.state.energy <= 0:
        self.state.status = "Inactive"
        return False

    # Simulate data processing
    print(f"Node {self.node_id} processing: {data}")
    self.state.energy -= self.dna.energy_consumption_rate  # Consume energy
    self.state.data_processed += 1

    # Store data in memory bank
    self.memory_bank.add_data(data)

    # Add data to memory graph
    if isinstance(data, dict):
        data_id = f"data_{self.node_id}_{self.state.data_processed}"
        self.memory_graph.add_data(data_id, data)
        self.memory_graph.add_relationship(self.node_id, data_id, {"type": "processed"})

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

        # Add insight to memory graph
        insight_id = f"insight_{self.node_id}_{len(self.knowledge_base[task])}"
        self.memory_graph.add_insight(insight_id, insight)
        self.memory_graph.add_relationship(self.node_id, insight_id, {"type": "generated"})
        self.memory_graph.add_relationship(data_id, insight_id, {"type": "based_on"})

    # Update last activity time
    self.state.last_activity = time.time()
    self.state.status = "Active"

    return True

# ... other methods ...


