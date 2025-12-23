def __init__(self, node_id: Optional[str] = None, dna: Optional[GeneticCode] = None, parent_id: Optional[str] = None):
    # ... existing initialization code ...
    self.memory_bank = MemoryBank(capacity=self.dna.memory_capacity)

def process_data(self, data: Any):
    """Processes a given data unit, consuming energy and storing it in memory."""
    if self.state.energy <= 0:
        self.state.status = "Inactive"
        return False

    # Simulate data processing
    print(f"Node {self.node_id} processing: {data}")
    self.state.energy -= self.dna.energy_consumption_rate  # Consume energy
    self.state.data_processed += 1

    # Store data in memory bank
    self.memory_bank.add_data(data)

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

    # Update last activity time
    self.state.last_activity = time.time()
    self.state.status = "Active"

    return True

# ... other methods ...



