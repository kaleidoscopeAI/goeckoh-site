def __init__(self, node_id: Optional[str] = None, dna: Optional[GeneticCode] = None, parent_id: Optional[str] = None, memory_graph: Optional[MemoryGraph] = None):
    # ... existing initialization code ...
    self.task_queue = deque()

def process_data(self, data: Any):
    """Processes a given data unit or task, consuming energy and storing it in memory."""
    if self.state.energy <= 0:
        self.state.status = "Inactive"
        logging.info(f"Node {self.node_id} is inactive due to low energy.")
        return False

    # Check if there are tasks in the queue
    if self.task_queue:
        task = self.task_queue.popleft()
        return self.process_task(task)

    # If no tasks, process data as before
    print(f"Node {self.node_id} processing data: {data}")
    self.state.energy -= self.dna.energy_consumption_rate
    self.state.data_processed += 1
    self.memory_bank.add_data(data)

    # ... rest of the existing data processing logic ...

    return True

def add_task_to_queue(self, task: Dict[str, Any]):
    """
    Adds a task to the node's task queue.

    Args:
        task (Dict[str, Any]): The task to add.
    """
    self.task_queue.append(task)

def process_task(self, task: Dict[str, Any]):
    """
    Processes a given task, consuming energy and updating the knowledge base.

    Args:
        task (Dict[str, Any]): The task to process.
    """
    task_type = task.get("type")
    if not self.can_handle_task(task_type):
        logging.info(f"Node {self.node_id} cannot handle task of type {task_type}.")
        return

    # Consume energy
    self.state.energy -= self.dna.energy_consumption_rate
    self.state.data_processed += 1

    # Process task and generate insight
    insight = self._generate_insight(task)
    self.knowledge_base.setdefault(task_type, []).append(insight)

    # Update memory
    self.memory_bank.add_data({"task": task, "insight": insight})

    # Update status
    self.state.last_activity = time.time()
    self.state.status = "Active"

    logging.info(f"Node {self.node_id} processed task: {task_type}")

# ... other methods ...



