# ... existing methods ...

def can_handle_task(self, task_type: str) -> bool:
    """
    Checks if the node can handle a specific type of task.

    Args:
        task_type (str): The type of task.

    Returns:
        bool: True if the node can handle the task, False otherwise.
    """
    # Simple check based on DNA traits (can be made more sophisticated)
    return task_type in self.dna.traits

def process_task(self, task: Dict[str, Any]):
    """
    Processes a given task, consuming energy and updating the knowledge base.

    Args:
        task (Dict[str, Any]): The task to process.
    """
    if self.state.energy <= self.dna.energy_consumption_rate:
        self.state.status = "Inactive"
        logging.info(f"Node {self.node_id} is inactive due to low energy.")
        return

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

def _generate_insight(self, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates an insight based on the task.

    Args:
        task (Dict[str, Any]): The task processed.

    Returns:
        Dict[str, Any]: The generated insight.
    """
    # Placeholder for insight generation logic
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "insight": f"Insight for task '{task.get('type')}' generated at {timestamp} by {self.node_id}."
    }


