class PythonNode:
    def __init__(self, node_id: int, traits: Dict[str, float], energy: float):
        self.node_id = node_id
        self.traits = traits
        self.energy = energy
        self.tasks_completed = 0
        self.memory = []

    def specialize(self, task_type: str):
        """Specialize the node for a specific task type."""
        self.task_type = task_type

    def perform_task(self):
        """Perform a task, consuming energy."""
        if self.energy > 1.0:  # Ensure a minimum energy threshold of 1.0
            self.tasks_completed += 1
            self.energy -= random.uniform(0.5, 2.0)
            task_log = f"Node {self.node_id} performed a task in {self.task_type}."
            self.memory.append(task_log)
            logging.info(task_log)
        else:
            logging.warning(f"Node {self.node_id} does not have enough energy to perform a task.")

    def dump_data(self) -> Dict:
        """Dump processed data for analysis."""
        return {
            "node_id": self.node_id,
            "data": random.random(),
            "traits": self.traits
        }

