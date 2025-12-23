class ProcessMetrics:
    cpu_usage: float
    memory_usage: float
    io_ops: int
    priority: int

class DecisionEngine:
    def __init__(self):
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])  # cpu, mem, io, prio

    def compute_score(self, metrics: ProcessMetrics) -> float:
        values = np.array([metrics.cpu_usage, metrics.memory_usage, metrics.io_ops / 1000, metrics.priority / 10])
        return np.dot(self.weights, values)

class OmniMindState:
    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.metrics_history: Dict[str, ProcessMetrics] = {}

    def update_metrics(self, node_id: str, metrics: ProcessMetrics):
        self.metrics_history[node_id] = metrics
        score = self.decision_engine.compute_score(metrics)
        # Allocate resources based on score (simulated)
        print(f"[DA] Node {node_id} score: {score:.2f} - Allocating resources...")

