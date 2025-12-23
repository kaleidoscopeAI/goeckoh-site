def __init__(self):
    self.decision_engine = DecisionEngine()
    self.metrics_history: Dict[str, ProcessMetrics] = {}

def update_metrics(self, node_id: str, metrics: ProcessMetrics):
    self.metrics_history[node_id] = metrics
    score = self.decision_engine.compute_score(metrics)
    # Allocate resources based on score (simulated)
    print(f"[DA] Node {node_id} score: {score:.2f} - Allocating resources...")

