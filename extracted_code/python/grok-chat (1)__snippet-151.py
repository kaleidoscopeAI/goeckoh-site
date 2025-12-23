def __init__(self):
    self.weights = np.array([0.4, 0.3, 0.2, 0.1])  # cpu, mem, io, prio

def compute_score(self, metrics: ProcessMetrics) -> float:
    values = np.array([metrics.cpu_usage, metrics.memory_usage, metrics.io_ops / 1000, metrics.priority / 10])
    return np.dot(self.weights, values)

