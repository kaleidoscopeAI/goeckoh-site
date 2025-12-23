class NodeMetrics:
    """
    Comprehensive metrics tracking system for individual nodes.
    """

    def __init__(self, node_id: str, max_history: int = 1000):
        self.node_id = node_id
        self.creation_time = datetime.now()
        self.processing_history = deque(maxlen=max_history)
        self.total_processed_data = 0
        self.total_energy_consumed = 0.0

        # Statistical aggregates
        self.running_stats = {
            "avg_processing_time": 0.0,
            "avg_input_size": 0.0,
            "avg_output_size": 0.0,
            "avg_energy_per_task": 0.0,
            "processing_success_rate": 0.0,
        }

    def record_processing(self, input_size: int, output_size: int, energy_used: float, processing_time: float, success: bool = True):
        """Records metrics for a processing event."""
        timestamp = datetime.now()
        self.processing_history.append({
            "timestamp": timestamp,
            "input_size": input_size,
            "output_size": output_size,
            "energy_used": energy_used,
            "processing_time": processing_time,
            "success": success
        })
        self
