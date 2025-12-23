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
    self.total_processed_data += output_size
    self.total_energy_consumed += energy_used

    self._update_running_stats()

def _update_running_stats(self):
    """Updates running statistics based on the processing history."""
    if not self.processing_history:
        return

    successful_tasks = [p for p in self.processing_history if p["success"]]
    if successful_tasks:
        avg_processing_time = np.mean([p["processing_time"] for p in successful_tasks])
        avg_input_size = np.mean([p["input_size"] for p in successful_tasks])
        avg_output_size = np.mean([p["output_size"] for p in successful_tasks])
        avg_energy_per_task = self.total_energy_consumed / len(successful_tasks) if successful_tasks else 0.0
        processing_success_rate = len(successful_tasks) / len(self.processing_history)
    else:
        avg_processing_time = 0.0
        avg_input_size = 0.0
        avg_output_size = 0.0
        avg_energy_per_task = 0.0
        processing_success_rate = 0.0

    self.running_stats.update({
        "avg_processing_time": avg_processing_time,
        "avg_input_size": avg_input_size,
        "avg_output_size": avg_output_size,
        "avg_energy_per_task": avg_energy_per_task,
        "processing_success_rate": processing_success_rate,
    })

def get_summary(self) -> Dict[str, Any]:
    """Returns a summary of the node's metrics."""
    return {
        "node_id": self.node_id,
        "creation_time": self.creation_time.isoformat(),
        "total_processed_data": self.total_processed_data,
        "total_energy_consumed": self.total_energy_consumed,
        "running_stats": self.running_stats,
        "last_updated": datetime.now().isoformat()
    }

def export_metrics(self, format: str = "json") -> str:
    """Exports metrics in specified format."""
    metrics = self.get_summary()

    if format.lower() == "json":
        return json.dumps(metrics, indent=2)
    elif format.lower() == "csv":
        csv_lines = ["metric,value"]
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    csv_lines.append(f"{key}.{sub_key},{sub_value}")
            else:
                csv_lines.append(f"{key},{value}")
        return "\n".join(csv_lines)
    else:
        raise ValueError(f"Unsupported export format: {format}")
