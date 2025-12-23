"""Result from a quantum swarm computation task"""
task_id: str
node_id: str
result_value: float
confidence: float
timestamp: float = field(default_factory=time.time)
computation_time: float = 0.0
metadata: Dict[str, Any] = field(default_factory=dict)

