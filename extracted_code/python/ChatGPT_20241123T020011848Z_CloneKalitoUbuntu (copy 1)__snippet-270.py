"""Metrics used in objective score calculation"""
growth_factor: float = 0.0
energy_usage: float = 0.0
knowledge_gain: float = 0.0
total_score: float = 0.0
timestamp: float = 0.0

def __post_init__(self):
    self.timestamp = time.time()

