from dataclasses import dataclass
from enum import Enum

class SystemState(Enum):
    INITIALIZING = "initializing"
    LEARNING = "learning"
    INTEGRATING = "integrating"
    EVOLVING = "evolving"
    MAINTAINING = "maintaining"
    EMERGENT = "emergent"
    CRITICAL = "critical"

@dataclass
class OrganicMetrics:
    health: float = 1.0
    coherence: float = 0.0
    complexity: float = 0.0
    adaptability: float = 0.0
    emergence_level: float = 0.0
    energy_efficiency: float = 1.0
    learning_rate: float = 0.0
    integration_density: float = 0.0
