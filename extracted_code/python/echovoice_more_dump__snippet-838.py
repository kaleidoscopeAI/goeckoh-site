# Enums and Dataclasses (from uin.py, organic_metrics)
class SystemState(Enum):
    INITIALIZING = "initializing"
    LEARNING = "learning"
    INTEGRATING = "integrating"
    EVOLVING = "evolving"
    MAINTAINING = "maintaining"
    EMERGENT = "emergent"
    CRITICAL = "critical"

