"""
Deep Reasoning Core with GCL-based safety gating
Only allows complex operations when coherence is high
"""

def __init__(self, heart: CrystallineHeart):
    self.heart = heart
    self.current_mode = SystemMode.NORMAL

def update_mode(self, gcl: float, stress: float) -> SystemMode:
    """Determine operating mode from metrics"""
    # Compute composite safety score
    safety_score = gcl - (0.5 * stress)

    if safety_score < 0.3:
        mode = SystemMode.CRISIS
    elif safety_score < 0.5:
        mode = SystemMode.ELEVATED
    elif safety_score < 0.8:
        mode = SystemMode.NORMAL
    else:
        mode = SystemMode.FLOW

    self.current_mode = mode
    return mode

def can_execute(self, task_complexity: float) -> bool:
    """
    Gate task execution based on mode

    Args:
        task_complexity: 0.0 (trivial) to 1.0 (highly complex)
    """
    if self.current_mode == SystemMode.CRISIS:
        # Only self-preservation/calming allowed
        return task_complexity < 0.1

    elif self.current_mode == SystemMode.ELEVATED:
        # Low-risk tasks only
        return task_complexity < 0.3

    elif self.current_mode == SystemMode.NORMAL:
        # Standard operations
        return task_complexity < 0.7

    else:  # FLOW
        # Full capability
        return True

def execute_if_safe(
    self,
    task_fn: callable,
    complexity: float,
    fallback_fn: Optional[callable] = None
) -> any:
    """
    Execute task only if current mode allows

    Args:
        task_fn: The function to execute
        complexity: Task complexity score
        fallback_fn: Safe fallback if gated
    """
    if self.can_execute(complexity):
        return task_fn()
    else:
        print(f"[GATE BLOCKED] Mode={self.current_mode.value}, "
              f"Complexity={complexity:.2f}")
        if fallback_fn:
            return fallback_fn()
        return None


