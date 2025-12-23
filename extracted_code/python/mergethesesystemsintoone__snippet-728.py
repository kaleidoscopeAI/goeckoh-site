magnitude: float
direction: np.array
frequency: float

def propagate(self, distance: float) -> float:
    """Calculate energy propagation over distance."""
    return self.magnitude * np.exp(-distance / DECAY_CONSTANT)

