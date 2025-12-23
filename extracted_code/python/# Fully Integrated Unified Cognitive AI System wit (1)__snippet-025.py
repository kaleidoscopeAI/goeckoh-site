import numpy as np
import math
import ctypes
import time

class CustomRandom:
    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def randint(self, low, high):
        return self._rng.integers(low, high)

    def uniform(self, low, high):
        return self._rng.uniform(low, high)

class Vector:
    def __init__(self, components):
        self.components = np.array(components, dtype=np.float64)

    def __sub__(self, other):
        return Vector(self.components - other.components)

    def __mul__(self, scalar):
        return Vector(self.components * scalar)

    def dot(self, other):
        return np.dot(self.components, other.components)

    def norm(self):
        return np.linalg.norm(self.components)

# Dummy redact_pii
def redact_pii(text: str) -> str:
    # Remove or mask personally identifiable information
    return text

class CompleteNode:
    def __init__(self, id, position, energy=0.0, awareness=0.0, knowledge=0.0, emotional_state=None):
        self.id = id
        self.position = position
        self.energy = energy
        self.awareness = awareness
        self.knowledge = knowledge
        self.emotional_state = emotional_state or EmotionalState(0.0, 0.0, 0.0)
