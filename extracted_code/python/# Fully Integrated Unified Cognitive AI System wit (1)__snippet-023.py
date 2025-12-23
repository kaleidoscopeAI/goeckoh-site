from typing import List

class Vector:
    def __init__(self, components: List[float]):
        self.components = components

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar: float) -> "Vector":
        return Vector([a * scalar for a in self.components])

    def dot(self, other: "Vector") -> float:
        return sum(a * b for a, b in zip(self.components, other.components))

    def norm(self) -> float:
        return sum(a * a for a in self.components) ** 0.5

class EmotionalState:
    def __init__(self, valence: float, arousal: float, coherence: float):
        self.valence = valence
        self.arousal = arousal
        self.coherence = coherence

class CompleteNode:
    def __init__(self, id: int, position: Vector, energy: float, awareness: float,
                 knowledge: float, emotional_state: EmotionalState):
        self.id = id
        self.position = position
        self.energy = energy
        self.awareness = awareness
        self.knowledge = knowledge
        self.emotional_state = emotional_state
