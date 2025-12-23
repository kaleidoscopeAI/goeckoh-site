from vector import Vector

class EmotionalState:
    def __init__(self, valence=0.0, arousal=0.0, coherence=0.0):
        self.valence = valence
        self.arousal = arousal
        self.coherence = coherence

class CompleteNode:
    def __init__(self, id: int, position: Vector, energy=0.0, awareness=0.0,
                 knowledge=0.0, emotional_state=None):
        self.id = id
        self.position = position
        self.energy = energy
        self.awareness = awareness
        self.knowledge = knowledge
        self.emotional_state = emotional_state or EmotionalState()
