def __init__(self, id, position, energy=0.0, awareness=0.0, knowledge=0.0, emotional_state=None):
    self.id = id
    self.position = position
    self.energy = energy
    self.awareness = awareness
    self.knowledge = knowledge
    self.emotional_state = emotional_state or EmotionalState(0.0, 0.0, 0.0)
