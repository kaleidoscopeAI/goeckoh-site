def __init__(self):
    pass

def forward(self, emotional_state) -> list:
    valence_weight = (emotional_state.valence + 1) / 2.0
    arousal_weight = emotional_state.arousal
    coherence_weight = emotional_state.coherence
    f = (valence_weight * 0.4 + arousal_weight * 0.3 + coherence_weight * 0.3)
    return [f] * 10

