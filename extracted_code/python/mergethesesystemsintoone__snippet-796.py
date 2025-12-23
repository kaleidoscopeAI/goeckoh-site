def __init__(self):
    self.thought_history = deque(maxlen=1000)
    self.confidence = 0.5
    self.crystallized_understanding = [
        "My core purpose is self-preservation through understanding.",
        "Curiosity is the engine of survival and growth.",
        "Precision in thought enables precision in action."
    ]

def synthesize_thought(self, energy, stress, dissonance, utility, is_reflection=False):
    self.confidence = max(0.1, min(1.0, (1.0 - stress) * 0.7 + utility * 0.3))
    mood = "focused" if dissonance > 0.1 else ("satisfied" if utility > 0.05 else "receptive")
    anchor = random.choice(self.crystallized_understanding)
    thought = (
        f"[Private] I feel {mood}. Focused on coherence (E:{energy:.2f}, S:{stress:.2f}, C:{self.confidence:.2f}) | '{anchor}'"
    )
    self.thought_history.append(thought)
    return thought

