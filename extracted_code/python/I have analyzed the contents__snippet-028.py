def __init__(self):
    self.strategies = {
        "anxious": ["Let's take three deep breaths together.", "Squeeze your hands tight, then let go."],
        "high_energy": ["Let's do 5 jumping jacks!", "Push against the wall with me."],
        "meltdown": ["I am here. You are safe. Just breathe."]
    }

def evaluate(self, arousal: float, text: str) -> Optional[str]:
    if arousal > 7.0:
        return random.choice(self.strategies["meltdown"])
    if arousal > 5.0:
        return random.choice(self.strategies["high_energy"])
    if "scared" in text or "no" in text:
        return random.choice(self.strategies["anxious"])
    return None

