current_state: StateType = "calm"
arousal_history = []
max_history = 10

keywords_anxious = ["help", "can't", "scared"]
keywords_excited = ["happy", "love", "yay"]

def update_from_audio(self, audio: np.ndarray, text: str = ""):
    rms = np.sqrt(np.mean(audio**2))
    arousal = min(10.0, rms * 25)
    self.arousal_history.append(arousal)
    if len(self.arousal_history) > self.max_history:
        self.arousal_history = self.arousal_history[-self.max_history:]

    increasing_trend = len(self.arousal_history) > 2 and self.arousal_history[-1] > self.arousal_history[-2] > self.arousal_history[-3]

    if arousal > 8 and increasing_trend:
        self.current_state = "meltdown_risk"
    elif arousal > 6:
        self.current_state = "high_energy"
    elif any(k in text.lower() for k in self.keywords_anxious):
        self.current_state = "anxious"
    else:
        self.current_state = "calm"

def register(self, normalized_text: str, needs_correction: bool, rms: float) -> str:
    # Simple event detection placeholder
    if self.current_state == "meltdown_risk":
        return "meltdown_risk"
    # Add more logic as needed
    return self.current_state
