def __init__(self):
    self.arousal = 0.0
    self.valence = 0.0
    self.coherence = 1.0
    self.state_history = deque(maxlen=100)

def update(self, audio_rms: float, text_sentiment: float = 0.0):
    # Simplified ODE for emotional dynamics
    decay = 0.95
    drive = audio_rms * 10.0

    self.arousal = (self.arousal * decay) + drive
    self.valence = (self.valence * 0.98) + text_sentiment

    # Coherence drops with high arousal (entropy increases)
    self.coherence = 1.0 / (1.0 + self.arousal * 0.1)

    self.state_history.append((self.arousal, self.valence))
    return self.arousal, self.valence

