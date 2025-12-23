engine: pyttsx3.Engine
profile: VoiceProfile
config: VoiceCrystalConfig

def speak(self, text: str, style: Style = "neutral", mode: Mode = "outer") -> None:
    if not text:
        return

    # Style modulation: Adjust rate/volume
    rate = 150 if style == "excited" else 100 if style == "calm" else 120
    volume = 0.8 if mode == "inner" else 1.0
    self.engine.setProperty('rate', rate)
    self.engine.setProperty('volume', volume)

    # Prosody sim: No librosa, vary pauses
    sentences = text.split('.')
    for sent in sentences:
        if sent.strip():
            self.engine.say(sent.strip())
            time.sleep(0.1 if style == "calm" else 0.05)  # Simple rhythm

    self.engine.runAndWait()
